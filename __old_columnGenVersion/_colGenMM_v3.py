from init_project import *
#
from gurobipy import *
from datetime import datetime
import numpy as np
from time import clock
#
from _utils.recording import record_log, O_GL, X_GL
from _utils.mm_utils import get_routeFromOri


def run(problem, log_fpath=None, numThreads=None, TimeLimit=None):
    startTime = clock()
    #
    # Solve a master problem
    #
    bB, \
    T, r_i, v_i, _lambda, P, D, N, \
    K, w_k, t_ij, _delta = problem
    input4subProblem = [T, r_i, v_i, _lambda, P, D, N, K, w_k, t_ij, _delta]
    #
    # generate initial bundles
    #
    _e_bi = [[0 for _ in range(len(T))] for _ in range(bB)]
    for i in T:
        _e_bi[i % bB][i] = 1
    colums = set()
    e_bi = []
    B = []
    for l in _e_bi:
        if sum(l) == 0:
            continue
        e_bi.append(l)
        colums.add(tuple(l))
        B.append([i for i, v in enumerate(l) if v != 0])
    p_b = []
    for b in range(len(B)):
        bundle = [i for i, v in enumerate(e_bi[b]) if v == 1]
        p = 0
        if bundle:
            br = sum([r_i[i] for i in bundle])
            for k, w in enumerate(w_k):
                if minTimePD(bundle, k, t_ij) < _delta:
                    p += w * br
        p_b.append(p)
    #
    logContents = 'Initial bundles\n'
    for b in B:
        logContents += '\t %s\n' % str(b)
    if log_fpath:
        with open(log_fpath, 'wt') as f:
            f.write(logContents)
    else:
        print(logContents)
    #
    # Define decision variables
    #
    m = Model('materProblem')
    q_b = {}
    for b in range(len(B)):
        q_b[b] = m.addVar(vtype=GRB.BINARY, name="q[%d]" % b)
    m.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for b in range(len(B)):
        obj += p_b[b] * q_b[b]
    m.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    #
    taskAC = {}
    for i in T:  # eq:taskA
        taskAC[i] = m.addConstr(quicksum(e_bi[b][i] * q_b[b] for b in range(len(B))) == 1, name="taskAC[%d]" % i)
    numBC = m.addConstr(quicksum(q_b[b] for b in range(len(B))) == bB, name="numBC")
    m.update()
    #
    counter = 0
    while True:
        if len(B) == len(T) ** 2 - 1:
            break
        counter += 1
        relax = m.relax()
        relax.Params.OutputFlag = X_GL
        relax.optimize()
        #
        pi_i = [relax.getConstrByName("taskAC[%d]" % i).Pi for i in T]
        mu = relax.getConstrByName("numBC").Pi
        c_b, bundle = subProblem(pi_i, mu, B, input4subProblem, counter, log_fpath, numThreads, TimeLimit)
        if c_b == None:
            break
        vec = [0 for _ in range(len(T))]
        logContents = '\n\n'
        logContents += '%dth iteration (%s)\n' % (counter, str(datetime.now()))
        logContents += '\t Dual V\n'
        logContents += '\t\t Pi: %s\n' % str([round(v, 3) for v in pi_i])
        logContents += '\t\t mu: %.3f\n' % mu
        logContents += '\t new B. %s\n' % str(bundle)
        logContents += '\t red. C. %.3f\n' % c_b
        record_log(log_fpath, logContents)
        for i in bundle:
            vec[i] = 1
        p = c_b + (np.array(vec) * np.array(pi_i)).sum() + mu
        if c_b <= 0:
            break
        e_bi.append(vec)
        colums.add(tuple(vec))
        p_b.append(p)
        #
        col = Column()
        for i in range(len(T)):
            if e_bi[len(B)][i] > 0:
                col.addTerms(e_bi[len(B)][i], taskAC[i])
        col.addTerms(1, numBC)
        #
        q_b[len(B)] = m.addVar(obj=p_b[len(B)], vtype=GRB.BINARY, name="q[%d]" % len(B), column=col)
        B.append(bundle)
        m.update()
    #
    # Settings
    #
    if TimeLimit is not None:
        m.setParam('TimeLimit', TimeLimit)
    if numThreads is not None:
        m.setParam('Threads', numThreads)
    if log_fpath is not None:
        m.setParam('LogFile', log_fpath)
    m.setParam('OutputFlag', X_GL)
    #
    # Run Gurobi (Optimization)
    #
    m.optimize()
    endTime = clock()
    eliTime = endTime - startTime
    chosenB = [B[b] for b in range(len(B)) if q_b[b].x > 0.5]
    #
    logContents = '\n\n'
    logContents += 'Summary\n'
    logContents += '\t Sta.Time: %s\n' % str(startTime)
    logContents += '\t End.Time: %s\n' % str(endTime)
    logContents += '\t Eli.Time: %d\n' % eliTime
    logContents += '\t ObjV: %.3f\n' % m.objVal
    logContents += '\t chosen B.: %s\n' % str(chosenB)
    record_log(log_fpath, logContents)
    #
    return m.objVal, eliTime


def minTimePD(b, k, t_ij, log_fpath=None, numThreads=None, TimeLimit=None):
    def addSubTourElimC_minPD(m, where):
        if where == GRB.callback.MIPSOL:
            curRoute = []
            for i in m._N:
                for j in m._N:
                    if i == j:
                        continue
                    if m.cbGetSolution(m._x_ij[i, j]) > 0.5:
                        curRoute.append((i, j))
            route = get_routeFromOri(curRoute, m._N)
            expr = 0
            if len(route) != len(m._N) - 1:
                for i, j in route:
                    expr += m._x_ij[i, j]
                m.cbLazy(expr <= len(route) - 1)  # eq:subTourElim
    #
    _kP, _kM = 'ori%d' % k, 'dest%d' % k
    N = {_kP, _kM}
    P, D = set(), set()
    for i in b:
        P.add('p0%d' % i); D.add('d%d' % i)
    N = N.union(P)
    N = N.union(D)
    #
    # Define decision variables
    #
    m = Model('minTimePD')
    x_ij, o_i = {}, {}
    for i in N:
        o_i[i] = m.addVar(vtype=GRB.INTEGER, name='o[%s]' % i)
        for j in N:
            x_ij[i, j] = m.addVar(vtype=GRB.BINARY, name='x[%s,%s]' % (i, j))
    m.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for i in N:
        for j in N:
            obj += t_ij[i, j] * x_ij[i, j]
    obj -= t_ij[_kM, _kP]
    obj -= t_ij[_kP, _kM]
    m.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constrains
    #
    # Flow based routing
    #  eq:pathPD
    m.addConstr(x_ij[_kM, _kP] == 1,
                name='cf')
    m.addConstr(quicksum(x_ij[_kP, j] for j in P) == 1,
                name='pPDo')
    m.addConstr(quicksum(x_ij[i, _kM] for i in D) == 1,
                name='pPDi')
    #  # eq:XpathPD
    m.addConstr(quicksum(x_ij[_kP, j] for j in D) == 0,
                name='XpPDo')
    m.addConstr(quicksum(x_ij[i, _kM] for i in P) == 0,
                name='XpPDi')
    #
    for i in N:  # eq:outFlow
        if i == _kM: continue
        m.addConstr(quicksum(x_ij[i, j] for j in N if j != _kP) == 1,
                    name='tOF[%s]' % i)
    for j in N:  # eq:inFlow
        if j == _kP: continue
        m.addConstr(quicksum(x_ij[i, j] for i in N if i != _kM) == 1,
                    name='tIF[%s]' % j)
    for i in N:  # eq:XselfFlow
        m.addConstr(x_ij[i, i] == 0,
                    name='XsF[%s]' % i)
    for i in N:
        for j in N:  # eq:direction
            m.addConstr(x_ij[i, j] + x_ij[j, i] <= 1,
                        name='dir[%s,%s]' % (i, j))
    #  # eq:initOrder
    m.addConstr(o_i[_kP] == 1,
                name='ordOri')
    m.addConstr(o_i[_kM] == len(N),
                name='ordDest')
    for i in b:  # eq:pdSequnce
        m.addConstr(o_i['p0%d' % i] <= o_i['d%d' % i], name='ord[%s]' % i)
    for i in N:
        for j in N:  # eq:ordering
            if i == _kM or j == _kP:
                continue
            m.addConstr(o_i[i] + 1 <= o_i[j] + len(N) * (1 - x_ij[i, j]),
                        name='ord[%s,%s]' % (i, j))
    #
    # For callback function
    #
    m._N = N
    m._x_ij = x_ij
    m.params.LazyConstraints = 1
    #
    # Settings
    #
    if TimeLimit is not None:
        m.setParam('TimeLimit', TimeLimit)
    if numThreads is not None:
        m.setParam('Threads', numThreads)
    if log_fpath is not None:
        m.setParam('LogFile', log_fpath)
    m.Params.OutputFlag = X_GL
    #
    # Run Gurobi (Optimization)
    #
    m.optimize(addSubTourElimC_minPD)
    #
    return m.objVal


def subProblem(pi_i, mu, B, input4subProblem, counter, log_fpath=None, numThreads=None, TimeLimit=None):
    def addLazyC_subProblem(m, where):
        if where == GRB.callback.MIPSOL:
            tNodes = []
            selectedTasks = set()
            for i in m._T:
                if m.cbGetSolution(m._z_i[i]) > 0.5:
                    tNodes.append('p0%d' % i); tNodes.append('d%d' % i)
                    selectedTasks.add(i)
            #
            for b in m._B:
                if len(b) == len(selectedTasks):
                    m.cbLazy(quicksum(m._z_i[i] for i in b) <= len(b) - 1)
            #
            for k in m._K:
                ptNodes = tNodes[:] + ['ori%d' % k, 'dest%d' % k]
                selectedEdges = [(i, j) for j in ptNodes for i in ptNodes if m.cbGetSolution(m._x_kij[k, i, j]) > 0.5]
                route = get_routeFromOri(selectedEdges, ptNodes)
                if len(route) != len(ptNodes) - 1:
                    expr = 0
                    for i, j in route:
                        expr += m._x_kij[k, i, j]
                    m.cbLazy(expr <= len(route) - 1)  # eq:subTourElim
    #
    T, r_i, v_i, _lambda, P, D, N, K, w_k, t_ij, _delta = input4subProblem
    bigM1 = sum(r_i) * 2
    bigM2 = (len(T) + 1) * 2
    bigM3 = sum(t_ij.values())
    #
    # Define decision variables
    #
    m = Model('subProblem %d' % counter)
    z_i, y_k, a_k, o_ki, x_kij = {}, {}, {}, {}, {}
    for i in T:
        z_i[i] = m.addVar(vtype=GRB.BINARY, name='z[%d]' % i)
    for k in K:
        y_k[k] = m.addVar(vtype=GRB.BINARY, name='y[%d]' % k)
        a_k[k] = m.addVar(vtype=GRB.CONTINUOUS, name='a[%d]' % k)
    for k in K:
        kN = N.union({'ori%d' % k, 'dest%d' % k})
        for i in kN:
            o_ki[k, i] = m.addVar(vtype=GRB.INTEGER, name='o[%d,%s]' % (k, i))
            for j in kN:
                x_kij[k, i, j] = m.addVar(vtype=GRB.BINARY, name='x[%d,%s,%s]' % (k, i, j))
    m.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for k in K:
        obj += w_k[k] * a_k[k]
    for i in T:
        obj -= pi_i[i] * z_i[i]
    obj -= mu
    m.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    # Linearization
    for k in K:  # eq:linAlpha
        m.addConstr(a_k[k] >= quicksum(r_i[i] * z_i[i] for i in T) - bigM1 * (1 - y_k[k]),
                    name='la1[%d]' % k)
        m.addConstr(a_k[k] <= bigM1 * y_k[k],
                    name='la2[%d]' % k)
        m.addConstr(a_k[k] <= quicksum(r_i[i] * z_i[i] for i in T),
                    name='la3[%d]' % k)
    #
    # Volume
    m.addConstr(quicksum(v_i[i] * z_i[i] for i in T) <= _lambda,
                name='vt')  # eq:volTh
    #
    # Flow based routing
    for k in K:
        #  # eq:circularF
        m.addConstr(x_kij[k, 'dest%d' % k, 'ori%d' % k] == 1,
                    name='cf[%d]' % k)
        #  # eq:pathPD
        m.addConstr(quicksum(x_kij[k, 'ori%d' % k, j] for j in P) == 1,
                    name='pPDo[%d]' % k)
        m.addConstr(quicksum(x_kij[k, i, 'dest%d' % k] for i in D) == 1,
                    name='pPDi[%d]' % k)
        #  # eq:XpathPD
        m.addConstr(quicksum(x_kij[k, 'ori%d' % k, j] for j in D) == 0,
                    name='XpPDo[%d]' % k)
        m.addConstr(quicksum(x_kij[k, i, 'dest%d' % k] for i in P) == 0,
                    name='XpPDi[%d]' % k)
        #
        for i in T:  # eq:taskOutFlow
            m.addConstr(quicksum(x_kij[k, 'p0%d' % i, j] for j in N) == z_i[i],
                        name='tOF[%d,%d]' % (k, i))
        for j in T:  # eq:taskInFlow
            m.addConstr(quicksum(x_kij[k, i, 'd%d' % j] for i in N) == z_i[j],
                        name='tIF[%d,%d]' % (k, j))
        #
        kP, kM = 'ori%d' % k, 'dest%d' % k
        kN = N.union({kP, kM})
        for i in kN:  # eq:XselfFlow
            m.addConstr(x_kij[k, i, i] == 0,
                        name='Xsf[%d,%s]' % (k, i))
        for j in kN:  # eq:flowCon
            m.addConstr(quicksum(x_kij[k, i, j] for i in kN) == quicksum(x_kij[k, j, i] for i in kN),
                        name='fc[%d,%s]' % (k, j))
        for i in kN:
            for j in kN:  # eq:direction
                if i == j: continue
                m.addConstr(x_kij[k, i, j] + x_kij[k, j, i] <= 1,
                            name='dir[%d,%s,%s]' % (k, i, j))
        #
        m.addConstr(o_ki[k, kP] == 1,
                    name='iOF[%d]' % k)
        m.addConstr(o_ki[k, kM] <= bigM2,
                    name='iOE[%d]' % k)
        for i in T:  # eq:pdSequnce
            m.addConstr(o_ki[k, 'p0%d' % i] <= o_ki[k, 'd%d' % i] + bigM2 * (1 - z_i[i]),
                        name='pdS[%d]' % k)
        for i in kN:
            for j in kN:  # eq:ordering
                if i == j: continue
                if i == kM or j == kP: continue
                m.addConstr(o_ki[k, i] + 1 <= o_ki[k, j] + bigM2 * (1 - x_kij[k, i, j]))
        #
        # Feasibility
        #  # eq:pathFeasibility
        m.addConstr(quicksum(t_ij[i, j] * x_kij[k, i, j] for i in kN for j in kN if not (i == kM and j == kP)) \
                    - t_ij[kP, kM] - _delta <= bigM3 * (1 - y_k[k]),
                    name='pf[%d]' % k)
    #
    # For callback function
    #
    m._B, m._T, m._K = B, T, K
    m._z_i, m._x_kij = z_i, x_kij
    m.params.LazyConstraints = 1
    #
    # Settings
    #
    if TimeLimit is not None:
        m.setParam('TimeLimit', TimeLimit)
    if numThreads is not None:
        m.setParam('Threads', numThreads)
    if log_fpath is not None:
        m.setParam('LogFile', log_fpath)
    m.Params.OutputFlag = O_GL
    #
    # Run Gurobi (Optimization)
    #
    m.optimize(addLazyC_subProblem)
    if m.status == GRB.Status.OPTIMAL:
        return m.objVal, [i for i in T if z_i[i].x > 0.05]
    elif m.status == GRB.Status.INFEASIBLE:
        return None, None
    else:
        assert False


if __name__ == '__main__':
    from problems import *
    from time import clock, time
    cSTime, cTTime = clock(), time()
    run(convert_p2i(*ex1()))
    print(clock() - cSTime, time() - cTTime)
