from init_project import *
#
from gurobipy import *
import numpy as np
#
from cpuinfo import get_cpu_info
from datetime import datetime

startTime = datetime.now()
print('Start time: %s' % str(startTime))

SUB_SUB_LOGGING = False

TimeLimit = 60 * 60
numThreads = 1


def run(problem):
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
    print('Initial bundles')
    for b in B:
        print('\t', b)
    m = Model('materProblem')
    ofpath = opath.join(dpath['gurobiLog'], 'masterProblem.log')
    m.setParam('LogFile', ofpath)
    #
    # Define decision variables
    #
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
    #
    m.update()  # must update before calling relax()
    m.write('initPM.lp')
    counter = 0
    while True:
        if len(B) == len(T) ** 2 - 1:
            break
        counter += 1
        relax = m.relax()
        # relax.write("RLPM_%dth.lp" % counter)
        relax.Params.OutputFlag = SUB_SUB_LOGGING
        relax.optimize()
        #
        pi_i = [relax.getConstrByName("taskAC[%d]" % i).Pi for i in T]
        mu = relax.getConstrByName("numBC").Pi
        c_b, bundle = subProblem(pi_i, mu, B, input4subProblem)
        if c_b == None:
            break
        vec = [0 for _ in range(len(T))]
        print('', end='\n')
        print('%dth iteration' % counter, '(%s)' % str(datetime.now()))
        print('\t Dual V: Pi', [round(v, 3) for v in pi_i], 'mu', mu)
        print('\t new Bundle:', bundle)
        print('\t reduced C.: ', round(c_b, 3))
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
    m.Params.OutputFlag = SUB_SUB_LOGGING
    m.write('finalPM.lp')
    m.optimize()
    endTime = datetime.now()
    print('End time: %s' % str(endTime))
    #
    print('', end='\n')
    print('Summary')
    print('\t Elapsed time: %s' % str(endTime - startTime))
    print('\t ObjV:', m.objVal)
    print('\t Chosen Bundle:', [B[b] for b in range(len(B)) if q_b[b].x > 0.5], [q_b[b].x for b in range(len(B))])

    return m.objVal, (endTime - startTime).seconds


def subProblem(pi_i, mu, B, input4subProblem):
    T, r_i, v_i, _lambda, P, D, N, K, w_k, t_ij, _delta = input4subProblem
    bigM1 = sum(t_ij.values())
    bigM2 = sum(r_i) * 2
    #
    m = Model('subProblem')
    m.Params.OutputFlag = False
    ofpath = opath.join(dpath['gurobiLog'], 'subProblem.log')
    m.setParam('LogFile', ofpath)
    #
    # Define decision variables
    #
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
    #
    # Linearization
    #
    for k in K:  # eq:linAlpha
        m.addConstr(a_k[k] >= quicksum(r_i[i] * z_i[i] for i in T) - bigM2 * (1 - y_k[k]), name='la1[%d]' % k)
        m.addConstr(a_k[k] <= bigM2 * y_k[k], name='la2[%d]' % k)
        m.addConstr(a_k[k] <= quicksum(r_i[i] * z_i[i] for i in T), name='la3[%d]' % k)
    #
    # Volume
    #
    m.addConstr(quicksum(v_i[i] * z_i[i] for i in T) <= _lambda, name='volTH')  # eq:volTh
    #
    # Flow based routing
    for k in K:
        # m.addConstr(x_kij[k, 'dest%d' % k, 'ori%d' % k] == 1)
        #  # eq:pathPD
        m.addConstr(quicksum(x_kij[k, 'ori%d' % k, j] for j in P) == 1, name='pPDo[%d]' % k)
        m.addConstr(quicksum(x_kij[k, i, 'dest%d' % k] for i in D) == 1, name='pPDi[%d]' % k)
        #  # eq:XpathPD
        m.addConstr(quicksum(x_kij[k, 'ori%d' % k, j] for j in D) == 0, name='XpPDo[%d]' % k)
        m.addConstr(quicksum(x_kij[k, i, 'dest%d' % k] for i in P) == 0, name='XpPDi[%d]' % k)
        #
        for i in T:  # eq:taskOutFlow
            m.addConstr(quicksum(x_kij[k, 'p0%d' % i, j] for j in N) == z_i[i], name='tOF[%d,%d]' % (k, i))
        for j in T:  # eq:taskInFlow
            m.addConstr(quicksum(x_kij[k, i, 'd%d' % j] for i in N) == z_i[j], name='tIF[%d,%d]' % (k, j))
        kN = N.union({'ori%d' % k, 'dest%d' % k})
        for i in kN:  # eq:XselfFlow
            m.addConstr(x_kij[k, i, i] == 0, name='Xsf[%d,%s]' % (k, i))
        for j in kN:  # eq:flowCon
            m.addConstr(quicksum(x_kij[k, i, j] for i in kN) == quicksum(x_kij[k, j, i] for i in kN),
                        name='fc[%d,%s]' % (k, j))
        for i in kN:
            for j in kN:  # eq:direction
                if i == j: continue
                m.addConstr(x_kij[k, i, j] + x_kij[k, j, i] <= 1,
                            name='dir[%d,%s,%s]' % (k, i, j))
    #
    # Feasibility
    #
    for k in K:
        kP, kM = 'ori%d' % k, 'dest%d' % k
        kN = N.union({kP, kM})
        #  # eq:pathFeasiblity
        m.addConstr(quicksum(t_ij[i, j] * x_kij[k, i, j] for i in kN for j in kN) - t_ij[kP, kM] - _delta <= bigM1 * (1 - y_k[k]),
                    name='pf[%d]' % k)
    #
    # For callback function
    #
    m._B = B
    m._T, m._K  = T, K
    m._z_i, m._x_kij, m._o_ki = z_i, x_kij, o_ki
    m.params.LazyConstraints = 1
    #
    # Optimization
    #
    m.write('subProblem.lp')
    m.optimize(addLazyC_subProblem)
    #
    # if m.status != GRB.Status.OPTIMAL:
    #     m.computeIIS()
    #     m.write('subProblem.ilp')
    #     m.write('subProblem.lp')
    #
    # print(m.status)
    if m.status == GRB.Status.OPTIMAL:
        print()
        for k in K:
            print('k%d' % (k), end='\n\t')
            edges = []
            kN = N.union({'ori%d' % k, 'dest%d' % k})
            for i in kN:
                for j in kN:
                    if x_kij[k, i, j].x > 0.05:
                        edges.append((i, j))
            print(route_display(edges))
        return m.objVal, [i for i in T if z_i[i].x > 0.05]
    elif m.status == GRB.Status.INFEASIBLE:
        return None, None
    else:
        assert False


def addLazyC_subProblem(m, where):
    if where == GRB.callback.MIPSOL:
        tNodes = []
        numTaskInBundle = 0
        for i in m._T:
            if m.cbGetSolution(m._z_i[i]) > 0.5:
                tNodes.append('p0%d' % i)
                tNodes.append('d%d' % i)
                numTaskInBundle += 1
        #
        for b in m._B:
            if len(b) == numTaskInBundle:
                m.cbLazy(quicksum(m._z_i[i] for i in b) <= len(b) - 1)
        #
        for k in m._K:
            ptNodes = tNodes[:] + ['ori%d' % k, 'dest%d' % k]
            selectedEdges = [(i, j) for j in ptNodes for i in ptNodes if m.cbGetSolution(m._x_kij[k, i, j]) > 0.5]
            subtours = get_subtours(selectedEdges, ptNodes)
            for sb in subtours:
                if not sb:
                    continue
                expr = 0
                visited_nodes = set()
                for i, j in sb:
                    expr += m._x_kij[k, i, j]
                    visited_nodes.add(i)
                    visited_nodes.add(j)
                if len(visited_nodes) == len(ptNodes):
                    #  # eq:initOrder
                    m.cbLazy(m._o_ki[k, 'ori%d' % k] == 1)
                    m.cbLazy(m._o_ki[k, 'dest%d' % k] == 2 * (numTaskInBundle + 1))
                    for i in m._T:  # eq:pdSequnce
                        m.cbLazy(m._o_ki[k, 'p0%d' % i] <= m._o_ki[k, 'd%d' % i])
                    for i in ptNodes:
                        for j in ptNodes:  # eq:ordering
                            if i == j: continue
                            if i == 'dest%d' % k and j == 'ori%d' % k: continue
                            m.cbLazy(m._o_ki[k, i] + 1 <= m._o_ki[k, j] + 2 * (1 - m._x_kij[k, i, j]) * (numTaskInBundle + 1))
                else:
                    m.cbLazy(expr <= len(visited_nodes) - 1)  # eq:subTourElim


def minTimePD(b, k, t_ij):
    _kP, _kM = 'ori%d' % k, 'dest%d' % k
    N = {_kP, _kM}
    P, D = set(), set()
    for i in b:
        P.add('p0%d' % i)
        D.add('d%d' % i)
    N = N.union(P)
    N = N.union(D)
    #
    m = Model('minPD')
    m.Params.OutputFlag = SUB_SUB_LOGGING
    ofpath = opath.join(dpath['gurobiLog'], 'minPD.log')
    m.setParam('LogFile', ofpath)
    #
    # Define decision variables
    #
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
    obj -= t_ij[_kP, _kM]
    m.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constrains
    #
    # Flow based routing
    #  eq:pathPD
    m.addConstr(quicksum(x_ij[_kP, j] for j in P) == 1, name='pPDo')
    m.addConstr(quicksum(x_ij[i, _kM] for i in D) == 1, name='pPDi')
    #  # eq:XpathPD
    m.addConstr(quicksum(x_ij[_kP, j] for j in D) == 0, name='XpPDo')
    m.addConstr(quicksum(x_ij[i, _kM] for i in P) == 0, name='XpPDi')
    #
    for i in N:  # eq:outFlow
        if i == _kM: continue
        m.addConstr(quicksum(x_ij[i, j] for j in N if j != _kP) == 1, name='tOF[%s]' % i)
    for j in N:  # eq:inFlow
        if j == _kP: continue
        m.addConstr(quicksum(x_ij[i, j] for i in N if i != _kM) == 1, name='tIF[%s]' % j)
    for i in N:  # eq:XselfFlow
        m.addConstr(x_ij[i, i] == 0, name='XsF[%s]' % i)
    for i in N:
        for j in N:  # eq:direction
            m.addConstr(x_ij[i, j] + x_ij[j, i] <= 1, name='dir[%s,%s]' % (i, j))
    #  # eq:initOrder
    m.addConstr(o_i[_kP] == 1, name='ordOri')
    m.addConstr(o_i[_kM] == len(N), name='ordDest')
    for i in b:  # eq:pdSequnce
        m.addConstr(o_i['p0%d' % i] <= o_i['d%d' % i], name='ord[%s]' % i)
    for i in N:
        for j in N:  # eq:ordering
            if i == _kM or j == _kP:
                continue
            m.addConstr(o_i[i] + 1 <= o_i[j] + len(N) * (1 - x_ij[i, j]), name='ord[%s,%s]' % (i, j))
    #
    # For callback function
    #
    m._N = N
    m._x_ij = x_ij
    #
    m.params.LazyConstraints = 1
    m.optimize(addSubTourElimC_minPD)
    #
    if m.status != GRB.Status.OPTIMAL:
        m.computeIIS()
        m.write('minPD.ilp')
    return m.objVal


def addSubTourElimC_minPD(m, where):
    if where == GRB.callback.MIPSOL:
        curRoute = []
        for i in m._N:
            for j in m._N:
                if i == j:
                    continue
                if m.cbGetSolution(m._x_ij[i, j]) > 0.5:
                    curRoute.append((i, j))
        subtours = get_subtours(curRoute, m._N)
        for sb in subtours:
            if not sb:
                continue
            expr = 0
            visited_nodes = set()
            for i, j in sb:
                expr += m._x_ij[i, j]
                visited_nodes.add(i)
                visited_nodes.add(j)
            if len(visited_nodes) == len(m._N):
                continue
            m.cbLazy(expr <= len(visited_nodes) - 1)  # eq:subTourElim


def get_subtours(edges, nodes):
    subtours = []
    visited, adj = {}, {}
    for i in nodes:
        visited[i] = False
        adj[i] = []
    for i, j in edges:
        adj[i].append(j)
    while True:
        for i in visited.keys():
            if not visited[i]:
                break
        thistour = []
        while True:
            visited[i] = True
            neighbors = [j for j in adj[i] if not visited[j]]
            if len(neighbors) == 0:
                break
            thistour.append((i, neighbors[0]))
            i = neighbors[0]
        if thistour:
            subtours.append(thistour)
        if all(visited.values()):
            break
    return subtours


def route_display(edges):
    route = [e for e in edges if e[0].startswith('ori')]
    while len(route) != len(edges):
        for e in edges:
            if e[0] == route[-1][1]:
                route.append(e)
                break
    return route


if __name__ == '__main__':
    from problems import *

    # masterProblem(convert_input4MathematicalModel(*ex1()))
    run(convert_input4MathematicalModel(*ex8()))
