from init_project import *
#
from gurobipy import *
from datetime import datetime
from time import clock
#
from _utils.logging import record_logs, O_GL, X_GL
from _utils.mm_utils import get_routeFromOri

def run(problem, log_fpath=None, numThreads=None, TimeLimit=None):
    def addLazyC(m, where):
        if where == GRB.callback.MIPSOL:
            for b in m._B:
                tNodes = []
                for i in m._T:
                    if m.cbGetSolution(m._z_bi[b, i]) > 0.5:
                        tNodes.append('p%d' % i)
                        tNodes.append('d%d' % i)
                for k in m._K:
                    ptNodes = tNodes[:] + ['ori%d' % k, 'dest%d' % k]
                    selectedEdges = [(i, j) for j in ptNodes for i in ptNodes if
                                     m.cbGetSolution(m._x_bkij[b, k, i, j]) > 0.5]
                    route = get_routeFromOri(selectedEdges, ptNodes)
                    if len(route) != len(ptNodes) - 1:
                        expr = 0
                        for i, j in route:
                            expr += m._x_bkij[b, k, i, j]
                        m.cbLazy(expr <= len(route) - 1)  # eq:subTourElim
    #
    startTime = clock()
    bB, \
    T, r_i, v_i, _lambda, P, D, N, \
    K, w_k, t_ij, _delta = convert_input4MathematicalModel(*problem)
    B = list(range(bB))
    bigM1 = sum(r_i) * 2
    bigM2 = (len(T) + 1) * 2
    bigM3 = sum(t_ij.values())
    #
    # Define decision variables
    #
    m = Model('exactModel')
    z_bi, y_bk, a_bk, x_bkij, o_bki = {}, {}, {}, {}, {}
    #
    for b in B:
        for i in T:
            z_bi[b, i] = m.addVar(vtype=GRB.BINARY, name='z[%d,%d]' % (b, i))
        for k in K:
            y_bk[b, k] = m.addVar(vtype=GRB.BINARY, name='y[%d,%d]' % (b, k))
            a_bk[b, k] = m.addVar(vtype=GRB.CONTINUOUS, name='a[%d,%d]' % (b, k))
        for k in K:
            kN = N.union({'ori%d' % k, 'dest%d' % k})
            for i in kN:
                o_bki[b, k, i] = m.addVar(vtype=GRB.INTEGER, name='o[%d,%d,%s]' % (b, k, i))
                for j in kN:
                    x_bkij[b, k, i, j] = m.addVar(vtype=GRB.BINARY, name='x[%d,%d,%s,%s]' % (b, k, i, j))
    m.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for b in B:
        for k in K:  # eq:linear_proObj
            obj += w_k[k] * a_bk[b, k]
    m.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    # Linearization
    for b in B:
        for k in K:  # eq:linAlpha
            m.addConstr(a_bk[b, k] >= quicksum(r_i[i] * z_bi[b, i] for i in T) - bigM1 * (1 - y_bk[b, k]),
                        name='la1[%d,%d]' % (b, k))
            m.addConstr(a_bk[b, k] <= bigM1 * y_bk[b, k],
                        name='la2[%d,%d]' % (b, k))
            m.addConstr(a_bk[b, k] <= quicksum(r_i[i] * z_bi[b, i] for i in T),
                        name='la3[%d,%d]' % (b, k))
    #
    # Bundle
    for i in T:  # eq:taskA
        m.addConstr(quicksum(z_bi[b, i] for b in B) == 1,
                    name='ta[%d]' % i)
    for b in B:  # eq:volTh
        m.addConstr(quicksum(z_bi[b, i] * v_i[i] for i in T) <= _lambda,
                    name='vt[%d]' % b)
    #
    # Flow based routing
    for b in B:
        for k in K:
            #  # eq:circularF
            m.addConstr(x_bkij[b, k, 'dest%d' % k, 'ori%d' % k] == 1,
                        name='cf[%d,%d]' % (b, k))
            #  # eq:pathPD
            m.addConstr(quicksum(x_bkij[b, k, 'ori%d' % k, j] for j in P) == 1,
                        name='pPDo[%d,%d]' % (b, k))
            m.addConstr(quicksum(x_bkij[b, k, i, 'dest%d' % k] for i in D) == 1,
                        name='pPDi[%d,%d]' % (b, k))
            #  # eq:XpathPD
            m.addConstr(quicksum(x_bkij[b, k, 'ori%d' % k, j] for j in D) == 0,
                        name='XpPDo[%d,%d]' % (b, k))
            m.addConstr(quicksum(x_bkij[b, k, i, 'dest%d' % k] for i in P) == 0,
                        name='XpPDi[%d,%d]' % (b, k))
            for i in T:  # eq:taskOutFlow
                m.addConstr(quicksum(x_bkij[b, k, 'p%d' % i, j] for j in N) == z_bi[b, i],
                            name='tOF[%d,%d,%d]' % (b, k, i))
            for j in T:  # eq:taskInFlow
                m.addConstr(quicksum(x_bkij[b, k, i, 'd%d' % j] for i in N) == z_bi[b, j],
                            name='tIF[%d,%d,%d]' % (b, k, j))
            #
            kP, kM = 'ori%d' % k, 'dest%d' % k
            kN = N.union({kP, kM})
            for i in kN:  # eq:XselfFlow
                m.addConstr(x_bkij[b, k, i, i] == 0,
                            name='Xsf[%d,%d,%s]' % (b, k, i))
            for j in kN:  # eq:flowCon
                m.addConstr(quicksum(x_bkij[b, k, i, j] for i in kN) == quicksum(x_bkij[b, k, j, i] for i in kN),
                            name='fc[%d,%d,%s]' % (b, k, j))
            for i in kN:
                for j in kN:  # eq:direction
                    if i == j: continue
                    m.addConstr(x_bkij[b, k, i, j] + x_bkij[b, k, j, i] <= 1,
                                name='dir[%d,%d,%s,%s]' % (b, k, i, j))
            m.addConstr(o_bki[b, k, kP] == 1,
                        name='iOF[%d,%d]' % (b, k))
            m.addConstr(o_bki[b, k, kM] <= bigM2,
                        name='iOE[%d,%d]' % (b, k))
            for i in T:  # eq:pdSequnce
                m.addConstr(o_bki[b, k, 'p%d' % i] <= o_bki[b, k, 'd%d' % i] + bigM2 * (1 - z_bi[b, i]),
                        name='pdS[%d,%d]' % (b, k))
            for i in kN:
                for j in kN:  # eq:ordering
                    if i == j: continue
                    if i == kM or j == kP: continue
                    m.addConstr(o_bki[b, k, i] + 1 <= o_bki[b, k, j] + bigM2 * (1 - x_bkij[b, k, i, j]))
            #
            # Feasibility
            #  # eq:pathFeasibility
            m.addConstr(quicksum(t_ij[i, j] * x_bkij[b, k, i, j] for i in kN for j in kN if not (i == kM and j == kP)) \
                        - t_ij[kP, kM] <= _delta + bigM3 * (1 - y_bk[b, k]),
                        name='pf[%d,%d]' % (b, k))
    #
    m._B, m._T, m._K = B, T, K
    m._z_bi, m._x_bkij = z_bi, x_bkij
    m.params.LazyConstraints = 1
    #
    # setting
    #
    if TimeLimit is not None:
        m.setParam('TimeLimit', TimeLimit)
    if numThreads is not None:
        m.setParam('Threads', numThreads)
    if log_fpath is not None:
        m.setParam('LogFile', log_fpath)
    m.setParam('OutputFlag', O_GL)
    #
    # Run Gurobi (Optimization)
    #
    m.optimize(addLazyC)
    endTime = clock()
    eliTime = endTime - startTime
    chosenB = [[i for i in T if z_bi[b, i].x > 0.5] for b in B]
    #
    logContents = '\n\n'
    logContents += 'Summary\n'
    logContents += '\t Sta.Time: %s\n' % str(startTime)
    logContents += '\t End.Time: %s\n' % str(endTime)
    logContents += '\t Eli.Time: %d\n' % eliTime
    logContents += '\t ObjV: %.3f\n' % m.objVal
    logContents += '\t chosen B.: %s\n' % str(chosenB)
    record_logs(log_fpath, logContents)
    #
    return m.objVal, eliTime


def convert_input4MathematicalModel(travel_time, \
                                    flows, paths, \
                                    tasks, rewards, volumes, \
                                    num_bundles, volume_th, detour_th):
    #
    # Bundle
    #
    bB = num_bundles
    _lambda = volume_th
    #
    # Task
    #
    T = list(range(len(tasks)))
    iP, iM = list(zip(*[tasks[i] for i in T]))
    r_i, v_i = rewards, volumes
    P, D = set(), set()
    _N = {}
    for i in T:
        P.add('p%d' % i)
        D.add('d%d' % i)
        #
        _N['p%d' % i] = iP[i]
        _N['d%d' % i] = iM[i]
    #
    # Path
    #
    K = list(range(len(paths)))
    kP, kM = list(zip(*[paths[k] for k in K]))
    sum_f_k = sum(flows[i][j] for i in range(len(flows)) for j in range(len(flows)))
    w_k = [flows[i][j] / float(sum_f_k) for i, j in paths]
    _delta = detour_th
    t_ij = {}
    for k in K:
        _kP, _kM = 'ori%d' % k, 'dest%d' % k
        t_ij[_kP, _kP] = travel_time[kP[k], kP[k]]
        t_ij[_kM, _kM] = travel_time[kM[k], kM[k]]
        t_ij[_kP, _kM] = travel_time[kP[k], kM[k]]
        t_ij[_kM, _kP] = travel_time[kM[k], kP[k]]
        for i in _N:
            t_ij[_kP, i] = travel_time[kP[k], _N[i]]
            t_ij[i, _kP] = travel_time[_N[i], kP[k]]
            #
            t_ij[_kM, i] = travel_time[kM[k], _N[i]]
            t_ij[i, _kM] = travel_time[_N[i], kM[k]]
    for i in _N:
        for j in _N:
            t_ij[i, j] = travel_time[_N[i], _N[j]]
    N = set(_N.keys())
    #
    return bB, \
           T, r_i, v_i, _lambda, P, D, N, \
           K, w_k, t_ij, _delta


if __name__ == '__main__':
    from problems import *
    print(run(ex2()))
