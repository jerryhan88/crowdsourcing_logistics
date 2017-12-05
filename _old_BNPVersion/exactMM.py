from init_project import *
#
from gurobipy import *
#
from problems import *
import time
#
from _utils.recording import *
from _utils.mm_utils import get_routeFromOri


def run(problem, log_fpath=None, numThreads=None, TimeLimit=None):
    def addLazyC(m, where):
        if where == GRB.callback.MIPSOL:
            for b in m._B:
                tNodes = []
                for i in m._T:
                    if m.cbGetSolution(m._z_bi[b, i]) > 0.5:
                        tNodes.append('p0%d' % i)
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
    startCpuTime, startWallTime = time.clock(), time.time()
    bB, \
    T, r_i, v_i, _lambda, P, D, N, \
    K, w_k, t_ij, _delta = convert_p2i(*problem)
    B = list(range(bB))
    bigM1 = sum(r_i) * 2
    bigM2 = (len(T) + 1) * 2
    bigM3 = sum(t_ij.values())
    #
    # Define decision variables
    #
    exactMM = Model('exactModel')
    z_bi, y_bk, a_bk, x_bkij, o_bki = {}, {}, {}, {}, {}
    #
    for b in B:
        for i in T:
            z_bi[b, i] = exactMM.addVar(vtype=GRB.BINARY, name='z[%d,%d]' % (b, i))
        for k in K:
            y_bk[b, k] = exactMM.addVar(vtype=GRB.BINARY, name='y[%d,%d]' % (b, k))
            a_bk[b, k] = exactMM.addVar(vtype=GRB.CONTINUOUS, name='a[%d,%d]' % (b, k))
        for k in K:
            kN = N.union({'ori%d' % k, 'dest%d' % k})
            for i in kN:
                o_bki[b, k, i] = exactMM.addVar(vtype=GRB.INTEGER, name='o[%d,%d,%s]' % (b, k, i))
                for j in kN:
                    x_bkij[b, k, i, j] = exactMM.addVar(vtype=GRB.BINARY, name='x[%d,%d,%s,%s]' % (b, k, i, j))
    exactMM.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for b in B:
        for k in K:  # eq:linear_proObj
            obj += w_k[k] * a_bk[b, k]
    exactMM.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    # Linearization
    for b in B:
        for k in K:  # eq:linAlpha
            exactMM.addConstr(a_bk[b, k] >= quicksum(r_i[i] * z_bi[b, i] for i in T) - bigM1 * (1 - y_bk[b, k]),
                        name='la1[%d,%d]' % (b, k))
            exactMM.addConstr(a_bk[b, k] <= bigM1 * y_bk[b, k],
                        name='la2[%d,%d]' % (b, k))
            exactMM.addConstr(a_bk[b, k] <= quicksum(r_i[i] * z_bi[b, i] for i in T),
                        name='la3[%d,%d]' % (b, k))
    #
    # Bundle
    for i in T:  # eq:taskA
        exactMM.addConstr(quicksum(z_bi[b, i] for b in B) == 1,
                    name='ta[%d]' % i)
    for b in B:  # eq:volTh
        exactMM.addConstr(quicksum(z_bi[b, i] * v_i[i] for i in T) <= _lambda,
                    name='vt[%d]' % b)
    #
    # Flow based routing
    for b in B:
        for k in K:
            #  # eq:circularF
            exactMM.addConstr(x_bkij[b, k, 'dest%d' % k, 'ori%d' % k] == 1,
                        name='cf[%d,%d]' % (b, k))
            #  # eq:pathPD
            exactMM.addConstr(quicksum(x_bkij[b, k, 'ori%d' % k, j] for j in P) == 1,
                        name='pPDo[%d,%d]' % (b, k))
            exactMM.addConstr(quicksum(x_bkij[b, k, i, 'dest%d' % k] for i in D) == 1,
                        name='pPDi[%d,%d]' % (b, k))
            #  # eq:XpathPD
            exactMM.addConstr(quicksum(x_bkij[b, k, 'ori%d' % k, j] for j in D) == 0,
                        name='XpPDo[%d,%d]' % (b, k))
            exactMM.addConstr(quicksum(x_bkij[b, k, i, 'dest%d' % k] for i in P) == 0,
                        name='XpPDi[%d,%d]' % (b, k))
            for i in T:  # eq:taskOutFlow
                exactMM.addConstr(quicksum(x_bkij[b, k, 'p0%d' % i, j] for j in N) == z_bi[b, i],
                            name='tOF[%d,%d,%d]' % (b, k, i))
            for j in T:  # eq:taskInFlow
                exactMM.addConstr(quicksum(x_bkij[b, k, i, 'd%d' % j] for i in N) == z_bi[b, j],
                            name='tIF[%d,%d,%d]' % (b, k, j))
            #
            kP, kM = 'ori%d' % k, 'dest%d' % k
            kN = N.union({kP, kM})
            for i in kN:  # eq:XselfFlow
                exactMM.addConstr(x_bkij[b, k, i, i] == 0,
                            name='Xsf[%d,%d,%s]' % (b, k, i))
            for j in kN:  # eq:flowCon
                exactMM.addConstr(quicksum(x_bkij[b, k, i, j] for i in kN) == quicksum(x_bkij[b, k, j, i] for i in kN),
                            name='fc[%d,%d,%s]' % (b, k, j))
            for i in kN:
                for j in kN:  # eq:direction
                    if i == j: continue
                    exactMM.addConstr(x_bkij[b, k, i, j] + x_bkij[b, k, j, i] <= 1,
                                name='dir[%d,%d,%s,%s]' % (b, k, i, j))
            exactMM.addConstr(o_bki[b, k, kP] == 1,
                        name='iOF[%d,%d]' % (b, k))
            exactMM.addConstr(o_bki[b, k, kM] <= bigM2,
                        name='iOE[%d,%d]' % (b, k))
            for i in T:  # eq:pdSequnce
                exactMM.addConstr(o_bki[b, k, 'p0%d' % i] <= o_bki[b, k, 'd%d' % i] + bigM2 * (1 - z_bi[b, i]),
                        name='pdS[%d,%d]' % (b, k))
            for i in kN:
                for j in kN:  # eq:ordering
                    if i == j: continue
                    if i == kM or j == kP: continue
                    exactMM.addConstr(o_bki[b, k, i] + 1 <= o_bki[b, k, j] + bigM2 * (1 - x_bkij[b, k, i, j]))
            #
            # Feasibility
            #  # eq:pathFeasibility
            exactMM.addConstr(quicksum(t_ij[i, j] * x_bkij[b, k, i, j] for i in kN for j in kN if not (i == kM and j == kP)) \
                              - t_ij[kM, kP] - t_ij[kP, kM] <= _delta + bigM3 * (1 - y_bk[b, k]),
                        name='pf[%d,%d]' % (b, k))
    #
    exactMM._B, exactMM._T, exactMM._K = B, T, K
    exactMM._z_bi, exactMM._x_bkij = z_bi, x_bkij
    exactMM.params.LazyConstraints = 1
    #
    # setting
    #
    if TimeLimit is not None:
        exactMM.setParam('TimeLimit', TimeLimit)
    if numThreads is not None:
        exactMM.setParam('Threads', numThreads)
    if log_fpath is not None:
        exactMM.setParam('LogFile', log_fpath)
    exactMM.setParam('OutputFlag', O_GL)
    #
    # Run Gurobi (Optimization)
    #
    exactMM.optimize(addLazyC)
    #
    endCpuTime, endWallTime = time.clock(), time.time()
    eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
    chosenB = [[i for i in T if z_bi[b, i].x > 0.5] for b in B]
    #
    logContents = '\n\n'
    logContents += 'Summary\n'
    logContents += '\t Cpu Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(startCpuTime)
    logContents += '\t\t End.Time: %s\n' % str(endCpuTime)
    logContents += '\t\t Eli.Time: %f\n' % eliCpuTime
    logContents += '\t Wall Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(startWallTime)
    logContents += '\t\t End.Time: %s\n' % str(endWallTime)
    logContents += '\t\t Eli.Time: %f\n' % eliWallTime
    logContents += '\t ObjV: %.3f\n' % exactMM.objVal
    logContents += '\t Gap: %.3f\n' % exactMM.MIPGap
    logContents += '\t chosen B.: %s\n' % str(chosenB)
    record_log(log_fpath, logContents)
    #
    return exactMM.objVal, exactMM.MIPGap, eliCpuTime, eliWallTime


if __name__ == '__main__':
    from problems import *
    print(run(ex2()))
