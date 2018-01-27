import time
from gurobipy import *
#
from _util import log2file, res2file
from _util import set_grbSettings, get_routeFromOri
from problems import *


def run(probSetting, etcSetting, grbSetting):
    startCpuTime, startWallTime = time.clock(), time.time()
    if 'TimeLimit' not in etcSetting:
        etcSetting['TimeLimit'] = 1e400
    etcSetting['startTS'] = startCpuTime
    #
    def callbackF(m, where):
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
                        m.cbLazy(quicksum(m._x_bkij[b, k, i, j] for i, j in route)
                                 <= len(route) - 1)  # eq:subTourElim
        if where == GRB.Callback.MIP:
            if time.clock() - etcSetting['startTS'] > etcSetting['TimeLimit']:
                logContents = '\n\n'
                logContents += '======================================================================================\n'
                logContents += 'Interrupted by time limit\n'
                logContents += '======================================================================================\n'
                log2file(etcSetting['LogFile'], logContents)
                m.terminate()
    #
    inputs = convert_p2i(*probSetting['problem'])
    bB = inputs['bB']
    T, r_i, v_i, _lambda = list(map(inputs.get, ['T', 'r_i', 'v_i', '_lambda']))
    P, D, N = list(map(inputs.get, ['P', 'D', 'N']))
    K, w_k = list(map(inputs.get, ['K', 'w_k']))
    t_ij, _delta = list(map(inputs.get, ['t_ij', '_delta']))
    #
    B = list(range(bB))
    bigM1 = sum(r_i) * 2
    bigM2 = (len(T) + 1) * 2
    bigM3 = sum(t_ij.values())
    #
    # Define decision variables
    #
    EX = Model('EX')
    z_bi, y_bk, a_bk, x_bkij, o_bki = {}, {}, {}, {}, {}
    #
    for b in B:
        for i in T:
            z_bi[b, i] = EX.addVar(vtype=GRB.BINARY, name='z[%d,%d]' % (b, i))
        for k in K:
            y_bk[b, k] = EX.addVar(vtype=GRB.BINARY, name='y[%d,%d]' % (b, k))
            a_bk[b, k] = EX.addVar(vtype=GRB.CONTINUOUS, name='a[%d,%d]' % (b, k))
        for k in K:
            kN = N.union({'ori%d' % k, 'dest%d' % k})
            for i in kN:
                o_bki[b, k, i] = EX.addVar(vtype=GRB.INTEGER, name='o[%d,%d,%s]' % (b, k, i))
                for j in kN:
                    x_bkij[b, k, i, j] = EX.addVar(vtype=GRB.BINARY, name='x[%d,%d,%s,%s]' % (b, k, i, j))
    EX.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for b in B:
        for k in K:  # eq:linear_proObj
            obj += w_k[k] * a_bk[b, k]
    EX.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    # Linearization
    for b in B:
        for k in K:  # eq:linAlpha
            EX.addConstr(a_bk[b, k] <= bigM1 * y_bk[b, k],
                        name='la2[%d,%d]' % (b, k))
            EX.addConstr(a_bk[b, k] <= quicksum(r_i[i] * z_bi[b, i] for i in T),
                        name='la3[%d,%d]' % (b, k))
    #
    # Bundle
    for i in T:  # eq:taskA
        EX.addConstr(quicksum(z_bi[b, i] for b in B) <= 1,
                    name='ta[%d]' % i)
    for b in B:  # eq:volTh
        EX.addConstr(quicksum(z_bi[b, i] * v_i[i] for i in T) <= _lambda,
                    name='vt[%d]' % b)
    #
    # Flow based routing
    for b in B:
        for k in K:
            #  # eq:circularF
            EX.addConstr(x_bkij[b, k, 'dest%d' % k, 'ori%d' % k] == 1,
                        name='cf[%d,%d]' % (b, k))
            #  # eq:pathPD
            EX.addConstr(quicksum(x_bkij[b, k, 'ori%d' % k, j] for j in P) == 1,
                        name='pPDo[%d,%d]' % (b, k))
            EX.addConstr(quicksum(x_bkij[b, k, i, 'dest%d' % k] for i in D) == 1,
                        name='pPDi[%d,%d]' % (b, k))
            #  # eq:XpathPD
            EX.addConstr(quicksum(x_bkij[b, k, 'ori%d' % k, j] for j in D) == 0,
                        name='XpPDo[%d,%d]' % (b, k))
            EX.addConstr(quicksum(x_bkij[b, k, i, 'dest%d' % k] for i in P) == 0,
                        name='XpPDi[%d,%d]' % (b, k))
            for i in T:  # eq:taskOutFlow
                EX.addConstr(quicksum(x_bkij[b, k, 'p%d' % i, j] for j in N) == z_bi[b, i],
                            name='tOF[%d,%d,%d]' % (b, k, i))
            for j in T:  # eq:taskInFlow
                EX.addConstr(quicksum(x_bkij[b, k, i, 'd%d' % j] for i in N) == z_bi[b, j],
                            name='tIF[%d,%d,%d]' % (b, k, j))
            #
            kP, kM = 'ori%d' % k, 'dest%d' % k
            kN = N.union({kP, kM})
            for i in kN:  # eq:XselfFlow
                EX.addConstr(x_bkij[b, k, i, i] == 0,
                            name='Xsf[%d,%d,%s]' % (b, k, i))
            for j in kN:  # eq:flowCon
                EX.addConstr(quicksum(x_bkij[b, k, i, j] for i in kN) == quicksum(x_bkij[b, k, j, i] for i in kN),
                            name='fc[%d,%d,%s]' % (b, k, j))
            for i in kN:
                for j in kN:  # eq:direction
                    if i == j: continue
                    EX.addConstr(x_bkij[b, k, i, j] + x_bkij[b, k, j, i] <= 1,
                                name='dir[%d,%d,%s,%s]' % (b, k, i, j))
            EX.addConstr(o_bki[b, k, kP] == 1,
                        name='iOF[%d,%d]' % (b, k))
            EX.addConstr(o_bki[b, k, kM] <= bigM2,
                        name='iOE[%d,%d]' % (b, k))
            for i in T:  # eq:pdSequnce
                EX.addConstr(o_bki[b, k, 'p%d' % i] <= o_bki[b, k, 'd%d' % i] + bigM2 * (1 - z_bi[b, i]),
                        name='pdS[%d,%d]' % (b, k))
            for i in kN:
                for j in kN:  # eq:ordering
                    if i == j: continue
                    if i == kM or j == kP: continue
                    EX.addConstr(o_bki[b, k, i] + 1 <= o_bki[b, k, j] + bigM2 * (1 - x_bkij[b, k, i, j]))
            #
            # Feasibility
            #  # eq:pathFeasibility
            EX.addConstr(quicksum(t_ij[i, j] * x_bkij[b, k, i, j] for i in kN for j in kN) \
                              - t_ij[kM, kP] - t_ij[kP, kM] <= _delta + bigM3 * (1 - y_bk[b, k]),
                        name='pf[%d,%d]' % (b, k))
    #
    EX._B, EX._T, EX._K = B, T, K
    EX._z_bi, EX._x_bkij = z_bi, x_bkij
    EX.params.LazyConstraints = 1
    #
    # Run Gurobi (Optimization)
    #
    set_grbSettings(EX, grbSetting)
    EX.optimize(callbackF)
    #
    endCpuTime, endWallTime = time.clock(), time.time()
    eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
    logContents = '\n\n'
    logContents += '======================================================================================\n'
    logContents += 'Summary\n'
    logContents += '\t Cpu Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(startCpuTime)
    logContents += '\t\t End.Time: %s\n' % str(endCpuTime)
    logContents += '\t\t Eli.Time: %f\n' % eliCpuTime
    logContents += '\t Wall Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(startWallTime)
    logContents += '\t\t End.Time: %s\n' % str(endWallTime)
    logContents += '\t\t Eli.Time: %f\n' % eliWallTime
    logContents += '\t ObjV: %.3f\n' % EX.objVal
    logContents += '\t Gap: %.3f\n' % EX.MIPGap
    try:
        chosenB = [[i for i in T if z_bi[b, i].x > 0.5] for b in B]
        logContents += '\t chosen B.: %s\n' % str(chosenB)
        logContents += '\n'
        for b in B:
            bundle = [i for i in T if z_bi[b, i].x > 0.5]
            br = sum([r_i[i] for i in bundle])
            logContents += '%s (%d) \n' % (str(bundle), br)
            p = 0
            for k in K:
                kP, kM = 'ori%d' % k, 'dest%d' % k
                kN = N.union({kP, kM})
                _route = {}
                detourTime = 0
                for j in kN:
                    for i in kN:
                        if x_bkij[b, k, i, j].x > 0.5:
                            detourTime += t_ij[i, j]
                            _route[i] = j
                detourTime -= t_ij[kM, kP]
                detourTime -= t_ij[kP, kM]
                i = kP
                route = []
                while i != kM:
                    route.append(i)
                    i = _route[i]
                route.append(i)
                if y_bk[b, k].x > 0.5:
                    p += w_k[k] * br
                    logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w_k[k], detourTime, 1, str(route))
                else:
                    logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w_k[k], detourTime, 0, str(route))
            logContents += '\t\t\t\t\t\t %.3f\n' % p
        logContents += '======================================================================================\n'
        log2file(etcSetting['LogFile'], logContents)
        res2file(etcSetting['ResFile'], EX.objVal, EX.MIPGap, eliCpuTime, eliWallTime)
    except:
        res2file(etcSetting['ResFile'], -1, -1, eliCpuTime, eliWallTime)


if __name__ == '__main__':
    import os.path as opath
    from problems import paperExample, ex1
    #
    problem = paperExample()
    probSetting = {'problem': problem}
    exLogF = opath.join('_temp', 'paperExample_EX.log')
    exResF = opath.join('_temp', 'paperExample_EX.csv')

    # problem = ex1()
    # probSetting = {'problem': problem}
    # exLogF = opath.join('_temp', 'ex1_EX.log')
    # exResF = opath.join('_temp', 'ex1_EX.csv')


    etcSetting = {'LogFile': exLogF,
                  'ResFile': exResF}
    grbSetting = {'LogFile': exLogF}
    #
    run(probSetting, etcSetting, grbSetting)
