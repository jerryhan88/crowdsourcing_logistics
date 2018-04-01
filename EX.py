import time
from gurobipy import *
#
from _util import log2file, res2file
from _util import set_grbSettings
from problems import *


def run(problem, etcSetting, grbSetting):
    startCpuTime, startWallTime = time.clock(), time.time()
    if 'TimeLimit' not in etcSetting:
        etcSetting['TimeLimit'] = 1e400
    etcSetting['startTS'] = startCpuTime
    #
    def callbackF(m, where):
        if where == GRB.Callback.MIP:
            if time.clock() - etcSetting['startTS'] > etcSetting['TimeLimit']:
                logContents = '\n\n'
                logContents += '======================================================================================\n'
                logContents += 'Interrupted by time limit\n'
                logContents += '======================================================================================\n'
                log2file(etcSetting['LogFile'], logContents)
                m.terminate()
    #
    inputs = convert_p2i(*problem)
    bB = inputs['bB']
    T, r_i, v_i, _lambda = list(map(inputs.get, ['T', 'r_i', 'v_i', '_lambda']))
    P, D, N = list(map(inputs.get, ['P', 'D', 'N']))
    K, w_k = list(map(inputs.get, ['K', 'w_k']))
    t_ij, _delta = list(map(inputs.get, ['t_ij', '_delta']))
    #
    B = list(range(bB))
    bigM1 = sum(r_i)
    bigM2 = len(N) + 2
    bigM3 = len(N) * max(t_ij.values())
    #
    # Define decision variables
    #
    EX = Model('EX')
    z_bi, y_bk, R_b, a_bk, o_ki, x_bkij = {}, {}, {}, {}, {}, {}
    #
    for b in B:
        for i in T:
            z_bi[b, i] = EX.addVar(vtype=GRB.BINARY, name='z[%d,%d]' % (b, i))
        for k in K:
            y_bk[b, k] = EX.addVar(vtype=GRB.BINARY, name='y[%d,%d]' % (b, k))
            a_bk[b, k] = EX.addVar(vtype=GRB.CONTINUOUS, name='a[%d,%d]' % (b, k))
        R_b[b] = EX.addVar(vtype=GRB.CONTINUOUS, name='R[%d]' % b)
    for k in K:
        kN = N.union({'ori%d' % k, 'dest%d' % k})
        for i in kN:
            o_ki[k, i] = EX.addVar(vtype=GRB.INTEGER, name='o[%d,%s]' % (k, i))
            for j in kN:
                for b in B:
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
    #
    for b in B:
        for k in K:  # eq:linAlpha
            EX.addConstr(a_bk[b, k] <= bigM1 * y_bk[b, k],
                        name='la1[%d,%d]' % (b, k))
            EX.addConstr(a_bk[b, k] <= R_b[b],
                        name='la2[%d,%d]' % (b, k))
    #
    # Bundle
    #
    for i in T:  # eq:taskA
        EX.addConstr(quicksum(z_bi[b, i] for b in B) <= 1,
                    name='ta[%d]' % i)
    for b in B:
        #  # eq:sumReward
        EX.addConstr(quicksum(r_i[i] * z_bi[b, i] for i in T) == R_b[b],
                     name='sr[%d]' % b)
        #  # eq:volTh
        EX.addConstr(quicksum(v_i[i] * z_bi[b, i] for i in T) <= _lambda,
                    name='vt[%d]' % b)
    #
    # Routing_Flow
    #
    for b in B:
        for k in K:
            kN = N.union({'ori%d' % k, 'dest%d' % k})
            #  # eq:initFlow
            EX.addConstr(quicksum(x_bkij[b, k, 'ori%d' % k, j] for j in kN) == 1,
                        name='pPDo[%d,%d]' % (b, k))
            EX.addConstr(quicksum(x_bkij[b, k, j, 'dest%d' % k] for j in kN) == 1,
                        name='pPDi[%d,%d]' % (b, k))
            #  # eq:noInFlowOutFlow
            EX.addConstr(quicksum(x_bkij[b, k, j, 'ori%d' % k] for j in kN) == 0,
                         name='XpPDo[%d,%d]' % (b, k))
            EX.addConstr(quicksum(x_bkij[b, k, 'dest%d' % k, j] for j in kN) == 0,
                         name='XpPDi[%d,%d]' % (b, k))
            for i in T:  # eq:taskOutFlow
                EX.addConstr(quicksum(x_bkij[b, k, 'p%d' % i, j] for j in kN) == z_bi[b, i],
                            name='tOF[%d,%d,%d]' % (b, k, i))
            for i in T:  # eq:taskInFlow
                EX.addConstr(quicksum(x_bkij[b, k, j, 'd%d' % i] for j in kN) == z_bi[b, i],
                            name='tIF[%d,%d,%d]' % (b, k, i))
            #
            for i in N:  # eq:flowCon
                EX.addConstr(quicksum(x_bkij[b, k, i, j] for j in kN) == quicksum(x_bkij[b, k, j, i] for j in kN),
                            name='fc[%d,%d,%s]' % (b, k, i))
    #
    # Routing_Ordering
    #
    for k in K:
        N_kM = N.union({'dest%d' % k})
        for i in N_kM:
            #  # eq:initOrder
            EX.addConstr(2 <= o_ki[k, i],
                         name='initO1[%d,%s]' % (k, i))
            EX.addConstr(o_ki[k, i] <= bigM2,
                         name='initO2[%d,%s]' % (k, i))
            for j in N_kM:
                for b in B:
                    #  # eq:subEli
                    EX.addConstr(o_ki[k, i] + 1 <= o_ki[k, j] + bigM2 * (1 - x_bkij[b, k, i, j]),
                         name='subEli[%d,%d,%s,%s]' % (b, k, i, j))
        for i in T:
            for b in B:
                #  # eq:pdSequence
                EX.addConstr(o_ki[k, 'p%d' % i] <= o_ki[k, 'd%d' % i] + bigM2 * (1 - z_bi[b, i]),
                             name='pdS[%d,%d,%d]' % (b, k, i))
    #
    # Detour feasibility
    #
    for b in B:
        for k in K:
            #  # eq:detourFeasibility
            kN = N.union({'ori%d' % k, 'dest%d' % k})
            LRS = LinExpr()
            for i in kN:
                for j in kN:
                    LRS += t_ij[i, j] * x_bkij[b, k, i, j]
            LRS -= t_ij['ori%d' % k, 'dest%d' % k]
            EX.addConstr(LRS <= _delta + bigM3 * (1 - y_bk[b, k]),
                        name='df[%d,%d]' % (b, k))
    #
    # Run Gurobi (Optimization)
    #
    set_grbSettings(EX, grbSetting)
    EX.params.LazyConstraints = 1
    EX.optimize(callbackF)
    #
    endCpuTime, endWallTime = time.clock(), time.time()
    eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
    logContents = 'Summary\n'
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
        log2file(etcSetting['LogFile'], logContents)
        res2file(etcSetting['ResFile'], EX.objVal, EX.MIPGap, eliCpuTime, eliWallTime)
    except:
        res2file(etcSetting['ResFile'], -1, -1, eliCpuTime, eliWallTime)


if __name__ == '__main__':
    import os.path as opath
    from problems import paperExample, ex1
    #
    problem = paperExample()
    # problem = ex1()
    problemName = problem[0]
    exLogF = opath.join('_temp', '%s_EX.log' % problemName)
    exResF = opath.join('_temp', '%s_EX.csv' % problemName)


    etcSetting = {'LogFile': exLogF,
                  'ResFile': exResF}
    grbSetting = {'LogFile': exLogF}
    #
    run(problem, etcSetting, grbSetting)
