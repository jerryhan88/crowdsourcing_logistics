import multiprocessing
import time
import pickle
from gurobipy import *
#
from _util import log2file, res2file
from problems import convert_prob2prmt

NUM_CORES = multiprocessing.cpu_count()


def run(problem, etc=None):
    startCpuTime, startWallTime = time.clock(), time.time()
    if 'TimeLimit' not in etc:
        etc['TimeLimit'] = 1e400
    etc['startTS'] = startCpuTime
    #
    def callbackF(m, where):
        if where == GRB.Callback.MIP:
            if time.clock() - etc['startTS'] > etc['TimeLimit']:
                logContents = '\n'
                logContents += 'Interrupted by time limit\n'
                log2file(etc['LogFile'], logContents)
                m.terminate()
    #
    inputs = convert_prob2prmt(*problem)
    bB = inputs['bB']
    T, _lambda = list(map(inputs.get, ['T', '_lambda']))
    P, D, N = list(map(inputs.get, ['P', 'D', 'N']))
    K, w_k = list(map(inputs.get, ['K', 'w_k']))
    t_ij, _delta = list(map(inputs.get, ['t_ij', '_delta']))
    #
    B = list(range(bB))
    bigM2 = len(N) + 2
    bigM3 = len(N) * max(t_ij.values())

    cB_M = 2
    cB_P = 4
    cW = sum(w_k) / len(w_k) * 2
    print(cW)
    #
    # Define decision variables
    #
    EX = Model('EX1')
    z_bi = {(b, i): EX.addVar(vtype=GRB.BINARY, name='z[%d,%d]' % (b, i))
            for b in B for i in T}
    y_bk = {(b, k): EX.addVar(vtype=GRB.BINARY, name='y[%d,%d]' % (b, k))
            for b in B for k in K}
    g_b = {b: EX.addVar(vtype=GRB.BINARY, name='g[%d]' % b)
           for b in B}
    #
    o_ki, x_bkij = {}, {}
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
        for i in T:  # eq:ObjF
            obj += z_bi[b, i]
    EX.setObjective(obj, GRB.MAXIMIZE)

    for b in B:
        EX.addConstr(quicksum(z_bi[b, i] for i in T) >= cB_M * g_b[b],
                     name='minTB1[%d]' % b)
        EX.addConstr(quicksum(z_bi[b, i] for i in T) <= cB_P * g_b[b],
                     name='minTB2[%d]' % b)
    #
    # Define constrains
    #
    #  Bundle
    #
    for i in T:  # eq:taskA
        EX.addConstr(quicksum(z_bi[b, i] for b in B) <= 1,
                    name='ta[%d]' % i)
    #
    #  Routing_Flow
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
            for i in T:
                #  # eq:taskOutFlow
                EX.addConstr(quicksum(x_bkij[b, k, 'p%d' % i, j] for j in kN) == z_bi[b, i],
                            name='tOF[%d,%d,%d]' % (b, k, i))
                #  # eq:taskInFlow
                EX.addConstr(quicksum(x_bkij[b, k, j, 'd%d' % i] for j in kN) == z_bi[b, i],
                            name='tIF[%d,%d,%d]' % (b, k, i))
            for i in N:  # eq:flowCon
                EX.addConstr(quicksum(x_bkij[b, k, i, j] for j in kN) == quicksum(x_bkij[b, k, j, i] for j in kN),
                            name='fc[%d,%d,%s]' % (b, k, i))
    #
    #  Routing_Ordering
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
        EX.addConstr(quicksum(w_k[k] * y_bk[b, k] for k in K) >= cW * g_b[b],
                     name='bg[%d]' % b)
    #
    # Run Gurobi (Optimization)
    #
    EX.params.LazyConstraints = 1
    EX.setParam('Threads', NUM_CORES)
    if etc['logFile']:
        EX.setParam('LogFile', etc['logFile'])
    EX.optimize(callbackF)
    #
    if EX.status == GRB.Status.INFEASIBLE:
        EX.write('%s.lp' % problemName)
        EX.computeIIS()
        EX.write('%s.ilp' % problemName)
    #
    if etc and EX.status != GRB.Status.INFEASIBLE:
        assert 'solFilePKL' in etc
        assert 'solFileCSV' in etc
        assert 'solFileTXT' in etc
        #
        with open(etc['solFileTXT'], 'w') as f:
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
            f.write(logContents)
            f.write('\n')
            chosenB = [[i for i in T if z_bi[b, i].x > 0.5] for b in B]
            logContents += '\t chosen B.: %s\n' % str(chosenB)
            logContents += '\n'
            for b in B:
                bundle = [i for i in T if z_bi[b, i].x > 0.5]
                logContents += '%s (%d) \n' % (str(bundle), len(bundle))
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
                        logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w_k[k], detourTime, 1, str(route))
                    else:
                        logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w_k[k], detourTime, 0, str(route))
            f.write(logContents)
        #
        res2file(etc['solFileCSV'], EX.objVal, EX.MIPGap, eliCpuTime, eliWallTime)
        #
        _z_bi = {(b, i): z_bi[b, i].x for b in B for i in T}
        _y_bk = {(b, k): y_bk[b, k].x for b in B for k in K}
        _g_b = {b: g_b[b].x for b in B}
        #
        _o_ki, _x_bkij = {}, {}
        for k in K:
            kN = N.union({'ori%d' % k, 'dest%d' % k})
            for i in kN:
                _o_ki[k, i] = o_ki[k, i].x
                for j in kN:
                    for b in B:
                        _x_bkij[b, k, i, j] = x_bkij[b, k, i, j].x
        sol = {'z_bi': _z_bi, 'y_bk': _y_bk, 'g_b': _g_b,
               'o_ki': _o_ki, 'x_bkij': _x_bkij}
        with open(etc['solFilePKL'], 'wb') as fp:
            pickle.dump(sol, fp)


if __name__ == '__main__':
    import os.path as opath
    from problems import euclideanDistEx0
    #
    problem = euclideanDistEx0()
    problemName = problem[0]
    #
    etc = {'solFilePKL': opath.join('_temp', 'sol_%s_EX1.pkl' % problemName),
           'solFileCSV': opath.join('_temp', 'sol_%s_EX1.csv' % problemName),
           'solFileTXT': opath.join('_temp', 'sol_%s_EX1.txt' % problemName),
           'logFile': opath.join('_temp', '%s_EX1.log' % problemName)
           }
    #
    run(problem, etc)
