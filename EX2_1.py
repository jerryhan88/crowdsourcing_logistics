import os.path as opath
import multiprocessing
import time
import pickle, csv
from gurobipy import *
#
from _util_logging import write_log, res2file

NUM_CORES = multiprocessing.cpu_count()
LOGGING_INTERVAL = 20


def itr2file(fpath, contents=[]):
    if not contents:
        if opath.exists(fpath):
            os.remove(fpath)
        with open(fpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            header = ['eliCpuTime', 'eliWallTime',
                      'objbst', 'objbnd', 'gap']
            writer.writerow(header)
    else:
        with open(fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow(contents)


def get_routeFromOri(edges, nodes):
    visited, adj = {}, {}
    for i in nodes:
        visited[i] = False
        adj[i] = []
    for i, j in edges:
        adj[i].append(j)
    route = []
    cNode = None
    for n in nodes:
        if n.startswith('ori'):
            cNode = n
            break
    else:
        assert False
    while not cNode.startswith('dest'):
        visited[cNode] = True
        neighbors = [j for j in adj[cNode] if not visited[j]]
        route.append((cNode, neighbors[0]))
        cNode = neighbors[0]
        if visited[cNode]:
            break
    return route


def run(prmt, etc=None):
    startCpuTime, startWallTime = time.clock(), time.time()
    if 'TimeLimit' not in etc:
        etc['TimeLimit'] = 1e400
    etc['startTS'] = startCpuTime
    etc['startCpuTime'] = startCpuTime
    etc['startWallTime'] = startWallTime
    etc['lastLoggingTime'] = startWallTime
    itr2file(etc['itrFileCSV'])
    #
    def callbackF(m, where):
        if where == GRB.Callback.MIP:
            if time.clock() - etc['startTS'] > etc['TimeLimit']:
                logContents = '\n'
                logContents += 'Interrupted by time limit\n'
                write_log(etc['logFile'], logContents)
                m.terminate()
            if time.time() - etc['lastLoggingTime'] > LOGGING_INTERVAL:
                etc['lastLoggingTime'] = time.time()
                eliCpuTimeP, eliWallTimeP = time.clock() - etc['startCpuTime'], time.time() - etc['startWallTime']
                objbst = m.cbGet(GRB.Callback.MIP_OBJBST)
                objbnd = m.cbGet(GRB.Callback.MIP_OBJBND)
                gap = abs(objbst - objbnd) / (0.000001 + abs(objbst))
                itr2file(etc['itrFileCSV'], ['%.2f' % eliCpuTimeP, '%.2f' % eliWallTimeP,
                                             '%.2f' % objbst, '%.2f' % objbnd, '%.2f' % gap])
        if where == GRB.callback.MIPSOL:

            for b in m._B:
                if m.cbGetSolution(m._g_b[b]) > 0.5:
                    m.cbLazy(quicksum(m._w_k[k] * m._y_bk[b, k] for k in m._K) >= m._cW)


            # for b in m._B:
            #     selectedTasks, tNodes = [], []
            #     for i in m._T:
            #         if m.cbGetSolution(m._z_bi[b, i]) > 0.5:
            #             selectedTasks.append(i)
            #             tNodes.append('p%d' % i)
            #             tNodes.append('d%d' % i)
            #     if len(tNodes) == 0:
            #         continue
            #     for k in m._K:
            #         ptNodes = tNodes[:] + ['ori%d' % k, 'dest%d' % k]
            #         selectedEdges = [(i, j) for j in ptNodes for i in ptNodes if
            #                          m.cbGetSolution(m._x_bkij[b, k, i, j]) > 0.5]
            #         route = get_routeFromOri(selectedEdges, ptNodes)
            #         if len(route) != len(ptNodes) - 1:
            #             expr = 0
            #             for i, j in route:
            #                 expr += m._x_bkij[b, k, i, j]
            #             m.cbLazy(expr <= len(route) - 1)  # eq:subTourElim

                    # else:
                    #     for i, j in selectedEdges:
                    #         m.cbLazy(m._o_ki[k, i] + 1 <= m._o_ki[k, j])
                    #     for i in selectedTasks:
                    #         m.cbLazy(m._o_ki[k, 'p%d' % i] <= m._o_ki[k, 'd%d' % i])





                        # bigM1 = m._bigM1
                        # ptNodes.pop(ptNodes.index('ori%d' % k))
                        # for i in ptNodes:
                        #     #  # eq:initOrder
                        #     m.cbLazy(2 <= o_ki[k, i])
                        #     m.cbLazy(o_ki[k, i] <= bigM1)
                        #     for j in ptNodes:
                        #         if m.cbGetSolution(m._x_bkij[b, k, i, j]) > 0.5:
                        #             m.cbLazy(o_ki[k, i] + 1 <= o_ki[k, j])
                        # for i in T:
                        #     if m.cbGetSolution(m._z_bi[b, i]) > 0.5:
                        #         m.cbLazy(o_ki[k, 'p%d' % i] <= o_ki[k, 'd%d' % i])

    #
    B, cB_M, cB_P = list(map(prmt.get, ['B', 'cB_M', 'cB_P']))
    T, P, D, N = list(map(prmt.get, ['T', 'P', 'D', 'N']))
    K, w_k = list(map(prmt.get, ['K', 'w_k']))
    t_ij, _delta = list(map(prmt.get, ['t_ij', '_delta']))
    cW = prmt['cW']
    #
    bigM1 = len(N) + 2
    bigM2 = len(N) * max(t_ij.values())
    #
    # Define decision variables
    #
    EX = Model('EX2')
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
    #
    #
    # Define constrains
    #
    #  Bundle
    #
    for i in T:  # eq:taskA
        EX.addConstr(quicksum(z_bi[b, i] for b in B) <= 1,
                    name='ta[%d]' % i)
    for b in B:
        EX.addConstr(quicksum(z_bi[b, i] for i in T) >= cB_M * g_b[b],
                     name='minTB1[%d]' % b)
        EX.addConstr(quicksum(z_bi[b, i] for i in T) <= cB_P * g_b[b],
                     name='minTB2[%d]' % b)
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


    for k in K:
        N_kM = N.union({'dest%d' % k})
        for i in N_kM:
            #  # eq:initOrder
            EX.addConstr(2 <= o_ki[k, i],
                         name='initO1[%d,%s]' % (k, i))
            EX.addConstr(o_ki[k, i] <= bigM1,
                         name='initO2[%d,%s]' % (k, i))
            for j in N_kM:
                for b in B:
                    #  # eq:subEli
                    EX.addConstr(o_ki[k, i] + 1 <= o_ki[k, j] + bigM1 * (1 - x_bkij[b, k, i, j]),
                         name='subEli[%d,%d,%s,%s]' % (b, k, i, j))
        for i in T:
            for b in B:
                #  # eq:pdSequence
                EX.addConstr(o_ki[k, 'p%d' % i] <= o_ki[k, 'd%d' % i] + bigM1 * (1 - z_bi[b, i]),
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
            EX.addConstr(LRS <= _delta + bigM2 * (1 - y_bk[b, k]),
                        name='df[%d,%d]' % (b, k))
    #     EX.addConstr(quicksum(w_k[k] * y_bk[b, k] for k in K) >= cW * g_b[b],
    #                  name='bg[%d]' % b)


    # for b in B:
    #     for k in K:
    #         EX.addConstr(y_bk[b, k] <= g_b[b],
    #                      name='temp[%d,%d]' % (b, k))
    #
    #     for k in K:
    #         kN = N.union({'ori%d' % k, 'dest%d' % k})
    #         for i in kN:
    #             for j in kN:
    #                 EX.addConstr(x_bkij[b, k, i, j] + x_bkij[b, k, j, i] <= 1,
    #                              name='temp2[%d,%d,%s,%s]' % (b, k, i, j))





    #
    # Run Gurobi (Optimization)
    #

    EX._B, EX._T, EX._K, EX._w_k, EX._cW = B, T, K, w_k, cW
    EX._z_bi, EX._x_bkij, EX._y_bk, EX._g_b = z_bi, x_bkij, y_bk, g_b









    EX.setParam('LazyConstraints', True)
    EX.setParam('Threads', NUM_CORES)
    if etc['logFile']:
        EX.setParam('LogFile', etc['logFile'])
    # EX.write('%s.mps' % prmt['problemName'])
    EX.optimize(callbackF)
    #
    if EX.status == GRB.Status.INFEASIBLE:
        EX.write('%s.lp' % prmt['problemName'])
        EX.computeIIS()
        EX.write('%s.ilp' % prmt['problemName'])
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
            logContents += '\t Cpu Time: %f\n' % eliCpuTime
            logContents += '\t Wall Time: %f\n' % eliWallTime
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
        sol = {
               'B': B, 'T': T, 'K': K, 'N': N,
               #
               'z_bi': _z_bi, 'y_bk': _y_bk, 'g_b': _g_b,
               'o_ki': _o_ki, 'x_bkij': _x_bkij}
        with open(etc['solFilePKL'], 'wb') as fp:
            pickle.dump(sol, fp)


if __name__ == '__main__':
    from problems import euclideanDistEx0
    from mrtScenario import mrtS1
    #
    # prmt = euclideanDistEx0()
    prmt = mrtS1()
    problemName = prmt['problemName']
    #
    etc = {'solFilePKL': opath.join('_temp', 'sol_%s_EX21.pkl' % problemName),
           'solFileCSV': opath.join('_temp', 'sol_%s_EX21.csv' % problemName),
           'solFileTXT': opath.join('_temp', 'sol_%s_EX21.txt' % problemName),
           'logFile': opath.join('_temp', '%s_EX21.log' % problemName),
           'itrFileCSV': opath.join('_temp', '%s_itrEX21.csv' % problemName),
           }
    #
    run(prmt, etc)
