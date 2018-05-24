import multiprocessing
import time
import pickle
import numpy as np
from gurobipy import *
#
from _util import write_log, itr2file, res2file

from RMP import generate_RMP
from PD import get_wsFeasiblity



NUM_CORES = multiprocessing.cpu_count()



def LS_run(prmt, cwl_inputs):
    T, cB_P = [prmt.get(k) for k in ['T', 'cB_P']]
    #
    pi_i, mu = [cwl_inputs.get(k) for k in ['pi_i', 'mu']]
    C, sC, c0 = [cwl_inputs.get(k) for k in ['C', 'sC', 'c0']]
    #
    Ts0 = C[c0]
    rc_Ts1 = []
    for i0 in T:
        if i0 in Ts0:
            continue
        Ts1 = Ts0[:] + [i0]
        if frozenset(tuple(Ts1)) in sC:
            continue
        if len(Ts1) > cB_P:
            continue
        vec = [0 for _ in range(len(T))]
        for i in Ts1:
            vec[i] = 1
        rc = len(Ts1) - (np.array(vec) * np.array(pi_i)).sum() - mu
        wsFeasiblity = get_wsFeasiblity(prmt, Ts1)
        if wsFeasiblity:
            rc_Ts1.append([rc, Ts1])
    #
    return rc_Ts1


def run(prmt, etc=None):
    startCpuTime, startWallTime = time.clock(), time.time()
    if 'TimeLimit' not in etc:
        etc['TimeLimit'] = 1e400
    etc['startTS'] = startCpuTime
    etc['startCpuTime'] = startCpuTime
    etc['startWallTime'] = startWallTime
    problemName = prmt['problemName']
    itr2file(etc['itrFileCSV'])
    #
    cwl_inputs = {}
    #
    # Generate initial singleton bundles
    #
    T, cB_P = [prmt.get(k) for k in ['T', 'cB_P']]
    C, sC, p_c, e_ci, TB = [], set(), [], [], set()
    for i in T:
        Ts = [i]
        C.append(Ts)
        sC.add(frozenset(tuple(Ts)))
        #
        p_c.append(0)
        #
        vec = [0 for _ in range(len(T))]
        vec[i] = 1
        e_ci.append(vec)
    #
    cwl_inputs['C'] = C
    cwl_inputs['sC'] = sC
    cwl_inputs['p_c'] = p_c
    cwl_inputs['e_ci'] = e_ci
    cwl_inputs['TB'] = TB
    #
    RMP, q_c, taskAC, numBC = generate_RMP(prmt, cwl_inputs)
    #
    counter, is_terminated = 0, False
    while True:
        if len(C) == len(T) ** 2 - 1:
            break
        LRMP = RMP.relax()
        LRMP.setParam('Threads', NUM_CORES)
        if etc['logFile']:
            LRMP.setParam('LogFile', etc['logFile'])
        LRMP.optimize()
        if LRMP.status == GRB.Status.INFEASIBLE:
            logContents = 'Relaxed model is infeasible!!\n'
            logContents += 'No solution!\n'
            write_log(etc['logFile'], logContents)
            #
            LRMP.write('%s.lp' % problemName)
            LRMP.computeIIS()
            LRMP.write('%s.ilp' % problemName)
            assert False
        #
        pi_i = [LRMP.getConstrByName("taskAC[%d]" % i).Pi for i in T]
        mu = LRMP.getConstrByName("numBC").Pi
        cwl_inputs['pi_i'] = pi_i
        cwl_inputs['mu'] = mu
        #
        logContents = 'Start %dth iteration\n' % counter
        logContents += '\t Columns\n'
        logContents += '\t\t # of columns %d\n' % len(cwl_inputs['C'])
        logContents += '\t\t %s\n' % str(cwl_inputs['C'])
        logContents += '\t\t %s\n' % str(['%.2f' % v for v in cwl_inputs['p_c']])
        logContents += '\t Relaxed objVal\n'
        logContents += '\t\t z: %.2f\n' % LRMP.objVal
        logContents += '\t\t RC: %s\n' % str(['%.2f' % LRMP.getVarByName("q[%d]" % c).RC for c in range(len(C))])
        logContents += '\t Dual V\n'
        logContents += '\t\t Pi: %s\n' % str(['%.2f' % v for v in pi_i])
        logContents += '\t\t mu: %.2f\n' % mu
        write_log(etc['logFile'], logContents)
        #
        c0, minRC = -1, 1e400
        for rc, c in [(LRMP.getVarByName("q[%d]" % c).RC, c) for c in range(len(C))]:
            Ts = C[c]
            if c in TB:
                continue
            if len(Ts) == cB_P:
                continue
            if rc < minRC:
                minRC = rc
                c0 = c
        if c0 == -1:
            break
        cwl_inputs['c0'] = c0
        #
        startCpuTimeP, startWallTimeP = time.clock(), time.time()
        rc_Ts1 = LS_run(prmt, cwl_inputs)
        #
        if time.clock() - etc['startTS'] > etc['TimeLimit']:
            break
        #
        endCpuTimeP, endWallTimeP = time.clock(), time.time()
        eliCpuTimeP, eliWallTimeP = endCpuTimeP - startCpuTimeP, endWallTimeP - startWallTimeP
        #
        logContents = '%dth iteration\n' % counter
        logContents += '\t Cpu Time: %f\n' % eliCpuTimeP
        logContents += '\t Wall Time: %f\n' % eliWallTimeP
        write_log(etc['logFile'], logContents)
        itr2file(etc['itrFileCSV'], [counter, '%.2f' % eliCpuTimeP, '%.2f' % eliWallTimeP, '%.2f' % LRMP.objVal,
                                     C[c0], '%.2f' % minRC,
                                     str(rc_Ts1)])
        if len(rc_Ts1) == 0:
            TB.add(c0)
        else:
            is_updated = False
            for rc, Ts1 in rc_Ts1:
                if rc < 0:
                    continue
                is_updated = True
                vec = [0 for _ in range(len(T))]
                for i in Ts1:
                    vec[i] = 1
                p = rc + (np.array(vec) * np.array(pi_i)).sum() + mu
                C, p_c, e_ci, sC = list(map(cwl_inputs.get, ['C', 'p_c', 'e_ci', 'sC']))
                e_ci.append(vec)
                p_c.append(p)
                #
                col = Column()
                for i in range(len(T)):
                    if e_ci[len(C)][i] > 0:
                        col.addTerms(e_ci[len(C)][i], taskAC[i])
                col.addTerms(1, numBC)
                #
                q_c[len(C)] = RMP.addVar(obj=p_c[len(C)], vtype=GRB.BINARY, name="q[%d]" % len(C), column=col)
                C.append(Ts1)
                sC.add(frozenset(tuple(Ts1)))
                RMP.update()
                #
            if not is_updated:
                TB.add(c0)
        if len(C) == len(TB):
            break
        counter += 1
    #
    # Handle termination
    #
    RMP.setParam('Threads', NUM_CORES)
    RMP.optimize()
    #
    if etc and RMP.status != GRB.Status.INFEASIBLE:
        assert 'solFilePKL' in etc
        assert 'solFileCSV' in etc
        assert 'solFileTXT' in etc
        #
        q_c = [RMP.getVarByName("q[%d]" % c).x for c in range(len(C))]
        chosenC = [(C[c], '%.2f' % q_c[c]) for c in range(len(C)) if q_c[c] > 0.5]
        with open(etc['solFileTXT'], 'w') as f:
            endCpuTime, endWallTime = time.clock(), time.time()
            eliCpuTime = endCpuTime - etc['startCpuTime']
            eliWallTime = endWallTime - etc['startWallTime']
            logContents = 'Summary\n'
            logContents += '\t Cpu Time: %f\n' % eliCpuTime
            logContents += '\t Wall Time: %f\n' % eliWallTime
            logContents += '\t ObjV: %.3f\n' % RMP.objVal
            logContents += '\t Gap: %.3f\n' % RMP.MIPGap
            logContents += 'Chosen bundles\n'
            logContents += '%s\n' % str(chosenC)
            f.write(logContents)
            f.write('\n')
        #
        res2file(etc['solFileCSV'], RMP.objVal, RMP.MIPGap, eliCpuTime, eliWallTime)
        #
        _q_c = {c: RMP.getVarByName("q[%d]" % c).x for c in range(len(C))}
        sol = {
            'C': C, 'p_c': p_c, 'e_ci': e_ci,
            #
            'q_c': _q_c}
        with open(etc['solFilePKL'], 'wb') as fp:
            pickle.dump(sol, fp)


if __name__ == '__main__':
    import os.path as opath
    from problems import euclideanDistEx0
    from mrtScenario import mrtS1, mrtS2
    #
    # prmt = euclideanDistEx0()
    # prmt = mrtS1()
    prmt = mrtS2()
    problemName = prmt['problemName']
    #
    etc = {'solFilePKL': opath.join('_temp', 'sol_%s_CWL.pkl' % problemName),
           'solFileCSV': opath.join('_temp', 'sol_%s_CWL.csv' % problemName),
           'solFileTXT': opath.join('_temp', 'sol_%s_CWL.txt' % problemName),
           'logFile': opath.join('_temp', '%s_CWL.log' % problemName),
           'itrFileCSV': opath.join('_temp', '%s_itrCWL.csv' % problemName),
           }
    #
    run(prmt, etc)
