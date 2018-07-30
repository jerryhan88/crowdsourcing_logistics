import os.path as opath
import multiprocessing
import time
import pickle
import numpy as np

from gurobipy import *
#
from RMP import generate_RMP
from CWL1 import itr2file
from CWL2 import LS_run
from _util_logging import write_log, res2file
from _util_cython import gen_cFile
prefix = 'GH'
gen_cFile(prefix)
from GH import run as GH_run

NUM_CORES = multiprocessing.cpu_count()
LOG_INTER_RESULTS = False
EPSILON = 0.00001


def run(prmt, etc=None):
    startCpuTime, startWallTime = time.clock(), time.time()
    if 'TimeLimit' not in etc:
        etc['TimeLimit'] = 1e400
    etc['startTS'] = startCpuTime
    etc['startCpuTime'] = startCpuTime
    etc['startWallTime'] = startWallTime
    itr2file(etc['itrFileCSV'])
    #
    cwl_inputs = {}
    T, cB_M, cB_P, K, bB = [prmt.get(k) for k in ['T', 'cB_M', 'cB_P', 'K', 'bB']]
    #
    C, sC, p_c, e_ci = [], set(), [], []
    TB = set()
    s_ck = {}
    for i in T:
        c = len(C)
        iP, iM = 'p%d' % i, 'd%d' % i
        for k in K:
            kP, kM = 'ori%d' % k, 'dest%d' % k
            s_ck[c, k] = [kP, iP, iM, kM]
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
    bc, s_bk = GH_run(prmt)
    for b in range(bB):
        Ts = bc[b]
        if len(Ts) < cB_M:
            continue
        c = len(C)
        for k in K:
            s_ck[c, k] = s_bk[b, k]
        C.append(Ts)
        sC.add(frozenset(tuple(Ts)))
        #
        p_c.append(len(Ts))
        #
        vec = [0 for _ in range(len(T))]
        for i in Ts:
            vec[i] = 1
        e_ci.append(vec)
    #
    cwl_inputs['C'] = C
    cwl_inputs['sC'] = sC
    cwl_inputs['p_c'] = p_c
    cwl_inputs['e_ci'] = e_ci
    cwl_inputs['TB'] = TB
    cwl_inputs['s_ck'] = s_ck
    #
    RMP, q_c, taskAC, numBC = generate_RMP(prmt, cwl_inputs)
    #
    counter, is_terminated = 0, False
    while True:
        if len(C) == len(T) ** 2 - 1:
            break
        LRMP = RMP.relax()
        LRMP.setParam('Threads', NUM_CORES)
        LRMP.setParam('OutputFlag', False)
        LRMP.optimize()
        if LRMP.status == GRB.Status.INFEASIBLE:
            logContents = 'Relaxed model is infeasible!!\n'
            logContents += 'No solution!\n'
            write_log(etc['logFile'], logContents)
            #
            LRMP.write('%s.lp' % prmt['problemName'])
            LRMP.computeIIS()
            LRMP.write('%s.ilp' % prmt['problemName'])
            assert False
        #
        pi_i = [LRMP.getConstrByName("taskAC[%d]" % i).Pi for i in T]
        mu = LRMP.getConstrByName("numBC").Pi
        cwl_inputs['pi_i'] = pi_i
        cwl_inputs['mu'] = mu
        #
        if LOG_INTER_RESULTS:
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
            if rc < -EPSILON:
                TB.add(c)
                continue
            if rc < minRC:
                minRC = rc
                c0 = c
        if c0 == -1:
            break
        cwl_inputs['c0'] = c0
        #
        rc_Ts1_seqs = LS_run(prmt, cwl_inputs)
        rc_Ts1 = [[o[0], o[1]] for o in rc_Ts1_seqs]
        if time.clock() - etc['startTS'] > etc['TimeLimit']:
            break
        #
        eliCpuTimeP, eliWallTimeP = time.clock() - etc['startCpuTime'], time.time() - etc['startWallTime']
        itr2file(etc['itrFileCSV'], [counter, '%.2f' % eliCpuTimeP, '%.2f' % eliWallTimeP,
                                     len(cwl_inputs['C']), len(cwl_inputs['TB']),
                                     '%.2f' % LRMP.objVal, C[c0], '%.2f' % minRC, str(rc_Ts1)])
        if len(rc_Ts1_seqs) == 0:
            TB.add(c0)
        else:
            is_updated = False
            for rc, Ts1, seqs in rc_Ts1_seqs:
                if rc < 0:
                    continue
                is_updated = True
                vec = [0 for _ in range(len(T))]
                for i in Ts1:
                    vec[i] = 1
                if sum(vec) < cB_M:
                    p = 0
                else:
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
                for k in K:
                    s_ck[len(C), k] = seqs[k]
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


def test():
    from mrtScenario import mrtS1, mrtS2
    #
    prmt = mrtS1()
    # prmt = mrtS2()
    problemName = prmt['problemName']
    #
    etc = {'solFilePKL': opath.join('_temp', 'sol_%s_CWL2.pkl' % problemName),
           'solFileCSV': opath.join('_temp', 'sol_%s_CWL2.csv' % problemName),
           'solFileTXT': opath.join('_temp', 'sol_%s_CWL2.txt' % problemName),
           'logFile': opath.join('_temp', '%s_CWL2.log' % problemName),
           'itrFileCSV': opath.join('_temp', '%s_itrCWL2.csv' % problemName),
           }
    #
    run(prmt, etc)


if __name__ == '__main__':
    from mrtScenario import mrtS1, mrtS2
    #
    prmt = mrtS1()
    # prmt = mrtS2()
    problemName = prmt['problemName']
    #
    etc = {'solFilePKL': opath.join('_temp', 'sol_%s_CWL5.pkl' % problemName),
           'solFileCSV': opath.join('_temp', 'sol_%s_CWL5.csv' % problemName),
           'solFileTXT': opath.join('_temp', 'sol_%s_CWL5.txt' % problemName),
           'logFile': opath.join('_temp', '%s_CWL5.log' % problemName),
           'itrFileCSV': opath.join('_temp', '%s_itrCWL5.csv' % problemName),
           }
    #
    run(prmt, etc)
