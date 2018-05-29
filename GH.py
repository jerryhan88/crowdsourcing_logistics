import os.path as opath
import time
import pickle
#
from _util_logging import res2file
from PD_IH import run as PD_IH_run


def estimate_WS(prmt, gh_inputs, b, i0):
    K, w_k, _delta = list(map(prmt.get, ['K', 'w_k', '_delta']))
    s_bk = gh_inputs['s_bk']
    ws, seqs = 0.0, []
    for k in K:
        seq0 = s_bk[b, k]
        detourTime, seq1 = PD_IH_run(prmt, {'seq0': seq0, 'i0': i0})
        if detourTime <= _delta:
            ws += w_k[k]
        seqs.append(seq1)
    #
    return ws, seqs


def run(prmt, etc={}):
    startCpuTime, startWallTime = time.clock(), time.time()
    if 'TimeLimit' not in etc:
        etc['TimeLimit'] = 1e400
    etc['startTS'] = startCpuTime
    etc['startCpuTime'] = startCpuTime
    etc['startWallTime'] = startWallTime
    #
    bB = prmt['bB']
    T, cB_M, cB_P = [prmt.get(k) for k in ['T', 'cB_M', 'cB_P']]
    K, w_k = [prmt.get(k) for k in ['K', 'w_k']]
    t_ij, _delta, cW = [prmt.get(k) for k in ['t_ij', '_delta', 'cW']]
    #
    a_t = [0.0 for _ in T]
    for i in T:
        iP, iM = 'p%d' % i, 'd%d' % i
        for k in K:
            kP, kM = 'ori%d' % k, 'dest%d' % k
            detourTime = t_ij[kP, iP] + t_ij[iP, iM] + t_ij[iM, kM] - t_ij[kP, kM]
            if detourTime < _delta:
                a_t[i] += w_k[k]
    #
    B = list(range(bB))
    bc = [[] for _ in B]
    s_bk = {(b, k): ['ori%d' % k, 'dest%d' % k] for b in B for k in K}
    gh_inputs = {'bc': bc, 's_bk': s_bk}
    T = T[:]
    while T:
        updated = [False for _ in B]
        for b in B:
            if cB_P == len(bc[b]):
                continue
            if not bc[b]:
                max_a_t, max_i = -1e400, None
                for i0 in T:
                    if max_a_t < a_t[i0]:
                        max_a_t, max_i = a_t[i0], i0
                ws1, seqs1 = estimate_WS(prmt, gh_inputs, b, max_i)
                if ws1 > cW:
                    updated[b] = True
                    i0 = max_i
            else:
                for i0 in T:
                    ws1, seqs1 = estimate_WS(prmt, gh_inputs, b, i0)
                    if ws1 > cW:
                        updated[b] = True
                        break
            #
            if updated[b]:
                bc[b].append(i0)
                for k in K:
                    s_bk[b, k] = seqs1[k]
                T.pop(T.index(i0))
            if not T:
                break
        if sum(updated) == 0:
            break
    #
    # Handle termination
    #
    if 'solFileCSV' in etc:
        assert 'solFilePKL' in etc
        assert 'solFileCSV' in etc
        assert 'solFileTXT' in etc
        #
        objVal = sum(len(o) for o in bc if cB_M <= len(o))
        with open(etc['solFileTXT'], 'w') as f:
            endCpuTime, endWallTime = time.clock(), time.time()
            eliCpuTime = endCpuTime - etc['startCpuTime']
            eliWallTime = endWallTime - etc['startWallTime']
            logContents = 'Summary\n'
            logContents += '\t Cpu Time: %f\n' % eliCpuTime
            logContents += '\t Wall Time: %f\n' % eliWallTime
            logContents += '\t ObjV: %.3f\n' % objVal
            logContents += 'Chosen bundles\n'
            logContents += '%s\n' % str(bc)
            f.write(logContents)
            f.write('\n')
        res2file(etc['solFileCSV'], objVal, -1, eliCpuTime, eliWallTime)
        #
        sol = {'bc': bc}
        with open(etc['solFilePKL'], 'wb') as fp:
            pickle.dump(sol, fp)
    #
    return bc, s_bk


if __name__ == '__main__':
    from mrtScenario import mrtS1, mrtS2
    #
    prmt = mrtS1()
    # prmt = mrtS2()
    problemName = prmt['problemName']
    #
    etc = {'solFilePKL': opath.join('_temp', 'sol_%s_GH.pkl' % problemName),
           'solFileCSV': opath.join('_temp', 'sol_%s_GH.csv' % problemName),
           'solFileTXT': opath.join('_temp', 'sol_%s_GH.txt' % problemName),
           'logFile': opath.join('_temp', '%s_GH.log' % problemName),
           }
    #
    run(prmt, etc)
