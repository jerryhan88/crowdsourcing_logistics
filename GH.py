import time
#
from _util import log2file, res2file
from problems import *


def run(probSetting, etcSetting, returnSol=False):
    startCpuTime, startWallTime = time.clock(), time.time()
    #
    inputs = convert_p2i(*probSetting['problem'])
    bB = inputs['bB']
    T, r_i, v_i, _lambda = list(map(inputs.get, ['T', 'r_i', 'v_i', '_lambda']))
    K, w_k = list(map(inputs.get, ['K', 'w_k']))
    t_ij, _delta = list(map(inputs.get, ['t_ij', '_delta']))
    #
    a_t = []
    for i in T:
        a = 0
        iP, iM = 'p%d' % i, 'd%d' % i
        for k in K:
            kP, kM = 'ori%d' % k, 'dest%d' % k
            d_ik = (t_ij[kP, iP] + t_ij[iP, iM] + t_ij[iM, kM]) / t_ij[kP, kM]
            a += r_i[i] * (w_k[k] / d_ik)
        a_t.append(a)
    #
    B = list(range(bB))
    bundles = [[] for _ in B]
    B_seq = [{} for _ in B]
    a_b = [0 for _ in B]
    while T:
        bundle_updatedX = [False for _ in B]
        for b in B:
            if _lambda <= sum(v_i[i] for i in bundles[b]):
                bundle_updatedX[b] = True
                continue
            best_a_b, best_bSeq, max_i = -1e400, None, None
            if not bundles[b]:
                max_a_t = -1e400
                for i in T:
                    if max_a_t < a_t[i]:
                        max_a_t, max_i = a_t[i], i
                best_a_b, best_bSeq = estimateBundleAtt(K, w_k, r_i, t_ij, _delta, bundles[b], B_seq[b], max_i)
            else:
                for i in T:
                    if _lambda < sum(v_i[i] for i in bundles[b]) + v_i[i]:
                        continue
                    est_a_b, est_bSeq = estimateBundleAtt(K, w_k, r_i, t_ij, _delta, bundles[b], B_seq[b], i)
                    if best_a_b < est_a_b:
                        best_a_b, best_bSeq, max_i = est_a_b, est_bSeq, i
            if a_b[b] < best_a_b:
                bundles[b].append(max_i)
                a_b[b], B_seq[b] = best_a_b, best_bSeq
                T.pop(T.index(max_i))
            else:
                bundle_updatedX[b] = True
            #
            if not T:
                break
        if sum(bundle_updatedX) == bB:
            break
    #
    endCpuTime, endWallTime = time.clock(), time.time()
    eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
    #
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
    logContents += '\t ObjV: %.3f\n' % sum(a_b)
    logContents += '\t chosen B.: %s\n' % str(bundles)
    #
    logContents += '\n'
    for b in B:
        br = sum([r_i[i] for i in bundles[b]])
        p = 0
        for k in K:
            kP, kM = 'ori%d' % k, 'dest%d' % k
            detourTime = calc_detourTime(kP, kM, B_seq[b][k], t_ij)
            route = [kP] + B_seq[b][k] + [kM]
            if detourTime <= _delta:
                p += w_k[k] * br
                logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w_k[k], detourTime, 1, str(route))
            else:
                logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w_k[k], detourTime, 0, str(route))
            logContents += '\t\t\t\t\t\t %.3f\n' % p
    logContents += '======================================================================================\n'
    log2file(etcSetting['LogFile'], logContents)
    try:
        res2file(etcSetting['ResFile'], sum(a_b), None, eliCpuTime, eliWallTime)
    except:
        res2file(etcSetting['ResFile'], -1, -1, eliCpuTime, eliWallTime)
    #
    if returnSol:
        return sum(a_b), bundles


def estimateBundleAtt(K, w_k, r_i, t_ij, _delta, b, bSeq, est_i):
    iP, iM = 'p%d' % est_i, 'd%d' % est_i
    ws, est_bSeq = 0, {}
    if not b:
        for k in K:
            kP, kM = 'ori%d' % k, 'dest%d' % k
            detourTime = t_ij[kP, iP] + t_ij[iP, iM] + t_ij[iM, kM]
            detourTime -= t_ij[kP, kM]
            est_bSeq[k] = [iP, iM]
            if detourTime <= _delta:
                ws += w_k[k]
    else:
        for k in K:
            kP, kM = 'ori%d' % k, 'dest%d' % k
            least_detourTime, best_seq = 1e400, None
            for i in range(len(bSeq[k])):
                if i == len(bSeq[k]) - 1:
                    j = i
                    #
                    new_seq = bSeq[k][:]
                    new_seq.insert(i, iP)
                    new_seq.insert(j + 1, iM)
                    detourTime = calc_detourTime(kP, kM, new_seq, t_ij)
                    if detourTime < least_detourTime:
                        least_detourTime, best_seq = detourTime, new_seq
                else:
                    for j in range(i, len(bSeq[k])):
                        new_seq = bSeq[k][:]
                        new_seq.insert(i, iP)
                        new_seq.insert(j + 1, iM)
                        detourTime = calc_detourTime(kP, kM, new_seq, t_ij)
                        if detourTime < least_detourTime:
                            least_detourTime, best_seq = detourTime, new_seq
            est_bSeq[k] = best_seq
            if least_detourTime <= _delta:
                ws += w_k[k]
    #
    est_a_b = (sum(r_i[i] for i in b) + r_i[est_i]) * ws
    return est_a_b, est_bSeq


def calc_detourTime(kP, kM, seq, t_ij):
    detourTime = t_ij[kP, seq[0]] + \
             sum(t_ij[seq[i], seq[i + 1]] for i in range(len(seq) - 1)) + \
             t_ij[seq[-1], kM]
    detourTime -= t_ij[kP, kM]
    return detourTime


if __name__ == '__main__':
    import os.path as opath
    from problems import paperExample
    #
    problem = paperExample()
    probSetting = {'problem': problem}
    ghLogF = opath.join('_temp', 'paperExample_GH.log')
    ghResF = opath.join('_temp', 'paperExample_GH.csv')
    etcSetting = {'LogFile': ghLogF,
                  'ResFile': ghResF}
    run(probSetting, etcSetting)
