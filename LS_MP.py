import time
import multiprocessing
import numpy as np
#
from PD import run as PD_run


def eval_bcs(probSetting, grbSetting, wsDict, bcs):
    inputs = probSetting['inputs']
    K, w_k, t_ij, _delta = [inputs.get(k) for k in ['K', 'w_k', 't_ij', '_delta']]
    for bcId, bc1 in bcs:
        ws = 0
        for k in K:
            probSetting = {'bc': bc1, 'k': k, 't_ij': t_ij}
            detourTime, route = PD_run(probSetting, grbSetting)
            if detourTime <= _delta:
                ws += w_k[k]
        wsDict[bcId] = ws


def run(probSetting, etcSetting, grbSetting):
    inputs = probSetting['inputs']
    T, r_i, v_i, _lambda = [inputs.get(k) for k in ['T', 'r_i', 'v_i', '_lambda']]
    #
    pi_i, mu = [probSetting.get(k) for k in ['pi_i', 'mu']]
    C, sC, c0 = [probSetting.get(k) for k in ['C', 'sC', 'c0']]
    #
    id_bc, bcsSeparation = {}, [[] for _ in range(etcSetting['numPros'])]
    bc0 = C[c0]
    for i0 in T:
        if i0 in bc0:
            continue
        bc1 = bc0[:] + [i0]
        if frozenset(tuple(bc1)) in sC:
            continue
        bcID = len(id_bc)
        id_bc[bcID] = bc1
        procID = bcID % etcSetting['numPros']
        bcsSeparation[procID].append((bcID, bc1))
    #
    ps = []
    wsDict = multiprocessing.Manager().dict()
    for bcs in bcsSeparation:
        if not bcs:
            continue
        p = multiprocessing.Process(target=eval_bcs, args=(probSetting, grbSetting, wsDict, bcs))
        ps.append(p)
        p.start()
    for p in ps:
        p.join()
    #
    max_p, max_bc = -1e400, None
    for bcID, ws in wsDict.items():
        bc1 = id_bc[bcID]
        br, p = 0, 0
        for i in bc1:
            br += r_i[i]
            p -= pi_i[i]
        p -= mu
        p += ws * br
        if max_p < p:
            max_p = p
            max_bc = bc1
    #
    return max_p, max_bc
