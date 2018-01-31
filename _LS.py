import time
from random import randrange
import numpy as np
#
from PD import run as PD_run


def run(probSetting, etcSetting, grbSetting):
    inputs = probSetting['inputs']
    T, r_i, v_i, _lambda = [inputs.get(k) for k in ['T', 'r_i', 'v_i', '_lambda']]
    K, w_k, t_ij, _delta = [inputs.get(k) for k in ['K', 'w_k', 't_ij', '_delta']]
    #
    pi_i, mu = [probSetting.get(k) for k in ['pi_i', 'mu']]
    C, sC, c0 = [probSetting.get(k) for k in ['C', 'sC', 'c0']]
    #
    bc0 = C[c0]
    #
    max_p, max_i = -1e400, None
    for i0 in T:
        if i0 in bc0:
            continue
        bc1 = bc0[:] + [i0]
        if frozenset(tuple(bc1)) in sC:
            continue
        vec = [0 for _ in range(len(T))]
        for i in bc1:
            vec[i] = 1
        br = sum([r_i[i] for i in bc1])
        p = 0 - (np.array(vec) * np.array(pi_i)).sum() - mu
        for k in K:
            probSetting = {'bc': bc1, 'k': k, 't_ij': t_ij}
            detourTime, route = PD_run(probSetting, grbSetting)
            if detourTime <= _delta:
                p += w_k[k] * br
        if max_p < p:
            max_p = p
            max_i = i0
    #
    return max_p, bc0[:] + [max_i]
