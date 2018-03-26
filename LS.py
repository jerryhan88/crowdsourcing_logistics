import time
from random import randrange
import numpy as np
#
from PD import calc_expectedProfit


def run(probSetting, etcSetting, grbSetting):
    inputs = probSetting['inputs']
    T, v_i, _lambda = [inputs.get(k) for k in ['T', 'v_i', '_lambda']]
    #
    pi_i, mu = [probSetting.get(k) for k in ['pi_i', 'mu']]
    C, sC, c0 = [probSetting.get(k) for k in ['C', 'sC', 'c0']]
    #
    bc0 = C[c0]
    #
    max_rc, max_i = -1e400, None
    for i0 in T:
        if i0 in bc0:
            continue
        bc1 = bc0[:] + [i0]
        if frozenset(tuple(bc1)) in sC:
            continue
        if sum(v_i[i] for i in bc1) > _lambda:
            continue
        vec = [0 for _ in range(len(T))]
        for i in bc1:
            vec[i] = 1
        rc = 0 - (np.array(vec) * np.array(pi_i)).sum() - mu
        rc += calc_expectedProfit(probSetting, grbSetting, bc1)

        if max_rc < rc:
            max_rc = rc
            max_i = i0
    #
    return max_rc, bc0[:] + [max_i]
