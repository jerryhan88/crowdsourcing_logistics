import numpy as np
#
from PD import calc_expectedProfit


def run(ori_inputs, cwl_inputs, grbSetting):
    T, v_i, _lambda = [ori_inputs.get(k) for k in ['T', 'v_i', '_lambda']]
    #
    pi_i, mu = [cwl_inputs.get(k) for k in ['pi_i', 'mu']]
    C, sC, c0 = [cwl_inputs.get(k) for k in ['C', 'sC', 'c0']]
    #
    Ts0 = C[c0]
    max_rc, max_i = -1e400, None
    for i0 in T:
        if i0 in Ts0:
            continue
        Ts1 = Ts0[:] + [i0]
        if frozenset(tuple(Ts1)) in sC:
            continue
        if sum(v_i[i] for i in Ts1) > _lambda:
            continue
        vec = [0 for _ in range(len(T))]
        for i in Ts1:
            vec[i] = 1
        rc = 0 - (np.array(vec) * np.array(pi_i)).sum() - mu
        rc += calc_expectedProfit(ori_inputs, grbSetting, Ts1)
        if max_rc < rc:
            max_rc = rc
            max_i = i0
    #
    return max_rc, Ts0[:] + [max_i]
