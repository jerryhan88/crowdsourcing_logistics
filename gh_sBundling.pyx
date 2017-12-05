import pickle
import numpy as np
#
from optRouting import run as or_run
from problems import *


def run(inputs, grbSetting={}):
    T, r_i, v_i, _lambda = list(map(inputs.get, ['T', 'r_i', 'v_i', '_lambda']))
    K, w_k = list(map(inputs.get, ['K', 'w_k']))
    t_ij, _delta = list(map(inputs.get, ['t_ij', '_delta']))
    N = inputs['N']
    _N = inputs['_N']
    _pi_i, mu = list(map(inputs.get, ['pi_i', 'mu']))
    pi_i = []
    for v in _pi_i:
        if v < 0:
            pi_i.append(0)
        else:
            pi_i.append(v)
    inclusiveC, exclusiveC = list(map(inputs.get, ['inclusiveC', 'exclusiveC']))
    #
    def calc_rc_get_dvs(b):
        _z_i, _y_k, _a_k, _o_ki, _x_kij, _s_PN = {}, {}, {}, {}, {}, {}
        #
        for i in T:
            _z_i[i] = 0
        for k in K:
            _y_k[k], _a_k[k] = 0, 0
        for k in K:
            kN = N.union({'ori%d' % k, 'dest%d' % k})
            for i in kN:
                _o_ki[k, i] = 0
                for j in kN:
                    _x_kij[k, i, j] = 0
        for ii, pair in enumerate(inclusiveC):
            sP, sN = 'sP[%d]' % ii, 'sN[%d]' % ii
            if set(pair).difference(set(b)):
                _s_PN[sP], _s_PN[sN] = 1, 0
            else:
                _s_PN[sP], _s_PN[sN] = 0, 1
        #
        for i in b:
            _z_i[i] = 1
        rs = sum([r_i[i] for i in b])
        ws = 0
        for k, w in enumerate(w_k):
            kP, kM = 'ori%d' % k, 'dest%d' % k
            probSetting = {'b': b, 'k': k, 't_ij': t_ij}
            detourTime, route = or_run(probSetting, grbSetting)
            for i in range(len(route) - 1):
                _o_ki[k, route[i]] = i + 1
                _x_kij[k, route[i], route[i + 1]] = 1
            assert route[-1] == kM
            _o_ki[k, kM] = len(route)
            _x_kij[k, kM, kP] = 1
            if detourTime <= _delta:
                _y_k[k] = 1
                _a_k[k] = rs
                ws += w
        rc = ws * rs
        rc -= sum([pi_i[i] for i in b])
        rc -= mu
        dvs = {'_z_i': _z_i, '_y_k': _y_k, '_a_k': _a_k,
               '_o_ki': _o_ki, '_x_kij': _x_kij, '_s_PN': _s_PN}
        return rc, dvs
    #
    L = sorted(list(set(_N.values())))
    vectors = []
    for i in T:
        v = [0 for _ in range(len(L))]
        iP, iM = 'p%d' % i, 'd%d' % i
        v[L.index(_N[iP])] = 1
        v[L.index(_N[iM])] = -1
        vectors.append(np.array(v))
    priorities = {}
    for i0 in T:
        dist_rP_task = []
        for i1 in T:
            if i0 == i1:
                continue
            v0, v1 = vectors[i0], vectors[i1]
            dist_rP_task.append((np.linalg.norm(v0 - v1), r_i[i1] - pi_i[i1], i1))
        priorities[i0] = sorted(dist_rP_task)
    #
    i0 = np.argmax(np.array(r_i) - np.array(pi_i))
    b0 = [i0]
    rc0, dvs0 = calc_rc_get_dvs(b0)
    while True:
        for ii, pair in enumerate(inclusiveC):
            if i0 in pair:
                i1 = pair[1] if pair[0] == i0 else pair[0]
                if i0 in b0 and i1 in b0:
                    continue
                break
        else:
            i1 = None
            for dist, rP, i in priorities[i0]:
                for pair in exclusiveC:
                    if i0 in pair and i in pair:
                        break
                else:
                    i1 = i
                if i1 is not None:
                    break
        if i1 in b0:
            break
        b1 = b0[:] + [i1]
        if _lambda < sum(v_i[i] for i in b1):
            break
        rc1, dvs1 = calc_rc_get_dvs(b1)
        if rc1 < rc0:
            break
        i0, b0 = i1, b1
        rc0, dvs0 = rc1, dvs1
    return rc0, dvs0


if __name__ == '__main__':
    pass