from problems import *


def run(problem):
    inputs = convert_p2i(*problem)
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
    B = [[] for _ in range(bB)]
    B_seq = [{} for _ in range(bB)]
    a_b = [0 for _ in range(bB)]
    while T:
        for b in range(len(B)):
            if _lambda <= sum(v_i[i] for i in B[b]):
                continue
            best_a_b, best_bSeq, max_i = -1e400, None, None
            if not B[b]:
                max_a_t = -1e400
                for i in T:
                    if max_a_t < a_t[i]:
                        max_a_t, max_i = a_t[i], i
                best_a_b, best_bSeq = estimateBundleAtt(K, w_k, r_i, t_ij, _delta, B[b], B_seq[b], max_i)
            else:
                for i in T:
                    if _lambda < sum(v_i[i] for i in B[b]) + v_i[i]:
                        continue
                    est_a_b, est_bSeq = estimateBundleAtt(K, w_k, r_i, t_ij, _delta, B[b], B_seq[b], i)
                    if best_a_b < est_a_b:
                        best_a_b, best_bSeq, max_i = est_a_b, est_bSeq, i
            #
            if max_i is not None:
                B[b].append(max_i)
                a_b[b], B_seq[b] = best_a_b, best_bSeq
            #
            if not T:
                break
            T.pop(T.index(max_i))
    #
    return sum(a_b), B


def estimateBundleAtt(K, w_k, r_i, t_ij, _delta, b, bSeq, est_i):
    def calc_detour(kP, kM, seq):
        detour = t_ij[kP, seq[0]] + \
                 sum(t_ij[seq[i], seq[i + 1]] for i in range(len(seq) - 1)) + \
                 t_ij[seq[-1], kM]
        detour -= t_ij[kM, kP]
        return detour
    #
    iP, iM = 'p%d' % est_i, 'd%d' % est_i
    ws, est_bSeq = 0, {}
    if not b:
        for k in K:
            kP, kM = 'ori%d' % k, 'dest%d' % k
            detour = t_ij[kP, iP] + t_ij[iP, iM] + t_ij[iM, kM]
            detour -= t_ij[kM, kP]
            est_bSeq[k] = [iP, iM]
            if detour <= _delta:
                ws += w_k[k]
    else:
        for k in K:
            kP, kM = 'ori%d' % k, 'dest%d' % k
            least_detour, best_seq = 1e400, None
            for i in range(len(bSeq[k])):
                if i == len(bSeq[k]) - 1:
                    j = i
                    #
                    new_seq = bSeq[k][:]
                    new_seq.insert(i, iP)
                    new_seq.insert(j + 1, iM)
                    detour = calc_detour(kP, kM, new_seq)
                    if detour < least_detour:
                        least_detour, best_seq = detour, new_seq
                else:
                    for j in range(i, len(bSeq[k])):
                        new_seq = bSeq[k][:]
                        new_seq.insert(i, iP)
                        new_seq.insert(j + 1, iM)
                        detour = calc_detour(kP, kM, new_seq)
                        if detour < least_detour:
                            least_detour, best_seq = detour, new_seq
            est_bSeq[k] = best_seq
            if least_detour <= _delta:
                ws += w_k[k]
    #
    est_a_b = (sum(r_i[i] for i in b) + r_i[est_i]) * ws
    return est_a_b, est_bSeq


def test():
    # from problems import *
    # print(run(ex2()))
    import pickle

    ifpath = 'nt05-np12-nb2-tv3-td4.pkl'
    with open(ifpath, 'rb') as fp:
        inputs = pickle.load(fp)

    objV, B = run(inputs)
    print(objV, B)


if __name__ == '__main__':
    test()
