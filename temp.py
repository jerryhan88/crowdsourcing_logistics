import pickle
from problems import *
from optRouting import run as optR_run


ifpath = 'nt15-np20-nb5-tv4-td5.pkl'

with open(ifpath, 'rb') as fp:
    inputs = pickle.load(fp)


travel_time, \
flows, paths, \
tasks, rewards, volumes, \
num_bundles, volume_th, detour_th = inputs

print(tasks)
print(rewards)

B = [5, 7, 9]
print(sum([rewards[i] for i in B]))


bB, \
T, r_i, v_i, _lambda, P, D, N, \
K, w_k, t_ij, _delta = convert_input4MathematicalModel(*inputs)


B = [[0, 6], [1, 7, 9, 14], [2, 4, 11], [3, 10, 12], [5, 8, 13]]
b_rw = []
for b in B:
    br = sum([r_i[i] for i in b])
    ws = 0
    for k, w in enumerate(w_k):
        probSetting = {'b': b, 'k': k, 't_ij': t_ij}
        detourTime, route = optR_run(probSetting, {})
        if detourTime <= _delta:
            ws += w
    b_rw.append((br, ws))


print(b_rw)
# print(T)
#
# print(P)