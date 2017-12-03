import pickle
import numpy as np
#
from optRouting import run as optR_run
from problems import *
#
ifpath = 'nt15-np20-nb5-tv4-td5.pkl'

with open(ifpath, 'rb') as fp:
    inputs = pickle.load(fp)


travel_time, \
flows, paths, \
tasks, rewards, volumes, \
num_bundles, volume_th, detour_th = inputs

bB, \
T, r_i, v_i, _lambda, P, D, N, \
K, w_k, t_ij, _delta = convert_input4MathematicalModel(*inputs)


print(tasks)
locations = set()
for pp, dp in tasks:
    locations.add(pp)
    locations.add(dp)

vectors = []
for pp, dp in tasks:
    v = [0 for _ in range(len(locations))]
    v[pp] = 1
    v[dp] = -1
    vectors.append(np.array(v))

distances = {}
for i0 in range(len(tasks)):
    dist_task = []
    for i1 in range(len(tasks)):
        if i0 == i1:
            continue
        v0, v1 = vectors[i0], vectors[i1]
        dist_task.append((np.linalg.norm(v0 - v1), i1))
    distances[i0] = sorted(dist_task)

r_i = np.array(rewards)
pi_i = np.array([1.179, 0.125, 0.482, 0.857, 0.0, 2.268, 2.0, -3.429, 0.964, -0.125, 0.0, 1.161, 0.0, 2.339, 0.0])
mu = 3.429
i0 = np.argmax(r_i - pi_i)


grbSettingOP = {}
b0 = [i0]
br = sum([r_i[i] for i in b0])
p0 = 0
for k, w in enumerate(w_k):
    probSetting = {'b': b0, 'k': k, 't_ij': t_ij}
    detourTime, route = optR_run(probSetting, grbSettingOP)
    if detourTime <= _delta:
        p0 += w * br
p0 -= sum([pi_i[i] for i in b0])
p0 -= mu
while True:
    minDist = distances[i0][0][0]
    maxModiReward, i1 = -1e400, None
    for dist, i in distances[i0]:
        if minDist < dist:
            break
        if maxModiReward < r_i[i] - pi_i[i]:
            maxModiReward = r_i[i] - pi_i[i]
            i1 = i
    if i1 in b0:
        break
    b1 = b0[:] + [i1]
    if _lambda < sum(v_i[i]for i in b1):
        break
    br = sum([r_i[i] for i in b1])
    p1 = 0
    for k, w in enumerate(w_k):
        probSetting = {'b': b1, 'k': k, 't_ij': t_ij}
        detourTime, route = optR_run(probSetting, grbSettingOP)
        if detourTime <= _delta:
            p1 += w * br
    p1 -= sum([pi_i[i] for i in b0])
    p1 -= mu
    if p1 < p0:
        break
    p0 = p1
    b0 = b1

print('0', p0, b0)
