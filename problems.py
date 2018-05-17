import os.path as opath
import time
import pickle
import numpy as np
from random import seed, random, randrange, uniform


def convert_prob2prmt(problemName,
                      flows, tasks,
                      numBundles, minTB, maxTB,
                      numLocs, travel_time, thDetour):
    B = list(range(numBundles))
    cB_M, cB_P = minTB, maxTB
    T = list(range(len(tasks)))
    iPs, iMs = list(zip(*[(tasks[i][1], tasks[i][2]) for i in T]))
    P, D = set(), set()
    _N = {}
    for i in T:
        P.add('p%d' % i)
        D.add('d%d' % i)
        #
        _N['p%d' % i] = iPs[i]
        _N['d%d' % i] = iMs[i]
    #
    # Path
    #
    c_k, temp_OD = [], []
    if type(flows) == list:
        for i in range(numLocs):
            for j in range(numLocs):
                if flows[i][j] == 0:
                    continue
                c = flows[i][j]
                temp_OD.append((i, j))
                c_k.append(c)
    else:
        assert type(flows) == dict
        for (i, j), c in flows.items():
            if c == 0:
                continue
            temp_OD.append((i, j))
            c_k.append(c)
    sumC = sum(c_k)
    K = list(range(len(c_k)))
    w_k = [c_k[k] / float(sumC) for k in K]
    _kP, _kM = list(zip(*[temp_OD[k] for k in K]))
    _delta = thDetour
    t_ij = {}
    for k in K:
        kP, kM = 'ori%d' % k, 'dest%d' % k
        t_ij[kP, kP] = travel_time[_kP[k]][_kP[k]]
        t_ij[kM, kM] = travel_time[_kM[k]][_kM[k]]
        t_ij[kP, kM] = travel_time[_kP[k]][_kM[k]]
        t_ij[kM, kP] = travel_time[_kM[k]][_kP[k]]
        for i in _N:
            t_ij[kP, i] = travel_time[_kP[k]][_N[i]]
            t_ij[i, kP] = travel_time[_N[i]][_kP[k]]
            #
            t_ij[kM, i] = travel_time[_kM[k]][_N[i]]
            t_ij[i, kM] = travel_time[_N[i]][_kM[k]]
    for i in _N:
        for j in _N:
            t_ij[i, j] = travel_time[_N[i]][_N[j]]
    N = set(_N.keys())
    #
    return {'problemName': problemName,
            'B': B, 'cB_M': cB_M, 'cB_P': cB_P,
            'K': K, 'w_k': w_k,
            'T': T, 'P': P, 'D': D, 'N': N,
            't_ij': t_ij, '_delta': _delta}


def get_dist_twoPoints(p0, p1):
    if type(p0) != np.ndarray:
        p0 = np.array(p0)
    if type(p1) != np.ndarray:
        p1 = np.array(p1)
    return np.sqrt(np.sum((p0 - p1) ** 2))


def get_dist_fLine(lp0, lp1, p2):
    vec01 = lp1 - lp0
    vec02 = p2 - lp0
    det = (vec01 ** 2).sum()
    a = (vec01 * vec02).sum() / det
    p3 = lp0 + a * vec01
    return get_dist_twoPoints(p2, p3)


def gen_pt_locs(clusters, min_dist_path, max_dist_path, min_dist_task, numPaths, numTasks):
    def get_fixedOri(center):
        ori = np.array([random(), random()])
        dist_co = get_dist_twoPoints(center, ori)
        while cr < dist_co:
            ori = np.array([random(), random()])
            dist_co = get_dist_twoPoints(center, ori)
        return ori, dist_co

    path_oridest = []
    for j in range(numPaths):
        i = j % len(clusters)
        cx, cy, cr = clusters[i]
        center = np.array([cx, cy])
        ori, dist_co = get_fixedOri(center)
        #
        dest = np.array([random(), random()])
        dist_cd = get_dist_twoPoints(center, dest)
        dist_od = get_dist_twoPoints(ori, dest)
        counter = 0
        while cr < dist_cd or not (min_dist_path < dist_od and dist_od < max_dist_path):
            counter += 1
            if counter == 10:
                ori, dist_co = get_fixedOri(center)
                counter = 0
            dest = np.array([random(), random()])
            dist_cd = get_dist_twoPoints(center, dest)
            dist_od = get_dist_twoPoints(ori, dest)
        path_oridest.append([ori, dest])
    #
    task_ppdp = []
    for j in range(numTasks):
        i = randrange(len(path_oridest))
        ori, dest = path_oridest[i]
        #
        pp = np.array([uniform(ori[0], dest[0]), uniform(ori[1], dest[1])])
        dist_fPath = get_dist_fLine(ori, dest, pp)
        while min_dist_task < dist_fPath:
            pp = np.array([uniform(ori[0], dest[0]), uniform(ori[1], dest[1])])
            dist_fPath = get_dist_fLine(ori, dest, pp)
        dp = np.array([uniform(ori[0], dest[0]), uniform(ori[1], dest[1])])
        dist_fPath = get_dist_fLine(ori, dest, dp)
        while min_dist_task < dist_fPath:
            dp = np.array([uniform(ori[0], dest[0]), uniform(ori[1], dest[1])])
            dist_fPath = get_dist_fLine(ori, dest, dp)
        task_ppdp.append([pp, dp])
    #
    return [(tuple(ori), tuple(dest)) for ori, dest in path_oridest], \
           [(tuple(pp), tuple(dp)) for pp, dp in task_ppdp]


def handle_locationNtt(flow_oridest, task_ppdp):
    lid_coord, coord_lid = {}, {}
    numLocs = 0
    for loc0, loc1 in flow_oridest + task_ppdp:
        for loc in [loc0, loc1]:
            if loc not in lid_coord:
                lid_coord[numLocs] = loc
                coord_lid[loc] = numLocs
                numLocs += 1
    travel_time = [[0 for _ in range(numLocs)] for _ in range(numLocs)]
    for pid0, coord0 in lid_coord.items():
        for pid1, coord1 in lid_coord.items():
            travel_time[pid0][pid1] = get_dist_twoPoints(coord0, coord1)
    return (numLocs, lid_coord, coord_lid), travel_time


def euclideanDistEx0(pkl_dir='_temp'):
    seed(1)
    min_dist_path, max_dist_path, min_dist_task = 0.4, 1.0, 0.08
    clusters = [(0.8, 0.2, 0.4),
                # (0.3, 0.8, 0.4),
                (0.4, 0.7, 0.3),
                ]
    #
    problemName = 'euclideanDistEx0'
    numPaths, numTasks = 5, 5
    thDetour = 0.85
    numBundles, minTB, maxTB = 2, 2, 3
    #
    flow_oridest, task_ppdp = gen_pt_locs(clusters, min_dist_path, max_dist_path, min_dist_task, numPaths, numTasks)
    (numLocs, lid_coord, coord_lid), travel_time = handle_locationNtt(flow_oridest, task_ppdp)
    #
    flows = {}
    for loc0, loc1 in flow_oridest:
        flows[coord_lid[loc0], coord_lid[loc1]] = 1
    tasks = []
    for i, (loc0, loc1) in enumerate(task_ppdp):
        tasks.append((i, coord_lid[loc0], coord_lid[loc1], 1, 1))
    #
    problem = [problemName,
               flows, tasks,
               numBundles, minTB, maxTB,
               numLocs, travel_time, thDetour]
    with open(opath.join(pkl_dir, 'problem_%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(problem, fp)
    prmt = convert_prob2prmt(*problem)
    with open(opath.join(pkl_dir, 'prmts_%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(prmt, fp)
    vizInputs = [flow_oridest, task_ppdp]
    with open(opath.join(pkl_dir, 'dplym_%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(vizInputs, fp)
    #
    return prmt


def randomProb_EuclDist(pkl_dir, numTasks=10, numBundles=3, thVolume=4, numPaths=5, thDetour=3):
    min_dist_path, max_dist_path, min_dist_task = 0.4, 1.0, 0.08
    clusters = [(0.8, 0.2, 0.3),
                (0.5, 0.5, 0.3),
                (0.4, 0.7, 0.3),
                ]
    #
    # problemName = 'nt%04d-nb%04d-np%03d-%d' % (numTasks, numBundles, numPaths, int(time.time()))
    problemName = 'nt%04d-nb%04d-np%03d-dt%.2f-vt%d' % (numTasks, numBundles, numPaths, float(thDetour), thVolume)
    path_oridest, task_ppdp = gen_pt_locs(clusters, min_dist_path, max_dist_path, min_dist_task, numPaths, numTasks)
    (numLocs, lid_coord, coord_lid), travel_time = handle_locationNtt(path_oridest, task_ppdp)


    flows = {}
    for loc0, loc1 in path_oridest:
        flows[coord_lid[loc0], coord_lid[loc1]] = 1
    tasks = []
    for i, (loc0, loc1) in enumerate(task_ppdp):
        tasks.append((i, coord_lid[loc0], coord_lid[loc1], 1, 1))
    #
    problemPrmts = [problemName,
                    numLocs, travel_time, flows,
                    len(tasks), tasks,
                    numBundles, thVolume, thDetour]
    vizInputs = [clusters, path_oridest, task_ppdp]
    #
    with open(opath.join(pkl_dir, 'prmts_%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(problemPrmts, fp)
    with open(opath.join(pkl_dir, 'dplym_%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(vizInputs, fp)


if __name__ == '__main__':
    # print(convert_p2i(*ex1('_temp')))
    euclideanDistEx0(pkl_dir='_temp')
