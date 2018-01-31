from init_project import *
import os.path as opath
import os
from random import randrange
from dataProcessing import *

import csv

class point(object):
    def __init__(self, pid, i, j):
        self.pid, self.i, self.j = pid, i, j

    def __repr__(self):
        return 'pid%d(%d,%d)' % (self.pid, self.i, self.j)


def convert_p2i(travel_time, \
                flows, paths, \
                tasks, rewards, volumes, \
                num_bundles, volume_th, detour_th):
    #
    # Bundle
    #
    bB = num_bundles
    _lambda = volume_th
    #
    # Task
    #
    T = list(range(len(tasks)))
    iPs, iMs = list(zip(*[tasks[i] for i in T]))
    r_i, v_i = rewards, volumes
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
    K = list(range(len(paths)))
    _kP, _kM = list(zip(*[paths[k] for k in K]))
    if type(flows) == list:
        sum_f_k = sum(flows[i][j] for i in range(len(flows)) for j in range(len(flows)))
        w_k = [flows[i][j] / float(sum_f_k) for i, j in paths]
    else:
        assert type(flows) == dict
        sum_f_k = sum(flows.values())
        w_k = [flows[i, j] / float(sum_f_k) for i, j in paths]
    _delta = detour_th
    t_ij = {}
    for k in K:
        kP, kM = 'ori%d' % k, 'dest%d' % k
        t_ij[kP, kP] = travel_time[_kP[k], _kP[k]]
        t_ij[kM, kM] = travel_time[_kM[k], _kM[k]]
        t_ij[kP, kM] = travel_time[_kP[k], _kM[k]]
        t_ij[kM, kP] = travel_time[_kM[k], _kP[k]]
        for i in _N:
            t_ij[kP, i] = travel_time[_kP[k], _N[i]]
            t_ij[i, kP] = travel_time[_N[i], _kP[k]]
            #
            t_ij[kM, i] = travel_time[_kM[k], _N[i]]
            t_ij[i, kM] = travel_time[_N[i], _kM[k]]
    for i in _N:
        for j in _N:
            t_ij[i, j] = travel_time[_N[i], _N[j]]
    N = set(_N.keys())
    #
    return {'bB': bB,
            'T': T, 'r_i': r_i, 'v_i': v_i, '_lambda': _lambda,
            'P': P, 'D': D, 'N': N,
            'K': K, 'w_k': w_k,
            't_ij': t_ij, '_delta': _delta,
            '_N': _N}


def input_validity(points, flows, paths, tasks, numBundles, thVolume):
    # assert len(flows) == len(points)
    #
    if type(flows) == list:
        for i, f in enumerate(flows):
            assert f[i] == 0
        #
        assert len(paths) == len(flows) * (len(flows) - 1)
    #
    for pp, dp in tasks:
        assert pp in points
        assert dp in points
    #
    assert len(tasks) <= numBundles * thVolume


def ex1():
    #
    # Define a network
    #
    points, travel_time = {}, {}
    pid = 0
    for i in range(3):
        for j in range(3):
            points[pid] = point(pid, i, j)
            pid += 1
    for p0 in points.values():
        for p1 in points.values():
            travel_time[p0.pid, p1.pid] = abs(p0.i - p1.i) + abs(p0.j - p1.j)
    #
    # Define flows and paths
    #
    flows = [
            [0, 1, 2, 1, 1, 2, 3, 2, 0],
            [0, 0, 2, 1, 2, 1, 1, 2, 1],
            [2, 1, 0, 2, 1, 1, 2, 1, 0],
            [1, 0, 3, 0, 1, 1, 2, 0, 2],
            [3, 2, 1, 2, 0, 1, 0, 2, 3],
            [2, 3, 1, 0, 2, 0, 3, 1, 1],
            [1, 1, 3, 2, 0, 2, 0, 3, 0],
            [0, 2, 2, 3, 0, 2, 2, 0, 2],
            [2, 2, 2, 3, 2, 0, 3, 0, 0]
            ]
    paths = [(i, j) for i in range(len(flows)) for j in range(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks = [(0, 2), (2, 1), (3, 5), (2, 7), (5, 1),
             (6, 8)]
    rewards = [1, 2, 3, 2, 1,
                3]
    volumes = [1, 1, 1, 1, 1,
                1]
    #
    # Inputs about bundles
    #
    numBundles = 4
    thVolume = 4
    thDetour = 6
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    #
    inputs = [travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    #
    return inputs


def ex2():
    #
    # Define a network
    #
    points, travel_time = {}, {}
    pid = 0
    for i in range(2):
        for j in range(2):
            points[pid] = point(pid, i, j)
            pid += 1
    for p0 in points.values():
        for p1 in points.values():
            travel_time[p0.pid, p1.pid] = abs(p0.i - p1.i) + abs(p0.j - p1.j)
    #
    # Define flows and paths
    #

    flows = [
        [0, 1, 2, 1],
        [0, 0, 2, 1],
        [2, 1, 0, 2],
        [1, 0, 3, 0],
    ]
    paths = [(i, j) for i in range(len(flows)) for j in range(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks = [(0, 2), (2, 3), (1, 2)]
    rewards = [1, 1, 1]
    volumes = [1, 1, 1]
    #
    # Inputs about bundles
    #
    numBundles = 2
    # thVolume = 2
    thVolume = 3
    thDetour = 3
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    #
    inputs = [travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    #
    return inputs


def paperExample():
    #
    # Define a network
    #
    points, travel_time = {}, {}
    pid = 0
    ij_pid = {}


    for i in range(5):
        for j in range(5):
            points[pid] = point(pid, i, j)
            ij_pid[i, j] = pid
            pid += 1
    for p0 in points.values():
        for p1 in points.values():
            travel_time[p0.pid, p1.pid] = abs(p0.i - p1.i) + abs(p0.j - p1.j)
    #
    # Define flows and paths
    #
    flows = {(ij_pid[2, 0], ij_pid[3, 4]): 3,
             (ij_pid[0, 0], ij_pid[4, 3]): 1,
             (ij_pid[0, 3], ij_pid[4, 1]): 2,
             }
    paths = list(flows.keys())
    #
    # Inputs about tasks
    #
    tasks = [
             # (pickup point, delivery point)
             (ij_pid[2, 2], ij_pid[2, 2]),  # (1^+, 1^-)
             (ij_pid[1, 4], ij_pid[2, 2]),  # (2^+, 2^-)
             (ij_pid[2, 0], ij_pid[3, 4]),  # (3^+, 3^-)
             (ij_pid[0, 1], ij_pid[4, 1]),  # (4^+, 4^-)
             (ij_pid[2, 3], ij_pid[3, 1]),  # (5^+, 5^-)
             ]
    rewards = [1, 1, 1, 1, 1]
    volumes = [1, 1, 1, 1, 1]
    #
    # Inputs about bundles
    #
    numBundles = 2
    thVolume = 3
    thDetour = 2
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    #
    inputs = [travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    #
    return inputs


def mrtNetExample1():
    numTasks = 10
    thVolume, thDetour = 3, 20
    prob_dpath = opath.join(dpath['experiment'], 'mrtNet-T%d-k%d-vt%d-dt%d' %
                            (numTasks, len(repStations) * (len(repStations) - 1), thVolume, thDetour))
    #
    _flows, \
    _tasks, rewards, volumes, \
    _points, _travel_time, \
    numBundles, thVolume, thDetour = get_mrtNetExample(prob_dpath, numTasks, thVolume, thDetour)

    #
    flows, tasks, points, travel_time = convert_mrtNet2ID(_flows, _tasks, _points, _travel_time)
    #
    paths = list(flows.keys())
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    #
    inputs = [travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    #
    return inputs



def convert_mrtNet2ID(_flows, _tasks, _points, _travel_time):
    STN2ID = {}
    points, travel_time = {}, {}
    for i, stn in enumerate(_points):
        STN2ID[stn] = i
        points[i] = None
    for (stn0, stn1), t in _travel_time.items():
        travel_time[STN2ID[stn0], STN2ID[stn1]] = t
    #
    flows = {}
    for (stn0, stn1), count in _flows.items():
        flows[STN2ID[stn0], STN2ID[stn1]] = count
    #
    tasks = []
    for stn0, stn1 in _tasks:
        tasks.append((STN2ID[stn0], STN2ID[stn1]))
    #
    return [flows, tasks, points, travel_time]


def get_mrtNetExample(prob_dpath, numTasks=10, thVolume=3, thDetour=20):
    pPath_fpath = opath.join(prob_dpath, 'mrtNet-path.csv')
    pTask_fpath = opath.join(prob_dpath, 'mrtNet-task.csv')
    pTravlTime_fpath = opath.join(prob_dpath, 'mrtNet-travelTime.csv')
    pEtc_fpath = opath.join(prob_dpath, 'mrtNet-etc.csv')
    if not opath.exists(prob_dpath):
        os.mkdir(prob_dpath)
        #
        with open(pPath_fpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_header = ['origin', 'destination', 'count', 'weight', 'pathSeq']
            writer.writerow(new_header)
        #
        with open(pTask_fpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_header = ['pickup', 'delivery', 'reward', 'volume']
            writer.writerow(new_header)
        #
        with open(pTravlTime_fpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_header = ['fSTN', 'tSTN', 'duration', 'pathSeq']
            writer.writerow(new_header)
        #
        with open(pEtc_fpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_header = ['attribute', 'value']
            writer.writerow(new_header)
        #
        _points, _travel_time = set(), {}
        with open(opath.join(dpath['flow'], 'flow-M%d%02d.csv' % (year, month))) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                fSTN, tSTN, _medianD = [row[cn] for cn in ['fSTN', 'tSTN', 'medianD']]
                if fSTN in xStations or tSTN in xStations:
                    continue
                _travel_time[fSTN, tSTN] = eval(_medianD)
        G = get_mrtNetwork()
        for stn0 in G.nodes:
            for stn1 in G.nodes:
                if stn0 == stn1:
                    duration = 0
                    _travel_time[stn0, stn1] = duration
                    with open(pTravlTime_fpath, 'a') as w_csvfile:
                        writer = csv.writer(w_csvfile, lineterminator='\n')
                        writer.writerow([stn0, stn1, duration, str([])])
                    continue
                _points.add(stn0)
                _points.add(stn1)
                pathSeq = nx.shortest_path(G, stn0, stn1)
                if (stn0, stn1) in _travel_time:
                    duration = _travel_time[stn0, stn1]
                else:
                    duration = sum([G.edges[pathSeq[i], pathSeq[i + 1]]['weight'] for i in range(len(pathSeq) - 1)])

                    _travel_time[stn0, stn1] = duration
                with open(pTravlTime_fpath, 'a') as w_csvfile:
                    writer = csv.writer(w_csvfile, lineterminator='\n')
                    writer.writerow([stn0, stn1, duration, str(pathSeq)])
        #
        candi_PD = set()
        csv_fpath = opath.join(dpath['flow'], 'rsFlow-%d%02d-rsN%d.csv' %
                               (year, month, len(repStations) * (len(repStations) - 1)))
        _flows, path_seq, total_count = {}, {}, 0
        with open(csv_fpath) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                fSTN, tSTN, _count, _path = [row[cn] for cn in ['fSTN', 'tSTN', 'count', 'path']]
                count = int(_count)
                pathSeq = []
                for STN in eval(_path):
                    pathSeq.append(STN)
                    candi_PD.add(STN)
                _flows[fSTN, tSTN] = count
                path_seq[fSTN, tSTN] = pathSeq
                total_count += count
        for (fSTN, tSTN), count in _flows.items():
            weight = count / float(total_count)
            with open(pPath_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                new_row = [fSTN, tSTN, count, weight, str(path_seq[fSTN, tSTN])]
                writer.writerow(new_row)
        #
        candi_PD = list(candi_PD)
        _tasks, rewards, volumes = [], [], []
        for _ in range(numTasks):
            r, v = 1, 1
            i, j = randrange(len(candi_PD)), randrange(len(candi_PD))
            while i == j:
                j = randrange(len(candi_PD))
            _tasks.append((candi_PD[i], candi_PD[j]))
            rewards.append(r)
            volumes.append(v)
            with open(pTask_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                new_row = [candi_PD[i], candi_PD[j], r, v]
                writer.writerow(new_row)
        #
        numBundles = int((numTasks / thVolume) * 1.5)
        for att, val in [('numBundles', numBundles),
                         ('thVolume', thVolume),
                         ('thDetour', thDetour)]:
            with open(pEtc_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([att, val])
    else:
        _flows = {}
        with open(pPath_fpath) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                origin, destination = [row[cn] for cn in ['origin', 'destination']]
                count = int(row['count'])
                _flows[origin, destination] = count
        #
        _tasks, rewards, volumes = [], [], []
        with open(pTask_fpath) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                pickup, delivery = [row[cn] for cn in ['pickup', 'delivery']]
                r, v = [int(row[cn]) for cn in ['reward', 'volume']]
                _tasks.append((pickup, delivery))
                rewards.append(r)
                volumes.append(v)
        #
        _points, _travel_time = {}, {}
        with open(pTravlTime_fpath) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                p0, p1 = [row[cn] for cn in ['fSTN', 'tSTN']]
                _points[p0] = None
                _points[p1] = None
                _travel_time[p0, p1] = float(row['duration'])
        #
        att_val = {}
        with open(pEtc_fpath) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                att_val[row['attribute']] = eval(row['value'])
        numBundles, thVolume, thDetour = [att_val[att] for att in ['numBundles', 'thVolume', 'thDetour']]
    #
    return [_flows,
            _tasks, rewards, volumes,
            _points, _travel_time,
            numBundles, thVolume, thDetour]


if __name__ == '__main__':
    # print(convert_p2i(*paperExample()))
    mrtNetExample1()
