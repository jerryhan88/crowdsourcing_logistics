from init_project import *
#
from random import randrange


class point(object):
    def __init__(self, pid, i, j):
        self.pid, self.i, self.j = pid, i, j

    def __repr__(self):
        return 'pid%d' % self.pid


class task(object):
    def __init__(self, tid, pp, dp, v, r):
        self.tid = tid
        self.pp, self.dp = pp, dp
        self.v, self.r = v, r

    def set_attr(self, attr):
        self.attr = attr

    def __repr__(self):
        return 't%d(%s->%s;%.03f)' % (self.tid, self.pp, self.dp, self.r)


class path(object):
    def __init__(self, ori, dest, w=None):
        self.ori, self.dest, self.w = ori, dest, w

    def __repr__(self):
        if self.w != None:
            return '%d->%d;%.03f' % (self.ori, self.dest, self.w)
        else:
            return '%d->%d' % (self.ori, self.dest)


class bundle(object):
    def __init__(self, bid, paths):
        self.bid = bid
        #
        self.tasks = {}
        self.path_pd_seq, self.path_detour = {}, {}
        for p in paths:
            self.path_pd_seq[p] = []
            self.path_detour[p] = 0
        self.bundle_attr = 0

    def __repr__(self):
        return 'b%d(ts:%s)' % (self.bid, ','.join(['t%d' % t.tid for t in self.tasks.values()]))


def convert_input4greedyHeuristic(travel_time,
                                  flows, paths,
                                  tasks, rewards, volumes,
                                  num_bundles, volume_th, detour_th):
    #
    # Convert inputs for the greedy heuristic
    #
    tasks = [task(i, pp, dd, volumes[i], rewards[i]) for i, (pp, dd) in enumerate(tasks)]
    total_flows = sum(flows[i][j] for i in range(len(flows)) for j in range(len(flows)))
    paths = [path(ori, dest, flows[ori][dest] / float(total_flows)) for ori, dest in paths]
    #
    return travel_time, tasks, paths, detour_th, volume_th, num_bundles


def convert_input4MathematicalModel(travel_time, \
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
    iP, iM = list(zip(*[tasks[i] for i in T]))
    r_i, v_i = rewards, volumes
    P, D = set(), set()
    _N = {}
    for i in T:
        P.add('p0%d' % i)
        D.add('d%d' % i)
        #
        _N['p0%d' % i] = iP[i]
        _N['d%d' % i] = iM[i]
    #
    # Path
    #
    K = list(range(len(paths)))
    kP, kM = list(zip(*[paths[k] for k in K]))
    sum_f_k = sum(flows[i][j] for i in range(len(flows)) for j in range(len(flows)))
    w_k = [flows[i][j] / float(sum_f_k) for i, j in paths]
    _delta = detour_th
    t_ij = {}
    for k in K:
        _kP, _kM = 'ori%d' % k, 'dest%d' % k
        t_ij[_kP, _kP] = travel_time[kP[k], kP[k]]
        t_ij[_kM, _kM] = travel_time[kM[k], kM[k]]
        t_ij[_kP, _kM] = travel_time[kP[k], kM[k]]
        t_ij[_kM, _kP] = travel_time[kM[k], kP[k]]
        for i in _N:
            t_ij[_kP, i] = travel_time[kP[k], _N[i]]
            t_ij[i, _kP] = travel_time[_N[i], kP[k]]
            #
            t_ij[_kM, i] = travel_time[kM[k], _N[i]]
            t_ij[i, _kM] = travel_time[_N[i], kM[k]]
    for i in _N:
        for j in _N:
            t_ij[i, j] = travel_time[_N[i], _N[j]]
    N = set(_N.keys())
    #
    return bB, \
           T, r_i, v_i, _lambda, P, D, N, \
           K, w_k, t_ij, _delta


def input_validity(points, flows, paths, tasks, numBundles, thVolume):
    # assert len(flows) == len(points)
    #
    for i, f in enumerate(flows):
        assert f[i] == 0
        # assert len(f) == len(points)
    #
    assert len(paths) == len(flows) * (len(flows) - 1)
    #
    for pp, dp in tasks:
        assert pp in points
        assert dp in points
    #
    assert len(tasks) <= numBundles * thVolume


def random_problem(numCols, numRows, maxFlow,
                   numTasks, minReward, maxReward, minVolume, maxVolume,
                   numBundles, volumeAlowProp, detourAlowProp):
    #
    # Define a network
    #
    points, travel_time = {}, {}
    pid = 0
    for i in range(numCols):
        for j in range(numRows):
            points[pid] = point(pid, i, j)
            pid += 1
    for p0 in points.values():
        for p1 in points.values():
            travel_time[p0.pid, p1.pid] = abs(p0.i - p1.i) + abs(p0.j - p1.j)
    #
    # Define flows and paths
    #
    flows = [[0 for _ in range(len(points))] for _ in range(len(points))]
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            flows[i][j] = randrange(maxFlow)
    paths = [(i, j) for i in range(len(flows)) for j in range(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks, rewards, volumes = [], [], []
    for _ in range(numTasks):
        i, j = randrange(len(points)), randrange(len(points))
        while i == j:
            j = randrange(len(points))
        tasks.append((i, j))
        rewards.append(minReward + randrange(maxReward))
        # volumes.append(minVolume + randrange(maxVolume))
        volumes.append(1)
    #
    # Inputs about bundles
    #
    thVolume = int(sum(volumes) / numBundles * volumeAlowProp)
    thDetour = int((numCols + numRows) * detourAlowProp)
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    #
    inputs = [travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    #
    return inputs


def ex0():
    #
    # Define a network
    #
    points = {
                0: point(0, 0, 0),
                1: point(1, 0, 0)
             }
    travel_time = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0,
    }
    #
    # Define flows and paths
    #
    flows = [
        [0, 1],
        [1, 0],
    ]
    paths = [(i, j) for i in range(len(flows)) for j in range(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks = [(0, 1)]
    rewards = [1]
    volumes = [1]
    #
    # Inputs about bundles
    #
    numBundles = 1
    thVolume = 10
    thDetour = 1000
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    #
    inputs = [travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    #
    return inputs


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

if __name__ == '__main__':
    maxFlow = 3
    minReward, maxReward = 1, 3
    minVolume, maxVolume = 1, 3
    volumeAlowProp, detourAlowProp = 1.5, 1.2
    numCols, numRows = 1, 4
    #
    numBundles = 4
    # for numBundles in [20, 30]:
    for numTasks in [10, 12]:
        inputs = random_problem(numCols, numRows, maxFlow,
                                numTasks, minReward, maxReward, minVolume, maxVolume,
                                numBundles, volumeAlowProp, detourAlowProp)



