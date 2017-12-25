from random import randrange


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



if __name__ == '__main__':
    print(convert_p2i(*paperExample()))


    # maxFlow = 3
    # minReward, maxReward = 1, 3
    # minVolume, maxVolume = 1, 3
    # volumeAlowProp, detourAlowProp = 1.5, 1.2
    # numCols, numRows = 1, 4
    # #
    # numBundles = 4
    # # for numBundles in [20, 30]:
    # for numTasks in [10, 12]:
    #     inputs = random_problem(numCols, numRows, maxFlow,
    #                             numTasks, minReward, maxReward, minVolume, maxVolume,
    #                             numBundles, volumeAlowProp, detourAlowProp)



