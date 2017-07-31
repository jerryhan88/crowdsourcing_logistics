from init_project import *
#
from random import randrange
from math import ceil

def input_validity(points, flows, paths, tasks, num_bundles, thVolume):
    # assert len(flows) == len(points)
    #
    for i, f in enumerate(flows):
        assert f[i] == 0
        # assert len(f) == len(points)
    #
    assert len(paths) == len(flows) * (len(flows) - 1)
    #
    for pp, dp in tasks:
        assert points.has_key(pp)
        assert points.has_key(dp)
    #
    assert len(tasks) <= num_bundles * thVolume


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
    paths = [(i, j) for i in xrange(len(flows)) for j in xrange(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks = [(0, 1)]
    rewards = [1]
    volumes = [1]
    #
    # Inputs about bundles
    #
    num_bundles = 1
    volume_th = 10
    detour_th = 1000
    #
    input_validity(points, flows, paths, tasks, num_bundles, volume_th)
    points = points.keys()
    return points, travel_time, \
           flows, paths, \
           tasks, rewards, volumes, \
           num_bundles, volume_th, detour_th


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
    for p0 in points.itervalues():
        for p1 in points.itervalues():
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
    paths = [(i, j) for i in xrange(len(flows)) for j in xrange(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks = [(0, 2), (2, 1), (3, 5), (2, 7), (5, 1),
             (6, 8), (2, 6), (8, 4), (7, 3), (4, 1)]
    rewards = [1, 2, 3, 2, 1,
                3, 1, 2, 1, 2]
    volumes = [1, 1, 1, 1, 1,
                1, 1, 1, 1, 1]
    #
    # Inputs about bundles
    #
    num_bundles = 5
    volume_th = 4
    detour_th = 8
    #
    input_validity(points, flows, paths, tasks, num_bundles, volume_th)
    points = points.keys()
    return points, travel_time, \
            flows, paths, \
            tasks, rewards, volumes, \
            num_bundles, volume_th, detour_th


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
    for p0 in points.itervalues():
        for p1 in points.itervalues():
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
    paths = [(i, j) for i in xrange(len(flows)) for j in xrange(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks = [(0, 2), (2, 3), (1, 2)]
    rewards = [1, 2, 1]
    volumes = [1, 1, 1]
    #
    # Inputs about bundles
    #
    num_bundles = 2
    # volume_th = 2
    volume_th = 3
    detour_th = 3
    #
    input_validity(points, flows, paths, tasks, num_bundles, volume_th)
    points = points.keys()
    return points, travel_time, \
           flows, paths, \
           tasks, rewards, volumes, \
           num_bundles, volume_th, detour_th



def ex3():
    #
    # Define a network
    #
    points, travel_time = {}, {}
    pid = 0
    for i in range(2):
        for j in range(2):
            points[pid] = point(pid, i, j)
            pid += 1
    for p0 in points.itervalues():
        for p1 in points.itervalues():
            travel_time[p0.pid, p1.pid] = abs(p0.i - p1.i) + abs(p0.j - p1.j)
    #
    # Define flows and paths
    #

    flows = [
        [0, 1, 2, 1],
        [1, 0, 2, 1],
        [2, 1, 0, 2],
        [1, 3, 3, 0],
    ]
    paths = [(i, j) for i in xrange(len(flows)) for j in xrange(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks = [(0, 2), (2, 3), (1, 2),
             ]
    rewards = [1, 2, 1]
    volumes = [1, 1, 1]
    #
    # Inputs about bundles
    #
    num_bundles = 3
    volume_th = 4
    detour_th = 6
    #
    input_validity(points, flows, paths, tasks, num_bundles, volume_th)
    points = points.keys()
    return points, travel_time, \
           flows, paths, \
           tasks, rewards, volumes, \
           num_bundles, volume_th, detour_th



def ex4():
    #
    # Define a network
    #
    points, travel_time = {}, {}
    pid = 0
    for i in range(2):
        for j in range(2):
            points[pid] = point(pid, i, j)
            pid += 1
    for p0 in points.itervalues():
        for p1 in points.itervalues():
            travel_time[p0.pid, p1.pid] = abs(p0.i - p1.i) + abs(p0.j - p1.j)
    #
    # Define flows and paths
    #

    flows = [
        [0, 1, 2],
        [1, 0, 2],
        [0, 1, 0],
    ]
    paths = [(i, j) for i in xrange(len(flows)) for j in xrange(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks = [(0, 2), (2, 0), (1, 2),
             ]
    rewards = [1, 2, 1]
    volumes = [1, 1, 1]
    #
    # Inputs about bundles
    #
    num_bundles = 4
    volume_th = 4
    detour_th = 6
    #
    input_validity(points, flows, paths, tasks, num_bundles, volume_th)
    points = points.keys()
    return points, travel_time, \
           flows, paths, \
           tasks, rewards, volumes, \
           num_bundles, volume_th, detour_th



class point(object):
    def __init__(self, pid, i, j):
        self.pid, self.i, self.j = pid, i, j

    def __repr__(self):
        return 'pid%d' % self.pid



def random_problem(numCols, numRows, maxFlow,
                   numTasks, minReward, maxReward, minVolume, maxVolume,
                   thVolume, bundleResidualProp, detourAlowProp):
    #
    # Define a network
    #
    points, travel_time = {}, {}
    pid = 0
    for i in range(numCols):
        for j in range(numRows):
            points[pid] = point(pid, i, j)
            pid += 1
    for p0 in points.itervalues():
        for p1 in points.itervalues():
            travel_time[p0.pid, p1.pid] = abs(p0.i - p1.i) + abs(p0.j - p1.j)
    #
    # Define flows and paths
    #
    flows = [[0 for _ in xrange(len(points))] for _ in xrange(len(points))]
    for i in xrange(len(points)):
        for j in xrange(len(points)):
            if i == j:
                continue
            flows[i][j] = randrange(maxFlow)
    paths = [(i, j) for i in xrange(len(flows)) for j in xrange(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks, rewards, volumes = [], [], []
    for _ in xrange(numTasks):
        i, j = randrange(len(points)), randrange(len(points))
        while i == j:
            j = randrange(len(points))
        tasks.append((i, j))
        rewards.append(minReward + randrange(maxReward))
        volumes.append(minVolume + randrange(maxVolume))
    #
    # Inputs about bundles
    #
    assert bundleResidualProp > 1.0
    numBundles = int(ceil(len(tasks) / float(thVolume)) * bundleResidualProp)
    thDetour = int((numCols + numRows) * detourAlowProp)
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    points = points.keys()
    return points, travel_time, \
           flows, paths, \
           tasks, rewards, volumes, \
           numBundles, thVolume, thDetour


if __name__ == '__main__':
    random_problem(2, 3, 3,
                   3, 1, 3, 1, 2,
                   4, 1.3, 0.5)