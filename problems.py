from init_project import *
#
from random import randrange
from math import ceil


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

def ex7():
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
            [0, 1, 2, 1, 1],
            [0, 0, 2, 1, 2],
            [2, 1, 0, 2, 1],
            [1, 0, 3, 0, 1],
            [3, 2, 1, 2, 0],
            ]
    paths = [(i, j) for i in range(len(flows)) for j in range(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks = [(0, 2), (2, 1), (3, 4), (2, 4), (2, 4), (3, 1)]
    rewards = [1, 1, 1, 1, 1, 1]
    volumes = [1, 1, 1, 1, 1, 1]
    #
    # Inputs about bundles
    #
    numBundles = 3
    thVolume = 4
    thDetour = 7
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    #
    inputs = [travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    #
    return inputs



def ex8():
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
            [0, 1, 2, 1, 1],
            [0, 0, 2, 1, 2],
            [2, 1, 0, 2, 1],
            [1, 0, 3, 0, 1],
            [3, 2, 1, 2, 0],
            ]
    paths = [(i, j) for i in range(len(flows)) for j in range(len(flows)) if i != j]
    #
    # Inputs about tasks
    #
    tasks = [(0, 2), (2, 1), (3, 4),
             (2, 4), (2, 4), (3, 1),
             (3, 0), (0, 3)]
    rewards = [1, 2, 3,
               2, 1, 1,
               1, 2]
    volumes = [1, 1, 1,
               1, 1, 1,
               1, 1]
    #
    # Inputs about bundles
    #
    numBundles = 4
    thVolume = 3
    thDetour = 4
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    #
    inputs = [travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    #
    return inputs


class point(object):
    def __init__(self, pid, i, j):
        self.pid, self.i, self.j = pid, i, j

    def __repr__(self):
        return 'pid%d' % self.pid


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



if __name__ == '__main__':
    maxFlow = 3
    minReward, maxReward = 1, 3
    minVolume, maxVolume = 1, 3
    thVolume = 5
    detourAlowProp = 0.8
    jobID = 0
    for i in range(2, 5):
        numCols = numRows = i
        for numTasks in range(4, 18, 2):
            for j in range(5, 10):
                bundleResidualProp = 1 + j / 10.0
                inputs = random_problem(numCols, numRows, maxFlow,
                                            numTasks, minReward, maxReward, minVolume, maxVolume,
                                            thVolume, bundleResidualProp, detourAlowProp)
    # numCols = numRows = 3
    # bundleResidualProp = 1 + 0.1
    # numTasks = 3
    # inputs, fn = random_problem(numCols, numRows, maxFlow,
    #                             numTasks, minReward, maxReward, minVolume, maxVolume,
    #                             thVolume, bundleResidualProp, detourAlowProp)
    # print(inputs)



