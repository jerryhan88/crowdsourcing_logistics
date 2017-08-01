from init_project import *
#
from random import randrange
from math import ceil
import pickle
import csv

problem_summary_fpath = opath.join(dpath['problem'], 'problem_summary.csv')

if not opath.exists(problem_summary_fpath):
    with open(problem_summary_fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        new_headers = ['fn', 'numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour']
        writer.writerow(new_headers)


def save_problem(inputs):
    points, travel_time, \
    flows, paths, \
    tasks, rewards, volumes, \
    numBundles, thVolume, thDetour = inputs
    numTasks, numPaths = map(len, [tasks, paths])
    fn = 'nt%d-np%d-nb%d-tv%d-td%d.pkl' % (numTasks, numPaths, numBundles, thVolume, thDetour)
    with open(problem_summary_fpath, 'a') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        writer.writerow([fn, numTasks, numPaths, numBundles, thVolume, thDetour])
    #
    ofpath = opath.join(dpath['problem'], fn)
    if not opath.exists(ofpath):
        with open(ofpath, 'wb') as fp:
            pickle.dump(inputs, fp)
    return fn


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
        assert points.has_key(pp)
        assert points.has_key(dp)
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
    numBundles = 1
    thVolume = 10
    thDetour = 1000
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    points = points.keys()
    #
    inputs = [points, travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    save_problem(inputs)
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
    numBundles = 5
    thVolume = 4
    thDetour = 8
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    points = points.keys()
    #
    inputs = [points, travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    save_problem(inputs)
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
    numBundles = 2
    # thVolume = 2
    thVolume = 3
    thDetour = 3
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    points = points.keys()
    #
    inputs = [points, travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    save_problem(inputs)
    return inputs



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
    numBundles = 3
    thVolume = 4
    thDetour = 6
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    points = points.keys()
    #
    inputs = [points, travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    save_problem(inputs)
    return inputs



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
    numBundles = 4
    thVolume = 4
    thDetour = 6
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    points = points.keys()
    #
    inputs = [points, travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    save_problem(inputs)
    return inputs


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
    #
    inputs = [points, travel_time,
              flows, paths,
              tasks, rewards, volumes,
              numBundles, thVolume, thDetour]
    fn = save_problem(inputs)
    return inputs, fn


if __name__ == '__main__':
    maxFlow = 3
    minReward, maxReward = 1, 3
    minVolume, maxVolume = 1, 2
    thVolume = 3
    detourAlowProp = 0.8
    jobID = 0
    # for i in range(2, 5):
    #     numCols = numRows = i
    #     for numTasks in range(4, 18, 2):
    #         for j in range(5, 10):
    #             bundleResidualProp = 1 + j / 10.0
    #             inputs, fn = random_problem(numCols, numRows, maxFlow,
    #                                         numTasks, minReward, maxReward, minVolume, maxVolume,
    #                                         thVolume, bundleResidualProp, detourAlowProp)

    numCols = numRows = 4
    numTasks = 4
    for j in range(5, 10):
        bundleResidualProp = 1 + j / 10.0
        inputs, fn = random_problem(numCols, numRows, maxFlow,
                                    numTasks, minReward, maxReward, minVolume, maxVolume,
                                    thVolume, bundleResidualProp, detourAlowProp)





    #
    # random_problem(2, 3, 3,
    #                3, 1, 3, 1, 2,
    #                4, 1.3, 0.5)