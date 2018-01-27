from init_project import *
#
import pickle
from random import choice, randrange
import time
#
from problems import point, input_validity
from GH import run as GH_run
from EX import run as EX_run
from BNP import run as BNP_run


def randomProb_5by5(numTasks=10, numBundles=3, thVolume=4, numPaths=5, minTravlTime4OD=5, maxFlow=3, thDetour=3):
    points, travel_time = {}, {}
    pid = 0
    for i in range(5):
        for j in range(5):
            points[pid] = point(pid, i, j)
            pid += 1
    for p0 in points.values():
        for p1 in points.values():
            travel_time[p0.pid, p1.pid] = abs(p0.i - p1.i) + abs(p0.j - p1.j)

    flows = {}
    od_paris = [(p0, p1) for p0, p1 in travel_time.keys() if travel_time[p0, p1] > minTravlTime4OD]
    assert numPaths <= len(od_paris)
    while len(flows) != numPaths:
        ori, dest = choice(od_paris)
        while (ori, dest) in flows:
            ori, dest = choice(od_paris)
        flows[ori, dest] = randrange(maxFlow)
    paths = list(flows.keys())
    #
    tasks, rewards, volumes = [], [], []
    pd_paris = [(p0, p1) for p0, p1 in travel_time.keys() if travel_time[p0, p1] != 0]
    for _ in range(numTasks):
        tasks.append(choice(pd_paris))
        rewards.append(1)
        volumes.append(1)
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    #
    problem = [travel_time,
               flows, paths,
               tasks, rewards, volumes,
               numBundles, thVolume, thDetour]
    #
    return problem



def gen_problems(problem_dpath):
    #
    # Generate problems
    #
    if not opath.exists(problem_dpath):
        os.mkdir(problem_dpath)
    #
    minTravlTime4OD, maxFlow = 5, 4
    #
    thVolume = 3
    numPaths = 10
    thDetour = 3

    numTasks, numBundles = 6, 3


    for numPaths in [10, 20]:
        for numTasks, numBundles in [(10, 4), (20, 8), (30, 12)]:
            problem = randomProb_5by5(numTasks, numBundles, thVolume, numPaths, minTravlTime4OD, maxFlow, thDetour)

            fn = 'nt%02d-np%d-nb%d-tv%d-td%d-%d.pkl' % (numTasks, numPaths, numBundles, thVolume, thDetour, int(time.time()))
            ofpath = opath.join(problem_dpath, fn)
            with open(ofpath, 'wb') as fp:
                pickle.dump(problem, fp)


def run_experiments(machine_num):
    _numThreads, _TimeLimit = 8, 4 * 60 * 60
    machine_dpath = opath.join(dpath['experiment'], 'm%d' % machine_num)
    problem_dpath = opath.join(machine_dpath, '__problems')
    for path in [machine_dpath, problem_dpath]:
        assert opath.exists(path), path
    log_dpath = opath.join(machine_dpath, 'log')
    res_dpath = opath.join(machine_dpath, 'res')
    bbt_dpath = opath.join(machine_dpath, 'bpt')
    err_dpath = opath.join(machine_dpath, 'err')
    for path in [log_dpath, res_dpath, err_dpath, bbt_dpath]:
        os.makedirs(path)
    problems_ifpathes = [opath.join(problem_dpath, fn) for fn in os.listdir(problem_dpath)
                         if fn.endswith('.pkl')]
    problems_ifpathes.sort()
    for i, ifpath in enumerate(problems_ifpathes):
        with open(ifpath, 'rb') as fp:
            problem = pickle.load(fp)
        prefix = opath.basename(ifpath)[:-len('.pkl')]
        ###############################################################
        # GH
        # probSetting = {'problem': problem}
        # ghLogF = opath.join(log_dpath, '%s-%s.log' % (prefix, 'GH'))
        # ghResF = opath.join(res_dpath, '%s-%s.csv' % (prefix, 'GH'))
        # etcSetting = {'LogFile': ghLogF,
        #               'ResFile': ghResF}
        # GH_run(probSetting, etcSetting)
        ###############################################################
        #
        ###############################################################
        # BNP
        probSetting = {'problem': problem}
        bnpLogF = opath.join(log_dpath, '%s-%s.log' % (prefix, 'BNP'))
        bnpResF = opath.join(res_dpath, '%s-%s.csv' % (prefix, 'BNP'))
        bptFile = opath.join(bbt_dpath, '%s-%s.csv' % (prefix, 'bnpTree'))
        etcSetting = {'LogFile': bnpLogF,
                      'ResFile': bnpResF,
                      'bptFile': bptFile,
                      'TimeLimit': _TimeLimit}
        grbSetting = {'LogFile': bnpLogF,
                      'Threads': _numThreads}
        BNP_run(probSetting, etcSetting, grbSetting)
        ###############################################################
        #
        ###############################################################
        # EX
        # probSetting = {'problem': problem}
        # exLogF = opath.join(log_dpath, '%s-%s.log' % (prefix, 'EX'))
        # exResF = opath.join(res_dpath, '%s-%s.csv' % (prefix, 'EX'))
        # etcSetting = {'LogFile': exLogF,
        #               'ResFile': exResF,
        #               'TimeLimit': _TimeLimit}
        # grbSetting = {'LogFile': exLogF,
        #               'Threads': _numThreads}
        # EX_run(probSetting, etcSetting, grbSetting)
        ###############################################################
        # os.remove(ifpath)


if __name__ == '__main__':
    # randomProb_5by5()
    # gen_problems(opath.join(dpath['experiment'], 'tempProb'))
    run_experiments(100)