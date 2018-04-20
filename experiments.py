import os.path as opath
import os
import shutil
import pickle
import csv
from random import choice, randrange
import multiprocessing
import time
#
from init_project import dpath
from problems import input_validity
# from GH import run as GH_run
# from EX import run as EX_run
from BNP import run as BNP_run
# from CWL import run as CWL_run


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


    for numPaths in [
                     5,
                     10,
                     20,
                     30
    ]:
        for numTasks, numBundles in [
                                     (10, 4),
                                     (20, 8),
                                     (30, 12),
                                     (40, 16), (50, 20), (60, 24),
                                     (70, 28),
                                     ]:
            problem = randomProb_5by5(problem_dpath, numTasks, numBundles, thVolume, numPaths, minTravlTime4OD, maxFlow, thDetour)


def randomProb_5by5(pkl_dir, numTasks=10, numBundles=3, thVolume=4, numPaths=5, minTravlTime4OD=5, maxFlow=3, thDetour=3):
    problemName = 'nt%04d-nb%04d-np%03d-tv%02d-td%02d-%d' % (numTasks, numBundles, numPaths,
                                                             thVolume, thVolume, int(time.time()))
    points, travel_time = {}, {}
    numLocations = 0
    ij_pid = {}
    for i in range(5):
        for j in range(5):
            points[numLocations] = (numLocations, i, j)
            ij_pid[i, j] = numLocations
            numLocations += 1
    travel_time = [[0 for _ in range(numLocations)] for _ in range(numLocations)]
    for p0 in points.values():
        for p1 in points.values():
            travel_time[p0[0]][p1[0]] = abs(p0[1] - p1[1]) + abs(p0[2] - p1[2])
    flows = {}
    od_paris = []
    for p0 in points.values():
        for p1 in points.values():
            if travel_time[p0[0]][p1[0]] > minTravlTime4OD:
                od_paris.append((p0[0], p1[0]))
    assert numPaths <= len(od_paris)
    while len(flows) != numPaths:
        ori, dest = choice(od_paris)
        while (ori, dest) in flows:
            ori, dest = choice(od_paris)
        flows[ori, dest] = randrange(maxFlow)
    paths = list(flows.keys())
    #
    tasks = []
    pd_paris = []
    for p0 in points.values():
        for p1 in points.values():
            if travel_time[p0[0]][p1[0]] != 0:
                pd_paris.append((p0[0], p1[0]))
    for tid in range(numTasks):
        pp, dp = choice(pd_paris)
        tasks.append((tid, pp, dp, 1, 1))
    #
    input_validity(points, flows, paths, tasks, numBundles, thVolume)
    #
    problem = [problemName,
               numLocations, travel_time, flows,
               len(tasks), tasks,
               numBundles, thVolume, thDetour]
    #
    with open(opath.join(pkl_dir, '%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(problem, fp)
    #
    return problem





def run_experiments(machine_num):
    _numThreads, _TimeLimit = multiprocessing.cpu_count(), 10 * 60 * 60
    machine_dpath = opath.join(dpath['experiment'], 'm%d' % machine_num)
    problem_dpath = opath.join(machine_dpath, '__problems')
    for path in [machine_dpath, problem_dpath]:
        assert opath.exists(path), path
    log_dpath = opath.join(machine_dpath, 'log')
    res_dpath = opath.join(machine_dpath, 'res')
    bbt_dpath = opath.join(machine_dpath, 'bpt')
    itr_dpath = opath.join(machine_dpath, 'itr')
    for path in [log_dpath, res_dpath, itr_dpath, bbt_dpath]:
        if opath.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
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
        # EX
        # problemName = problem[0]
        # log_fpath = opath.join(log_dpath, '%s-EX.log' % problemName)
        # res_fpath = opath.join(res_dpath, '%s-EX.csv' % problemName)
        # etcSetting = {'LogFile': log_fpath,
        #               'ResFile': res_fpath,
        #               'TimeLimit': _TimeLimit}
        # grbSetting = {'LogFile': log_fpath,
        #               'Threads': _numThreads}
        # EX_run(problem, etcSetting, grbSetting)
        ###############################################################
        #
        ###############################################################
        # BNP
        problemName = problem[0]
        log_fpath = opath.join(log_dpath, '%s-BNP.log' % problemName)
        res_fpath = opath.join(res_dpath, '%s-BNP.csv' % problemName)
        bpt_fpath = opath.join(bbt_dpath, '%s-bnpTree.csv' % problemName)
        etcSetting = {'LogFile': log_fpath,
                      'ResFile': res_fpath,
                      'bptFile': bpt_fpath,
                      'TimeLimit': _TimeLimit}
        grbSetting = {'LogFile': log_fpath,
                      'Threads': _numThreads}
        BNP_run(problem, etcSetting, grbSetting)
        ###############################################################
        #
        ###############################################################
        # CWL
        # probSetting = {'problem': problem}
        # cwlLogF = opath.join(log_dpath, '%s-%s.log' % (prefix, 'CWL'))
        # cwlResF = opath.join(res_dpath, '%s-%s.csv' % (prefix, 'CWL'))
        # itrFile = opath.join(itr_dpath, '%s-%s.csv' % (prefix, 'itrCWL'))
        # etcSetting = {'LogFile': cwlLogF,
        #               'ResFile': cwlResF,
        #               'itrFile': itrFile,
        #               # 'numPros': _numThreads,
        #               'TimeLimit': _TimeLimit}
        # grbSetting = {'LogFile': cwlLogF,
        #               'Threads': _numThreads}
        # CWL_run(probSetting, etcSetting, grbSetting)
        ###############################################################
        # os.remove(ifpath)


def summary():
    sum_fpath = opath.join(dpath['experiment'], 'experiment_summary.csv')
    with open(sum_fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['ts', 'numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour', 'avgNTinB',
                  'numDV', 'numCnts']
        header += ['ex_objV', 'ex_mipG(%)', 'ex_cpuT(h)', 'ex_cpuT(s)']
        header += ['bnp_objV', 'bnp_mipG(%)', 'bnp_cpuT(h)', 'bnp_cpuT(s)', 'bnp_exG(%)']
        header += ['cwl_objV', 'cwl_cpuT(h)', 'cwl_cpuT(s)', 'cwl_exG(%)', 'cwl_bnpG(%)']
        header += ['gh_objV', 'gh_cpuT(s)', 'gh_exG(%)', 'gh_bnpG(%)', 'gh_cwlG(%)']
        writer.writerow(header)
    #
    sum_dpath = opath.join(dpath['experiment'], 'summary')
    problem_dpath = opath.join(sum_dpath, '__problems')
    res_dpath = opath.join(sum_dpath, 'res')
    log_dpath = opath.join(sum_dpath, 'log')
    fns = os.listdir(problem_dpath)
    numTasks_fns = [(int(fn[:-len('.pkl')].split('-')[0][len('nt'):]), fn) for fn in fns if fn.endswith('.pkl')]
    numTasks_fns.sort()

    for _, fn in numTasks_fns:
        if not fn.endswith('.pkl'):
            continue
        prefix = fn[:-len('.pkl')]
        exLogF = opath.join(log_dpath, '%s-ex.log' % prefix)
        if opath.exists(exLogF):
            with open(exLogF, 'r') as f:
                l = f.readline()
                while l:
                    if l.startswith('Optimize a model with'):
                        break
                    l = f.readline()
            _rows, _cols = l.split(',')
            numRows = int(_rows[len('Optimize a model with '):-len(' rows')])
            numCols = int(_cols.split(' ')[1])
        else:
            numRows, numCols = None, None
        with open(sum_fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            contents = prefix.split('-')
            new_row = [contents[-1]]
            for i, p in enumerate(contents):
                if i == len(contents) - 1:
                    continue
                new_row.append(int(p[len('xx'):]))
            new_row += [new_row[1] / float(new_row[3])]
            new_row += [numCols, numRows]
            #
            exResF = opath.join(res_dpath, '%s-ex.csv' % prefix)
            exLogF = opath.join(log_dpath, '%s-ex.log' % prefix)
            exObjV, mipG, cpuTh, cpuTs = read_result(exResF, exLogF)
            new_row += [exObjV, mipG, cpuTh, cpuTs]
            #
            bnpResF = opath.join(res_dpath, '%s-bnp.csv' % prefix)
            bnpLogF = opath.join(log_dpath, '%s-bnp.log' % prefix)
            bnpObjV, mipG, cpuTh, cpuTs = read_result(bnpResF, bnpLogF)
            if type(bnpObjV) == float:
                exG = (exObjV - bnpObjV) / exObjV * 100 if type(exObjV) is float else '-'
            else:
                exG = None
            new_row += [bnpObjV, mipG, cpuTh, cpuTs, exG]
            #
            cwlResF = opath.join(res_dpath, '%s-cwl.csv' % prefix)
            cwlLogF = opath.join(log_dpath, '%s-cwl.log' % prefix)
            cwlObjV, mipG, cpuTh, cpuTs = read_result(cwlResF, cwlLogF)
            if type(cwlObjV) == float:
                exG = (exObjV - cwlObjV) / exObjV * 100 if type(exObjV) is float else '-'
                bnpG = (bnpObjV - cwlObjV) / bnpObjV * 100 if type(bnpObjV) is float else '-'
            else:
                exG, bnpG = None, None
            new_row += [cwlObjV, cpuTh, cpuTs, exG, bnpG]
            #
            ghResF = opath.join(res_dpath, '%s-gh.csv' % prefix)
            ghLogF = opath.join(log_dpath, '%s-gh.log' % prefix)
            ghObjV, mipG, cpuTh, cpuTs = read_result(ghResF, ghLogF)
            if type(ghObjV) == float:
                exG = (exObjV - ghObjV) / exObjV * 100 if type(exObjV) is float else '-'
                bnpG = (bnpObjV - ghObjV) / bnpObjV * 100 if type(bnpObjV) is float else '-'
                cwlG = (cwlObjV - ghObjV) / cwlObjV * 100 if type(cwlObjV) is float else '-'
            else:
                exG, bnpG, cwlG = None, None, None
            new_row += [ghObjV, cpuTs, exG, bnpG, cwlG]
            #
            writer.writerow(new_row)


def read_result(resF,logF):
    if opath.exists(resF):
        with open(resF) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                objV, mipG, _, cpuTs = [row[cn] for cn in ['objV', 'Gap', 'eliWallTime', 'eliCpuTime']]
        if eval(objV) == -1:
            objV, cpuTh, cpuTs, mipG = '-', '10h^', '-', '-'
        else:
            cpuTh = eval(cpuTs) / 3600
            mipG = eval(mipG) * 100 if mipG != '' else None
            objV = float(objV)
    else:
        if opath.exists(logF):
            objV, cpuTh, cpuTs, mipG = '-', '10h^', '-', '-'
        else:
            objV, cpuTh, cpuTs, mipG = None, None, None, None
    return objV, mipG, cpuTh, cpuTs


def gen_mrtProblems(problem_dpath):
    seedNum, numTasks, thVolume, thDetour = 0, 10, 3, 30
    #
    if not opath.exists(problem_dpath):
        os.mkdir(problem_dpath)
    for numTasks in [20, 40, 60, 80, 100]:
        prob_dpath = opath.join(dpath['experiment'], 'mrtNet-T%d-k%d-vt%d-dt%d' %
                                (numTasks, len(repStations) * (len(repStations) - 1), thVolume, thDetour))
        _flows, \
        _tasks, rewards, volumes, \
        _points, _travel_time, \
        numBundles, thVolume, thDetour = get_mrtNetExample(prob_dpath, numTasks, thVolume, thDetour)
        #
        for thD in [20, 40, 60, 80]:
            flows, tasks, points, travel_time = convert_mrtNet2ID(_flows, _tasks, _points, _travel_time)
            #
            paths = list(flows.keys())
            #
            input_validity(points, flows, paths, tasks, numBundles, thVolume)
            #
            inputs = [travel_time,
                      flows, paths,
                      tasks, rewards, volumes,
                      numBundles, thVolume, thD]
            #
            travel_time, \
            flows, paths, \
            tasks, rewards, volumes, \
            numBundles, thVolume, thD = inputs
            numTasks, numPaths = map(len, [tasks, paths])
            fn = 'nt%02d-np%d-nb%d-tv%d-td%d.pkl' % (numTasks, numPaths, numBundles, thVolume, thD)
            ofpath = opath.join(problem_dpath, fn)
            with open(ofpath, 'wb') as fp:
                pickle.dump(inputs, fp)

if __name__ == '__main__':
    # randomProb_5by5()
    # gen_problems(opath.join(dpath['experiment'], 'tempProb'))
    run_experiments(1)
    # gen_mrtProblems(opath.join(dpath['experiment'], 'tempProb'))
    # summary()