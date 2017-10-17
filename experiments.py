from init_project import *
#
from cpuinfo import get_cpu_info
from psutil import virtual_memory
import platform
import shutil
import pickle, csv
#
from exactMM import run as exactMM_run
from colGenMM import run as colGenMM_run

try:
    from greedyHeuristic import run as gHeuristic_run
except ModuleNotFoundError:
    from setup import cythonize

    cythonize('greedyHeuristic')
    #
    from greedyHeuristic import run as gHeuristic_run

from problems import *


def gen_problems(problem_dpath):
    #
    # Generate problems
    #
    problem_summary_fpath = opath.join(problem_dpath, '__problem_summary.csv')
    with open(problem_summary_fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        new_headers = ['fn', 'numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour']
        writer.writerow(new_headers)
    #
    maxFlow = 3
    minReward, maxReward = 1, 3
    minVolume, maxVolume = 1, 3
    volumeAlowProp, detourAlowProp = 1.5, 1.2
    numCols, numRows = 1, 5
    #
    numBundles = 4
    for numTasks in [6, 8, 10]:
        inputs = random_problem(numCols, numRows, maxFlow,
                                numTasks, minReward, maxReward, minVolume, maxVolume,
                                numBundles, volumeAlowProp, detourAlowProp)
        save_aProblem(inputs, problem_dpath, problem_summary_fpath)

        # for numTasks in range(6, 30, 2):
        #     for numBundles in range(3, max(4, int(numTasks / 4))):
        #
        #         inputs = random_problem(numCols, numRows, maxFlow,
        #                                 numTasks, minReward, maxReward, minVolume, maxVolume,
        #                                 numBundles, volumeAlowProp, detourAlowProp)
        #         save_aProblem(inputs, problem_dpath, problem_summary_fpath)


def save_aProblem(inputs, problem_dpath, problem_summary_fpath):
    travel_time, \
    flows, paths, \
    tasks, rewards, volumes, \
    numBundles, thVolume, thDetour = inputs
    numTasks, numPaths = map(len, [tasks, paths])
    fn = 'nt%02d-np%d-nb%d-tv%d-td%d.pkl' % (numTasks, numPaths, numBundles, thVolume, thDetour)
    with open(problem_summary_fpath, 'a') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        writer.writerow([fn, numTasks, numPaths, numBundles, thVolume, thDetour])
    ofpath = opath.join(problem_dpath, fn)
    with open(ofpath, 'wb') as fp:
        pickle.dump(inputs, fp)


def init_expEnv(initEnv=False):
    cpu_info = get_cpu_info()
    # exp_dpath = opath.join(dpath['experiment'], str(cpu_info['brand']))
    exp_dpath = opath.join(dpath['experiment'], str(platform.node()))
    problem_dpath = opath.join(exp_dpath, '__problem')
    log_dpath = opath.join(exp_dpath, 'log')
    res_dpath = opath.join(exp_dpath, 'res')
    if initEnv and opath.exists(exp_dpath):
        shutil.rmtree(exp_dpath)
    try:
        if not opath.exists(exp_dpath):
            for path in [exp_dpath, problem_dpath, log_dpath, res_dpath]:
                os.makedirs(path)
            #
            cpu_spec_fpath = opath.join(exp_dpath, '__cpuSpec.txt')
            with open(cpu_spec_fpath, 'w') as f:
                f.write('numProcessor: %d\n' % int(cpu_info['count']))
                f.write('bits: %d\n' % int(cpu_info['bits']))
                f.write('brand:%s' % str(cpu_info['brand']))
            gen_problems(problem_dpath)
    except:
        pass
    #
    return log_dpath, res_dpath, [opath.join(problem_dpath, fn) for fn in os.listdir(problem_dpath)
                                  if fn.endswith('.pkl')]


def record_res(fpath, nt, np, nb, tv, td, m, objV, gap, eliCpuTime, eliWallTiem):
    with open(fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour',
                  'method', 'objV', 'Gap', 'eliCpuTime', 'eliWallTime']
        writer.writerow(header)
        writer.writerow([nt, np, nb, tv, td, m, objV, gap, eliCpuTime, eliWallTiem])


def run_multipleCores(machine_num):
    cpu_info = get_cpu_info()
    _numThreads, _TimeLimit, _pfCst = int(cpu_info['count']), 4 * 60 * 60, 1.2
    #
    # log_dpath, res_dpath, problem_dpath = init_expEnv()
    machine_dpath = opath.join(dpath['experiment'], 'm%d' % machine_num)
    problem_dpath = opath.join(machine_dpath, '__problems')
    for path in [machine_dpath, problem_dpath]:
        assert opath.exists(path), path
    cpu_spec_fpath = opath.join(machine_dpath, '__cpuSpec.txt')
    with open(cpu_spec_fpath, 'w') as f:
        f.write('numProcessor: %d\n' % int(cpu_info['count']))
        f.write('bits: %d\n' % int(cpu_info['bits']))
        f.write('brand:%s\n' % str(cpu_info['brand']))
        f.write('memory:%d kb' % virtual_memory().total)
    log_dpath = opath.join(machine_dpath, 'log')
    res_dpath = opath.join(machine_dpath, 'res')
    for path in [log_dpath, res_dpath]:
        os.makedirs(path)
    problems_ifpathes = [opath.join(problem_dpath, fn) for fn in os.listdir(problem_dpath)
                         if fn.endswith('.pkl')]
    problems_ifpathes.sort()
    for i, ifpath in enumerate(problems_ifpathes):
        inputs = None
        with open(ifpath, 'rb') as fp:
            inputs = pickle.load(fp)
        prefix = opath.basename(ifpath)[:-len('.pkl')]
        nt, np, nb, tv, td = [int(v[len('xx'):]) for v in prefix.split('-')]
        #
        # gHeuristic
        #
        m = 'gHeuristic'
        try:
            objV, eliCpuTime = gHeuristic_run(inputs,
                                           log_fpath=opath.join(log_dpath, '%s-%s.log' % (prefix, m)))
        except:
            objV, eliCpuTime = -1, -1
        gap, eliWallTime = -1, -1
        record_res(opath.join(res_dpath, '%s-%s.csv' % (prefix, m)),
                   nt, np, nb, tv, td, m, objV, gap, eliCpuTime, eliWallTime)
        #
        # colGenMM
        #
        m = 'colGenMM'
        try:
            objV, gap, eliCpuTime, eliWallTime = colGenMM_run(inputs,
                                             log_fpath=opath.join(log_dpath, '%s-%s(%.2f).log' % (prefix, m, _pfCst)),
                                             numThreads=_numThreads, TimeLimit=_TimeLimit, pfCst=_pfCst)
        except:
            objV, gap, eliCpuTime, eliWallTime = -1, -1, -1, -1
        record_res(opath.join(res_dpath, '%s-%s(%.2f).csv' % (prefix, m, _pfCst)),
                   nt, np, nb, tv, td, m, objV, gap, eliCpuTime, eliWallTime)
        #
        # exactMM
        #
        m = 'exactMM'
        try:
            objV, gap, eliCpuTime, eliWallTime = exactMM_run(inputs,
                                        log_fpath=opath.join(log_dpath, '%s-%s.log' % (prefix, m)),
                                        numThreads=_numThreads, TimeLimit=_TimeLimit)
        except:
            objV, gap, eliCpuTime, eliWallTime = -1, -1, -1, -1
        record_res(opath.join(res_dpath, '%s-%s.csv' % (prefix, m)),
                   nt, np, nb, tv, td, m, objV, gap, eliCpuTime, eliWallTime)


def summary():
    sum_fpath = opath.join(dpath['experiment'], 'experiment_summary.csv')
    with open(sum_fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour',
                  'ex_objV', 'cg_objV', 'gh_objV',
                  'cg_optG', 'gh_optG',
                  'ex_comT', 'cg_comT', 'gh_comT', 'nodeSpec']
        writer.writerow(header)

    for dir_name in os.listdir(dpath['experiment']):
        if not dir_name.startswith('c'):
            continue
        dir_path = opath.join(dpath['experiment'], dir_name)
        spec_fpath = opath.join(dir_path, '__cpuSpec.txt')
        problem_dpath = opath.join(dir_path, '__problem')
        res_dpath = opath.join(dir_path, 'res')
        spec = None
        with open(spec_fpath, 'r') as f:
            spec = f.readlines()
        _numProcessor, _, _brand = spec
        numProcessor = _numProcessor.split(':')[1][:-1]
        brand = _brand.split(':')[1][:-1]
        for fn in os.listdir(problem_dpath):
            if not fn.endswith('.pkl'):
                continue
            with open(sum_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                new_row = []
                #
                prefix = fn[:-len('.pkl')]
                for p in prefix.split('-'):
                    new_row.append(int(p[len('xx'):]))
                #
                ex_res_fpath = opath.join(res_dpath, '%s-exactMM.csv' % prefix)
                cg_res_fpath = opath.join(res_dpath, '%s-colGenMM.csv' % prefix)
                he_res_fpath = opath.join(res_dpath, '%s-gHeuristic.csv' % prefix)
                objVs, comTs = [], []
                for fpath in [ex_res_fpath, cg_res_fpath, he_res_fpath]:
                    if opath.exists(fpath):
                        with open(fpath) as r_csvfile:
                            reader = csv.DictReader(r_csvfile)
                            for row in reader:
                                _, objV, comT = [row[cn] for cn in ['method', 'objV', 'eliTime']]
                    else:
                        objV, comT = -1, -1
                    objVs.append(objV)
                    comTs.append(comT)
                ex_objV, cg_objV, gh_objV = map(float, objVs)
                if ex_objV == -1:
                    cg_optG, gh_optG = -1, -1
                else:
                    cg_optG, gh_optG = (ex_objV - cg_objV) / ex_objV * 100, (ex_objV - gh_objV) / ex_objV * 100
                ex_comT, cg_comT, gh_comT = comTs
                nodeSpec = brand + '; cores ' + numProcessor
                new_row += [ex_objV, cg_objV, gh_objV]
                new_row += [cg_optG, gh_optG]
                new_row += [ex_comT, cg_comT, gh_comT]
                new_row += [nodeSpec]
                writer.writerow(new_row)


if __name__ == '__main__':
    summary()

    # cluster_run(0)
    # run_multipleCores()

    # run(0, num_workers=8)
    # local_run()
    # single_run('nt6-np20-nb4-tv2-td4.pkl')
    # run(0)
