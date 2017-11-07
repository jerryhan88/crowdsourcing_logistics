from init_project import *
#
from cpuinfo import get_cpu_info
from psutil import virtual_memory
import platform
import shutil, os
import pickle, csv
#
from __old.exactMM import run as exactMM_run

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
    numCols, numRows = 1, 4
    #
    numBundles = 4
    # for numBundles in [20, 30]:
    for numTasks in [10, 12]:
        inputs = random_problem(numCols, numRows, maxFlow,
                                numTasks, minReward, maxReward, minVolume, maxVolume,
                                numBundles, volumeAlowProp, detourAlowProp)
        save_aProblem(inputs, problem_dpath, problem_summary_fpath)


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
    _numThreads, _TimeLimit = int(cpu_info['count']), 4 * 60 * 60
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
        # m = 'gHeuristic'
        # objV, eliCpuTime, B = gHeuristic_run(inputs,
        #                                log_fpath=opath.join(log_dpath, '%s-%s.log' % (prefix, m)))
        # gap, eliWallTime = None, None
        # record_res(opath.join(res_dpath, '%s-%s.csv' % (prefix, m)),
        #            nt, np, nb, tv, td, m, objV, gap, eliCpuTime, eliWallTime)
        # #
        # bB, \
        # T, r_i, v_i, _lambda, P, D, N, \
        # K, w_k, t_ij, _delta = convert_input4MathematicalModel(*inputs)
        # objV = 0
        # startCpuTime, startWallTime = time.clock(), time.time()
        # for b in B:
        #     p = 0
        #     br = sum([r_i[i] for i in b])
        #     for k, w in enumerate(w_k):
        #         if minTimePD(b, k, t_ij, log_fpath=opath.join(log_dpath, '%s-%s(minPD).log' % (prefix, m))) < _delta:
        #             p += w * br
        #     objV += p
        # endCpuTime, endWallTime = time.clock(), time.time()
        # eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
        # gap = None
        # record_res(opath.join(res_dpath, '%s-%s(minPD).csv' % (prefix, m)),
        #            nt, np, nb, tv, td, m, objV, gap, eliCpuTime, eliWallTime)
        #
        # colGenMM
        #
        # m = 'colGenMM'
        # for _pfCst in [1.2, 1.5]:
        # for _pfCst in [1.5]:
        #     try:
        #         objV, gap, eliCpuTime, eliWallTime = colGenMM_run(inputs,
        #                                          log_fpath=opath.join(log_dpath, '%s-%s(%.2f).log' % (prefix, m, _pfCst)),
        #                                          numThreads=_numThreads, TimeLimit=_TimeLimit, pfCst=_pfCst)
        #     except:
        #         import sys
        #         with open('%s_error.txt' % sys.argv[0], 'w') as f:
        #             f.write(format_exc())
        #         objV, gap, eliCpuTime, eliWallTime = -1, -1, -1, -1
        #     record_res(opath.join(res_dpath, '%s-%s(%.2f).csv' % (prefix, m, _pfCst)),
        #                nt, np, nb, tv, td, m, objV, gap, eliCpuTime, eliWallTime)
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
        os.remove(ifpath)


def summary():
    sum_fpath = opath.join(dpath['experiment'], 'experiment_summary.csv')
    powCnsts = [1.20, 1.50]
    with open(sum_fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['nodeSpec',
                  'numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour', 'avgNTinB',
                  'numDV', 'numCnts',
                  'ex_objV', 'ex_wallT(h)', 'ex_wallT(s)', 'ex_cpuT(s)', 'ex_mipG(%)',
                  ]
        for pfConst in powCnsts:
            header += ['cg(%.2f)_objV' % pfConst,
                       'cg(%.2f)_wallT(h)' % pfConst,
                       'cg(%.2f)_wallT(s)' % pfConst,
                       'cg(%.2f)_cpuT(s)' % pfConst,
                       'cg(%.2f)_optG(%%)' % pfConst]
        header += ['gh_objV', 'gh_cpuT(s)', 'gh_optG(%)']
        header += ['gh_cg(%.2f)G' % pfConst for pfConst in powCnsts]
        header += ['ghOR_objV', 'ghOR_wallT(h)', 'ghOR_wallT(s)', 'ghOR_cpuT(s)', 'ghOR_optG(%)']
        header += ['ghOR_cg(%.2f)G' % pfConst for pfConst in powCnsts]
        writer.writerow(header)
    for machineName in os.listdir(dpath['experiment']):
        if not (machineName.startswith('_m') or machineName.startswith('m')):
            continue
        dir_path = opath.join(dpath['experiment'], machineName)
        spec_fpath = opath.join(dir_path, '__cpuSpec.txt')
        spec = None
        with open(spec_fpath, 'r') as f:
            spec = f.readlines()
        _numProcessor, _, _brand, _memoryS = spec
        numProcessor = _numProcessor.split(':')[1][:-1]
        brand = _brand.split(':')[1][:-1]
        memoryS = '%.2fGB' % (int(_memoryS.split(':')[1][:-3]) / (1024 ** 3))
        problem_dpath = opath.join(dir_path, '__problems')
        res_dpath = opath.join(dir_path, 'res')
        log_dpath = opath.join(dir_path, 'log')
        for fn in os.listdir(problem_dpath):
            if not fn.endswith('.pkl'):
                continue
            prefix = fn[:-len('.pkl')]
            ex_log_fpath = opath.join(log_dpath, '%s-exactMM.log' % prefix)
            if opath.exists(ex_log_fpath):
                with open(ex_log_fpath, 'r') as f:
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
                nodeSpec = brand + '; cores ' + numProcessor, '; memory ' + memoryS
                new_row = [nodeSpec]
                #
                for p in prefix.split('-'):
                    new_row.append(int(p[len('xx'):]))
                new_row += [new_row[1] / new_row[3]]
                #
                new_row += [numCols, numRows]
                #
                # exMM
                #
                ex_res_fpath = opath.join(res_dpath, '%s-exactMM.csv' % prefix)
                gh_res_fpath = opath.join(res_dpath, '%s-gHeuristic.csv' % prefix)
                ghPD_res_fpath = opath.join(res_dpath, '%s-gHeuristic(minPD).csv' % prefix)
                if opath.exists(ex_res_fpath):
                    with open(ex_res_fpath) as r_csvfile:
                        reader = csv.DictReader(r_csvfile)
                        for row in reader:
                            objV, mipG, wallTs, cpuT = [row[cn] for cn in ['objV', 'Gap', 'eliWallTime', 'eliCpuTime']]
                    if eval(objV) == -1:
                        objV, wallTh, wallTs, cpuT, mipG = '-', '4h^', '-', '-', '-'
                    else:
                        wallTh = eval(wallTs) / 3600
                        mipG = eval(mipG) * 100
                else:
                    objV, wallTh, wallTs, cpuT, mipG = None, None, None, None, None
                new_row += [objV, wallTh, wallTs, cpuT, mipG]
                if (objV is not None) and (objV is not '-'):
                    ex_objV = float(objV)
                else:
                    ex_objV = None
                #
                # colGenMM
                #
                colGenMM_objs = {}
                for pfConst in powCnsts:
                    cg_res_fpath = opath.join(res_dpath, '%s-colGenMM(%.2f).csv' % (prefix, pfConst))
                    if opath.exists(cg_res_fpath):
                        with open(cg_res_fpath) as r_csvfile:
                            reader = csv.DictReader(r_csvfile)
                            for row in reader:
                                objV, wallTs, cpuT = [row[cn] for cn in ['objV', 'eliWallTime', 'eliCpuTime']]
                        if eval(objV) == -1:
                            objV, wallTh, wallTs, cpuT, optG = '-', '4h*', '-', '-', '-'
                            colGenMM_objs[pfConst] = None
                        else:
                            wallTh = eval(wallTs) / 3600
                            optG = (ex_objV - eval(objV)) / ex_objV * 100 if type(ex_objV) is float else '-'
                            colGenMM_objs[pfConst] = float(objV)
                    else:
                        objV, wallTh, wallTs, cpuT, optG = None, None, None, None, None
                        colGenMM_objs[pfConst] = None
                    new_row += [objV, wallTh, wallTs, cpuT, optG]
                #
                # greedy heuristic
                #
                if not opath.exists(gh_res_fpath):
                    new_row += [None, None, None]
                    new_row += [None for _ in powCnsts]
                    new_row += [None, None, None, None, None]
                    new_row += [None for _ in powCnsts]
                    writer.writerow(new_row)
                    continue
                with open(gh_res_fpath) as r_csvfile:
                    reader = csv.DictReader(r_csvfile)
                    for row in reader:
                        objV, cpuT = [row[cn] for cn in ['objV', 'eliCpuTime']]
                optG = (ex_objV - eval(objV)) / ex_objV * 100 if type(ex_objV) is float else '-'
                new_row += [objV, cpuT, optG]
                for pfConst in powCnsts:
                    if colGenMM_objs[pfConst] is not None:
                        new_row += [(colGenMM_objs[pfConst] - eval(objV)) / colGenMM_objs[pfConst] * 100]
                    else:
                        new_row += [None]
                #
                if opath.exists(ghPD_res_fpath):
                    with open(ghPD_res_fpath) as r_csvfile:
                        reader = csv.DictReader(r_csvfile)
                        for row in reader:
                            objV, wallTs, cpuT = [row[cn] for cn in ['objV', 'eliWallTime', 'eliCpuTime']]
                    wallTh = eval(wallTs) / 3600
                    optG = (ex_objV - eval(objV)) / ex_objV * 100 if type(ex_objV) is float else '-'
                    new_row += [objV, wallTh, wallTs, cpuT, optG]
                    for pfConst in powCnsts:
                        if colGenMM_objs[pfConst] is not None:
                            new_row += [(colGenMM_objs[pfConst] - eval(objV)) / colGenMM_objs[pfConst] * 100]
                        else:
                            new_row += [None]
                else:
                    objV, wallTh, wallTs, cpuT, optG = None, None, None, None, None
                    new_row += [None for _ in powCnsts]
                writer.writerow(new_row)


if __name__ == '__main__':
    summary()

    # machine_dpath = opath.join(dpath['experiment'], 'm4')
    # os.makedirs(machine_dpath)
    # problem_dpath = opath.join(machine_dpath, '__problems')
    # os.makedirs(problem_dpath)
    # gen_problems(problem_dpath)

    # cluster_run(0)
    # run_multipleCores(2000)

    # run(0, num_workers=8)
    # local_run()
    # single_run('nt6-np20-nb4-tv2-td4.pkl')
    # run(0)
