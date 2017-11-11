from init_project import *
#
from cpuinfo import get_cpu_info
from psutil import virtual_memory
from traceback import format_exc
import pickle, csv
import time
#
from _utils.recording import *
from problems import *
#
from exactMM import run as exactMM_run
from bnpTree import BnPTree


def gen_problems(problem_dpath):
    #
    # Generate problems
    #
    if not opath.exists(problem_dpath):
        os.mkdir(problem_dpath)
    #
    maxFlow = 3
    minReward, maxReward = 1, 3
    minVolume, maxVolume = 1, 3
    volumeAlowProp, detourAlowProp = 1.5, 0.9
    numCols, numRows = 1, 4
    #

    numTasks, numBundles = 40, 10
    inputs = random_problem(numCols, numRows, maxFlow,
                            numTasks, minReward, maxReward, minVolume, maxVolume,
                            numBundles, volumeAlowProp, detourAlowProp)
    travel_time, \
    flows, paths, \
    tasks, rewards, volumes, \
    numBundles, thVolume, thDetour = inputs
    numTasks, numPaths = map(len, [tasks, paths])
    fn = 'nt%02d-np%d-nb%d-tv%d-td%d.pkl' % (numTasks, numPaths, numBundles, thVolume, thDetour)
    ofpath = opath.join(problem_dpath, fn)
    with open(ofpath, 'wb') as fp:
        pickle.dump(inputs, fp)


    # for numTasks, numBundles in [(5, 3), (10, 4), (15, 5), (20, 6)]:
    #     inputs = random_problem(numCols, numRows, maxFlow,
    #                             numTasks, minReward, maxReward, minVolume, maxVolume,
    #                             numBundles, volumeAlowProp, detourAlowProp)
    #     travel_time, \
    #     flows, paths, \
    #     tasks, rewards, volumes, \
    #     numBundles, thVolume, thDetour = inputs
    #     numTasks, numPaths = map(len, [tasks, paths])
    #     fn = 'nt%02d-np%d-nb%d-tv%d-td%d.pkl' % (numTasks, numPaths, numBundles, thVolume, thDetour)
    #     ofpath = opath.join(problem_dpath, fn)
    #     with open(ofpath, 'wb') as fp:
    #         pickle.dump(inputs, fp)


def run_multipleCores(machine_num):
    cpu_info = get_cpu_info()
    _numThreads, _TimeLimit = int(cpu_info['count']), 4 * 60 * 60
    _PoolSolutions = 1000
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
    bpt_dpath = opath.join(machine_dpath, 'bpt')
    err_dpath = opath.join(machine_dpath, 'err')
    for path in [log_dpath, res_dpath, err_dpath, bpt_dpath]:
        os.makedirs(path)
    problems_ifpathes = [opath.join(problem_dpath, fn) for fn in os.listdir(problem_dpath)
                         if fn.endswith('.pkl')]
    problems_ifpathes.sort()
    for i, ifpath in enumerate(problems_ifpathes):
        with open(ifpath, 'rb') as fp:
            inputs = pickle.load(fp)
        prefix = opath.basename(ifpath)[:-len('.pkl')]
        #
        # Run the exact model
        #
        exLogF = opath.join(log_dpath, '%s-%s.log' % (prefix, 'ex'))
        exResF = opath.join(res_dpath, '%s-%s.csv' % (prefix, 'ex'))
        #
        probSetting = {'problem': inputs}
        grbSetting = {'LogFile': exLogF,
                      'Threads': _numThreads,
                      'TimeLimit': _TimeLimit}
        etcSetting = {'exLogF': exLogF,
                      'exResF': exResF
                      }
        try:
            exactMM_run(probSetting, grbSetting, etcSetting)
        except:
            pass
        #
        # Run others
        #
        ghLogF = opath.join(log_dpath, '%s-%s.log' % (prefix, 'gh'))
        orLogF = opath.join(log_dpath, '%s-%s.log' % (prefix, 'or'))
        cgLogF = opath.join(log_dpath, '%s-%s.log' % (prefix, 'cg'))
        bnpLogF = opath.join(log_dpath, '%s-%s.log' % (prefix, 'bnp'))
        #
        ghResF = opath.join(res_dpath, '%s-%s.csv' % (prefix, 'gh'))
        orResF = opath.join(res_dpath, '%s-%s.csv' % (prefix, 'or'))
        cgResF = opath.join(res_dpath, '%s-%s.csv' % (prefix, 'cg'))
        bnpResF = opath.join(res_dpath, '%s-%s.csv' % (prefix, 'bnp'))
        #
        bptFile = opath.join(bpt_dpath, '%s-%s.csv' % (prefix, 'bnpTree'))
        #
        emsgFile = opath.join(err_dpath, '%s-%s.txt' % (prefix, 'bnp'))
        epklFile = opath.join(err_dpath, '%s-%s.pkl' % (prefix, 'bnp'))
        #
        probSetting = {'problem': inputs,
                       'inclusiveC': [], 'exclusiveC': []}
        grbSetting = {'LogFile': bnpLogF,
                      'Threads': _numThreads,
                      'TimeLimit': _TimeLimit,
                      'PoolSolutions': _PoolSolutions}
        etcSetting = {'ghLogF': ghLogF, 'orLogF': orLogF, 'cgLogF': cgLogF, 'bnpLogF': bnpLogF,
                      #
                      'ghResF': ghResF, 'orResF': orResF, 'cgResF': cgResF, 'bnpResF': bnpResF,
                      #
                      'bptFile': bptFile,
                      #
                      'EpklFile': epklFile, 'EmsgFile': emsgFile,
                      }
        try:
            BnPTree(probSetting, grbSetting, etcSetting).startBnP()
        except:
            pass
        #
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
            header += ['bnp(%.2f)_objV' % pfConst,
                       'bnp(%.2f)_wallT(h)' % pfConst,
                       'bnp(%.2f)_wallT(s)' % pfConst,
                       'bnp(%.2f)_cpuT(s)' % pfConst,
                       'bnp(%.2f)_optG(%%)' % pfConst]
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

    sum_dpath = opath.join(dpath['experiment'], 'summary')
    spec_fpath = opath.join(sum_dpath, '__cpuSpec.txt')
    with open(spec_fpath, 'r') as f:
        spec = f.readlines()
    _numProcessor, _, _brand, _memoryS = spec
    numProcessor = _numProcessor.split(':')[1][:-1]
    brand = _brand.split(':')[1][:-1]
    memoryS = '%.2fGB' % (int(_memoryS.split(':')[1][:-3]) / (1024 ** 3))
    problem_dpath = opath.join(sum_dpath, '__problems')
    res_dpath = opath.join(sum_dpath, 'res')
    log_dpath = opath.join(sum_dpath, 'log')
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
            # bnp
            #
            for pfConst in powCnsts:
                cg_res_fpath = opath.join(res_dpath, '%s-bnpM(%.2f).csv' % (prefix, pfConst))
                if opath.exists(cg_res_fpath):
                    with open(cg_res_fpath) as r_csvfile:
                        reader = csv.DictReader(r_csvfile)
                        for row in reader:
                            objV, wallTs, cpuT = [row[cn] for cn in ['objV', 'eliWallTime', 'eliCpuTime']]
                    if eval(objV) == -1:
                        objV, wallTh, wallTs, cpuT, optG = '-', '4h*', '-', '-', '-'
                    else:
                        wallTh = eval(wallTs) / 3600
                        optG = (ex_objV - eval(objV)) / ex_objV * 100 if type(ex_objV) is float else '-'
                else:
                    objV, wallTh, wallTs, cpuT, optG = None, None, None, None, None
                new_row += [objV, wallTh, wallTs, cpuT, optG]
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
    # run_multipleCores(0)
    summary()
    # gen_problems(opath.join(dpath['experiment'], 'tempProb'))

