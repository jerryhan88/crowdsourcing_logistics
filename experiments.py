from init_project import *
#
from cpuinfo import get_cpu_info
from psutil import virtual_memory

#
from _utils.recording import *
#
from exactMM import run as exactMM_run
from bnpTree import BnPTree
from problems import *


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
    volumeAlowProp, detourAlowProp = 1.5, 0.8
    numCols, numRows = 1, 4
    #
    numTasks, numBundles = 35, 11
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
    bbt_dpath = opath.join(machine_dpath, 'bpt')
    err_dpath = opath.join(machine_dpath, 'err')
    for path in [log_dpath, res_dpath, err_dpath, bbt_dpath]:
        os.makedirs(path)
    problems_ifpathes = [opath.join(problem_dpath, fn) for fn in os.listdir(problem_dpath)
                         if fn.endswith('.pkl')]
    problems_ifpathes.sort()
    for i, ifpath in enumerate(problems_ifpathes):
        with open(ifpath, 'rb') as fp:
            inputs = pickle.load(fp)
        prefix = opath.basename(ifpath)[:-len('.pkl')]
        # Run others
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
        bptFile = opath.join(bbt_dpath, '%s-%s.csv' % (prefix, 'bnpTree'))
        #
        emsgFile = opath.join(err_dpath, '%s-%s.txt' % (prefix, 'bnp'))
        epklFile = opath.join(err_dpath, '%s-%s.pkl' % (prefix, 'bnp'))
        #
        probSetting = {'problem': inputs,
                       'inclusiveC': [], 'exclusiveC': []}
        grbSetting = {'LogFile': bnpLogF,
                      'Threads': _numThreads,
                      'TimeLimit': _TimeLimit,
                      'PoolSolutions': _PoolSolutions,
                      # 'Method': 1  # dual simplex
                      # 'Method': 2  # barrier

                      }
        etcSetting = {'ghLogF': ghLogF, 'orLogF': orLogF, 'cgLogF': cgLogF, 'bnpLogF': bnpLogF,
                      #
                      'ghResF': ghResF, 'orResF': orResF, 'cgResF': cgResF, 'bnpResF': bnpResF,
                      #
                      'bptFile': bptFile,
                      #
                      'EpklFile': epklFile, 'EmsgFile': emsgFile,
                      #
                      'use_ghS': True
                      }
        try:
            BnPTree(probSetting, grbSetting, etcSetting).startBnP()
        except:
            pass
        #
        # Run the exact model
        #
        # exLogF = opath.join(log_dpath, '%s-%s.log' % (prefix, 'ex'))
        # exResF = opath.join(res_dpath, '%s-%s.csv' % (prefix, 'ex'))
        # #
        # probSetting = {'problem': inputs}
        # grbSetting = {'LogFile': exLogF,
        #               'Threads': _numThreads,
        #               'TimeLimit': _TimeLimit}
        # etcSetting = {'exLogF': exLogF,
        #               'exResF': exResF
        #               }
        # try:
        #     exactMM_run(probSetting, grbSetting, etcSetting)
        # except:
        #     pass
        #
        os.remove(ifpath)


def summary():
    sum_fpath = opath.join(dpath['experiment'], 'experiment_summary.csv')
    with open(sum_fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['nodeSpec',
                  'numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour', 'avgNTinB',
                  'numDV', 'numCnts']
        header += ['ex_objV', 'ex_mipG(%)',
                   'ex_wallT(h)', 'ex_wallT(s)', 'ex_cpuT(s)']
        header += ['bnp_objV', 'bnp_mipG(%)',
                   'bnp_wallT(h)', 'bnp_wallT(s)', 'bnp_cpuT(s)',
                   'bnp_exG(%)']
        header += ['opb_objV', 'opb_mipG(%)',
                   'opb_wallT(h)', 'opb_wallT(s)', 'opb_cpuT(s)',
                   'opb_exG(%)', 'opb_bnpG(%)']
        header += ['cg_objV', 'cg_mipG(%)',
                   'cg_wallT(h)', 'cg_wallT(s)', 'cg_cpuT(s)',
                   'cg_exG(%)', 'cg_bnpG(%)']
        header += ['or_objV',
                   'or_wallT(h)', 'or_wallT(s)', 'or_cpuT(s)',
                   'or_exG(%)', 'or_bnpG(%)', 'or_cgG(%)']
        header += ['gh_objV',
                   'gh_wallT(s)', 'gh_cpuT(s)',
                   'gh_exG(%)', 'gh_bnpG(%)', 'gh_cgG(%)', 'gh_orG(%)']
        writer.writerow(header)
    #
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
    bpt_dpath = opath.join(sum_dpath, 'bpt')
    fns = os.listdir(problem_dpath)
    numTasks_fns = [(int(fn[:-len('.pkl')].split('-')[0][len('nt'):]), fn) for fn in fns if fn.endswith('.pkl')]
    numTasks_fns.sort()
    for _, fn in numTasks_fns:
        if not fn.endswith('.pkl'):
            continue
        prefix = fn[:-len('.pkl')]
        print(prefix)
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
            nodeSpec = brand + '; cores ' + numProcessor, '; memory ' + memoryS
            new_row = [nodeSpec]
            #
            for p in prefix.split('-'):
                new_row.append(int(p[len('xx'):]))
            new_row += [new_row[1] / new_row[3]]
            new_row += [numCols, numRows]
            #
            exResF = opath.join(res_dpath, '%s-ex.csv' % prefix)
            exLogF = opath.join(log_dpath, '%s-ex.log' % prefix)
            if opath.exists(exLogF):
                if opath.exists(exResF):
                    objV, mipG, wallTs, cpuT = read_result(exResF)
                    if eval(objV) == -1:
                        objV, wallTh, wallTs, cpuT, mipG = '-', '4h^', '-', '-', '-'
                    else:
                        exObjV = float(objV)
                        wallTh, mipG = eval(wallTs) / 3600, eval(mipG) * 100
                else:
                    objV, wallTh, wallTs, cpuT, mipG = '-', '4h^', '-', '-', '-'
                    exObjV = objV
            else:
                objV, wallTh, wallTs, cpuT, mipG = None, None, None, None, None
                exObjV = objV
            new_row += [objV, mipG,
                        wallTh, wallTs, cpuT]
            #
            bnpResF = opath.join(res_dpath, '%s-bnp.csv' % prefix)
            if not opath.exists(bnpResF):
                new_row += [None for _ in ['bnp_objV', 'bnp_mipG(%)',
                                           'bnp_wallT(h)', 'bnp_wallT(s)', 'bnp_cpuT(s)',
                                           'bnp_exG(%)']]
                new_row += [None for _ in ['opb_objV', 'opb_mipG(%)',
                                           'opb_wallT(h)', 'opb_wallT(s)', 'opb_cpuT(s)',
                                           'opb_exG(%)', 'opb_bnpG(%)']]
                new_row += [None for _ in ['cg_objV', 'cg_mipG(%)',
                                           'cg_wallT(h)', 'cg_wallT(s)', 'cg_cpuT(s)'
                                           'cg_exG(%)', 'gh_bnpG(%)']]
                new_row += [None for _ in ['or_objV',
                                           'or_wallT(h)', 'or_wallT(s)', 'or_cpuT(s)',
                                           'or_exG(%)', 'or_bnpG(%)', 'or_cgG']]
                new_row += [None for _ in ['gh_objV',
                                           'gh_wallT(s)', 'gh_cpuT(s)',
                                           'gh_exG(%)', 'gh_bnpG(%)', 'gh_cgG', 'gh_orG']]
            else:
                objV, mipG, wallTs, cpuT = read_result(bnpResF)
                wallTh, mipG = eval(wallTs) / 3600, eval(mipG) * 100
                bnpObjV = float(objV)
                exG = (exObjV - bnpObjV) / exObjV * 100 if type(exObjV) is float else '-'
                new_row += [objV, mipG,
                            wallTh, wallTs, cpuT,
                            exG]
                #
                opbResF = opath.join(res_dpath, '%s-opb.csv' % prefix)
                if opath.exists(opbResF):
                    objV, mipG, wallTs, cpuT = read_result(opbResF)
                    wallTs, cpuT = map(float, [wallTs, cpuT])
                    opbObjV = float(objV)
                    wallTh, mipG = wallTs / 3600, None
                    exG = (exObjV - opbObjV) / exObjV * 100 if type(exObjV) is float else '-'
                    bnpG = (bnpObjV - opbObjV) / bnpObjV * 100
                    new_row += [objV, mipG,
                                wallTh, wallTs, cpuT,
                                exG, bnpG]
                else:
                    new_row += [None for _ in ['opb_objV', 'opb_mipG(%)',
                                               'opb_wallT(h)', 'opb_wallT(s)', 'opb_cpuT(s)',
                                               'opb_exG(%)', 'opb_bnpG(%)']]
                #
                cgResF = opath.join(res_dpath, '%s-cg.csv' % prefix)
                objV, mipG, wallTs, cpuT = read_result(cgResF)
                wallTs, cpuT = map(float, [wallTs, cpuT])
                cgObjV = float(objV)
                bptFile = opath.join(bpt_dpath, '%s-%s.csv' % (prefix, 'bnpTree'))
                with open(bptFile) as r_csvfile:
                    reader = csv.DictReader(r_csvfile)
                    for row in reader:
                        eliWallTime, eliCpuTime, eventType = [row[cn] for cn in
                                                              ['eliWallTime', 'eliCpuTime', 'eventType']]
                        if eventType == 'M':
                            break
                    wallTs += float(eliWallTime)
                    cpuT += float(eliCpuTime)
                wallTh, mipG = wallTs / 3600, eval(mipG) * 100
                exG = (exObjV - cgObjV) / exObjV * 100 if type(exObjV) is float else '-'
                bnpG = (bnpObjV - cgObjV) / bnpObjV * 100
                new_row += [objV, mipG,
                            wallTh, wallTs, cpuT,
                            exG, bnpG]
                #
                orResF = opath.join(res_dpath, '%s-or.csv' % prefix)
                objV, mipG, wallTs, cpuT = read_result(orResF)
                orObjV = float(objV)
                wallTh = eval(wallTs) / 3600
                exG = (exObjV - orObjV) / exObjV * 100 if type(exObjV) is float else '-'
                bnpG = (bnpObjV - orObjV) / bnpObjV * 100
                cgG = (cgObjV - orObjV) / cgObjV * 100
                new_row += [objV,
                            wallTh, wallTs, cpuT,
                            exG, bnpG, cgG]
                #
                ghResF = opath.join(res_dpath, '%s-gh.csv' % prefix)
                objV, mipG, wallTs, cpuT = read_result(ghResF)
                ghObjV = float(objV)
                exG = (exObjV - ghObjV) / exObjV * 100 if type(exObjV) is float else '-'
                bnpG = (bnpObjV - ghObjV) / bnpObjV * 100
                cgG = (cgObjV - ghObjV) / cgObjV * 100
                try:
                    orG = (orObjV - ghObjV) / orObjV * 100
                except ZeroDivisionError:
                    orG = -1
                new_row += [objV,
                            wallTs, cpuT,
                            exG, bnpG, cgG, orG]
            writer.writerow(new_row)


def read_result(resF):
    with open(resF) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            objV, mipG, wallTs, cpuT = [row[cn] for cn in ['objV', 'Gap', 'eliWallTime', 'eliCpuTime']]
    return objV, mipG, wallTs, cpuT

if __name__ == '__main__':
    run_multipleCores(10)
    # summary()
    # gen_problems(opath.join(dpath['experiment'], 'tempProb'))

