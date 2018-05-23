import os.path as opath
import os
import shutil
import pickle
import csv
from random import choice, randrange
import multiprocessing

#
from __path_organizer import exp_dpath
from problems import randomProb_EuclDist
from EX1 import run as EX1_run
from EX2 import run as EX2_run
# from BNP import run as BNP_run
# from CWL import run as CWL_run
# from GH import run as GH_run


def gen_problems(problem_dpath):
    #
    # Generate problems
    #
    if not opath.exists(problem_dpath):
        os.mkdir(problem_dpath)
    #
    thVolume, thDetour = 3, 0.8
    for numPaths in [5, 10]:
        for numTasks, numBundles in [
                    (6, 2),

            (8, 3), (10, 4),
        ]:
            randomProb_EuclDist(problem_dpath, numTasks, numBundles, thVolume, numPaths, thDetour)


def run_experiments(machine_num):
    _TimeLimit = 10 * 60 * 60
    machine_dpath = opath.join(exp_dpath, 'm%d' % machine_num)
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
            prmt = pickle.load(fp)
        problemName = prmt['problemName']
        ###############################################################
        # EX1
        etc = {'solFilePKL': opath.join(res_dpath, 'sol_%s_EX1.pkl' % problemName),
               'solFileCSV': opath.join(res_dpath, 'sol_%s_EX1.csv' % problemName),
               'solFileTXT': opath.join(res_dpath, 'sol_%s_EX1.txt' % problemName),
               'logFile': opath.join(log_dpath, '%s_EX1.log' % problemName)}
        EX1_run(prmt, etc)
        ###############################################################
        # EX2
        # etc = {'solFilePKL': opath.join(res_dpath, 'sol_%s_EX2.pkl' % problemName),
        #        'solFileCSV': opath.join(res_dpath, 'sol_%s_EX2.csv' % problemName),
        #        'solFileTXT': opath.join(res_dpath, 'sol_%s_EX2.txt' % problemName),
        #        'logFile': opath.join(log_dpath, '%s_EX2.log' % problemName)}
        # EX2_run(prmt, etc)
        ###############################################################
        #
        ###############################################################
        # BNP
        # problemName = problem[0]
        # log_fpath = opath.join(log_dpath, '%s-BNP.log' % problemName)
        # res_fpath = opath.join(res_dpath, '%s-BNP.csv' % problemName)
        # bpt_fpath = opath.join(bbt_dpath, '%s-bnpTree.csv' % problemName)
        # etcSetting = {'LogFile': log_fpath,
        #               'ResFile': res_fpath,
        #               'bptFile': bpt_fpath,
        #               'TimeLimit': _TimeLimit}
        # grbSetting = {'LogFile': log_fpath,
        #               'Threads': _numThreads}
        # BNP_run(problem, etcSetting, grbSetting)
        ###############################################################
        #
        ###############################################################
        # CWL
        # problemName = problem[0]
        # log_fpath = opath.join(log_dpath, '%s-CWL.log' % problemName)
        # res_fpath = opath.join(res_dpath, '%s-CWL.csv' % problemName)
        # itr_fpath = opath.join(itr_dpath, '%s-itrCWL.csv' % problemName)
        # etcSetting = {'LogFile': log_fpath,
        #               'ResFile': res_fpath,
        #               'itrFile': itr_fpath,
        #               'TimeLimit': _TimeLimit}
        # grbSetting = {'LogFile': log_fpath,
        #               'Threads': _numThreads}
        # CWL_run(problem, etcSetting, grbSetting)
        ###############################################################
        #
        ###############################################################
        # GH
        # problemName = problem[0]
        # log_fpath = opath.join(log_dpath, '%s-GH.log' % problemName)
        # res_fpath = opath.join(res_dpath, '%s-GH.csv' % problemName)
        # etcSetting = {'LogFile': log_fpath,
        #               'ResFile': res_fpath}
        # GH_run(problem, etcSetting)
        ###############################################################
        # os.remove(ifpath)


def summary():
    sum_fpath = opath.join(dpath['experiment'], 'experiment_summary.csv')
    with open(sum_fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['ts', 'numTasks', 'numBundles', 'numPaths', 'thVolume', 'thDetour', 'avgNTinB',
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
        print(fn)
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
                exG = (exObjV - bnpObjV) / exObjV * 100 if (type(exObjV) is float and exObjV != 0.0) else '-'
            else:
                exG = None
            new_row += [bnpObjV, mipG, cpuTh, cpuTs, exG]
            #
            cwlResF = opath.join(res_dpath, '%s-cwl.csv' % prefix)
            cwlLogF = opath.join(log_dpath, '%s-cwl.log' % prefix)
            cwlObjV, mipG, cpuTh, cpuTs = read_result(cwlResF, cwlLogF)
            if type(cwlObjV) == float:
                exG = (exObjV - cwlObjV) / exObjV * 100 if (type(exObjV) is float and exObjV != 0.0) else '-'
                bnpG = (bnpObjV - cwlObjV) / bnpObjV * 100 if (type(bnpObjV) is float and bnpObjV != 0.0) else '-'
            else:
                exG, bnpG = None, None
            new_row += [cwlObjV, cpuTh, cpuTs, exG, bnpG]
            #
            ghResF = opath.join(res_dpath, '%s-gh.csv' % prefix)
            ghLogF = opath.join(log_dpath, '%s-gh.log' % prefix)
            ghObjV, mipG, cpuTh, cpuTs = read_result(ghResF, ghLogF)
            if type(ghObjV) == float:
                exG = (exObjV - ghObjV) / exObjV * 100 if (type(exObjV) is float and exObjV != 0.0) else '-'
                bnpG = (bnpObjV - ghObjV) / bnpObjV * 100 if (type(bnpObjV) is float and bnpObjV != 0.0) else '-'
                cwlG = (cwlObjV - ghObjV) / cwlObjV * 100 if (type(cwlObjV) is float and cwlObjV != 0.0) else '-'
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



if __name__ == '__main__':
    gen_problems(opath.join(exp_dpath, 'tempProb'))
    run_experiments(101)
    # gen_mrtProblems(opath.join(dpath['experiment'], 'tempProb'))
    # summary()