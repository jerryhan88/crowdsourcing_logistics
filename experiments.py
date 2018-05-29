import os.path as opath
import os
import shutil
import pickle
import csv
from random import seed
from functools import reduce
#
from __path_organizer import exp_dpath
from _util_cython import gen_cFile
from mrtScenario import gen_instance, inputConvertPickle
from mrtScenario import PER25, PER50, PER75, STATIONS


def gen_problems(problem_dpath):
    if not opath.exists(problem_dpath):
        os.mkdir(problem_dpath)
        for dname in ['dplym', 'prmts']:
            os.mkdir(opath.join(problem_dpath, dname))
    #

    #  '4small', '5out', '7inter', '11interOut'

    stationSel = '7inter'
    stations = STATIONS[stationSel]
    min_durPD = 20
    minTB, maxTB = 2, 4
    flowPER, detourPER = PER75, PER25

    for numTasks in [
                    # 10,
                    # 50,
                    # 100,
                    # 200,
                    # 400,
                    # 800,
                    1600,
                     ]:
        numBundles = int(numTasks / ((minTB + maxTB) / 2)) + 1
        problemName = '%s-nt%d-mDP%d-mTB%d-dp%d-fp%d' % (stationSel, numTasks, min_durPD, maxTB, detourPER, flowPER)
        #
        seed(0)
        flow_oridest, task_ppdp, \
        flows, tasks, \
        numLocs, travel_time, thDetour, \
        minWS = gen_instance(stations, numTasks, min_durPD, detourPER, flowPER)
        problem = [problemName,
                   flows, tasks,
                   numBundles, minTB, maxTB,
                   numLocs, travel_time, thDetour,
                   minWS]
        inputConvertPickle(problem, flow_oridest, task_ppdp, problem_dpath)


def run_experiments(machine_num):
    from EX1 import run as EX1_run
    from EX2 import run as EX2_run
    from CWL1 import run as CWL1_run
    for prefix in ['CWL2', 'CWL3', 'CWL4', 'CWL5', 'GH']:
        gen_cFile(prefix)
    from CWL2 import run as CWL2_run
    from CWL3 import run as CWL3_run
    from CWL4 import run as CWL4_run
    from CWL5 import run as CWL5_run
    from GH import run as GH_run
    from BNP import run as BNP_run
    cwl_functions = [None, CWL1_run, CWL2_run, CWL3_run, CWL4_run, CWL5_run]
    #
    _TimeLimit = 10 * 60 * 60
    machine_dpath = opath.join(exp_dpath, 'm%d' % machine_num)
    problem_dpath = opath.join(machine_dpath, 'prmts')
    for path in [machine_dpath, problem_dpath]:
        assert opath.exists(path), path
    log_dpath = opath.join(machine_dpath, 'log')
    sol_dpath = opath.join(machine_dpath, 'sol')
    for path in [log_dpath, sol_dpath]:
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
        #
        ###############################################################
        # GH
        # etc = {'solFilePKL': opath.join(sol_dpath, 'sol_%s_GH.pkl' % problemName),
        #        'solFileCSV': opath.join(sol_dpath, 'sol_%s_GH.csv' % problemName),
        #        'solFileTXT': opath.join(sol_dpath, 'sol_%s_GH.txt' % problemName),
        #        'logFile': opath.join(log_dpath, '%s_GH.log' % problemName),
        #        }
        # GH_run(prmt, etc)
        ###############################################################
        #
        ###############################################################
        # CWL
        for cwl_no in range(4, 6):
            etc = {'solFilePKL': opath.join(sol_dpath, 'sol_%s_CWL%d.pkl' % (problemName, cwl_no)),
                   'solFileCSV': opath.join(sol_dpath, 'sol_%s_CWL%d.csv' % (problemName, cwl_no)),
                   'solFileTXT': opath.join(sol_dpath, 'sol_%s_CWL%d.txt' % (problemName, cwl_no)),
                   'logFile': opath.join(log_dpath, '%s_CWL%d.log' % (problemName, cwl_no)),
                   'itrFileCSV': opath.join(log_dpath, '%s_itrCWL%d.csv' % (problemName, cwl_no)),
                   }
            cwl_functions[cwl_no](prmt, etc)
        ###############################################################
        #
        ###############################################################
        # EX2
        # etc = {'solFilePKL': opath.join(sol_dpath, 'sol_%s_EX2.pkl' % problemName),
        #        'solFileCSV': opath.join(sol_dpath, 'sol_%s_EX2.csv' % problemName),
        #        'solFileTXT': opath.join(sol_dpath, 'sol_%s_EX2.txt' % problemName),
        #        'logFile': opath.join(log_dpath, '%s_EX2.log' % problemName),
        #        'itrFileCSV': opath.join(log_dpath, '%s_itrEX2.csv' % problemName)
        #        }
        # EX2_run(prmt, etc)
        ###############################################################
        #
        ###############################################################
        # EX1
        # etc = {'solFilePKL': opath.join(sol_dpath, 'sol_%s_EX1.pkl' % problemName),
        #        'solFileCSV': opath.join(sol_dpath, 'sol_%s_EX1.csv' % problemName),
        #        'solFileTXT': opath.join(sol_dpath, 'sol_%s_EX1.txt' % problemName),
        #        'logFile': opath.join(log_dpath, '%s_EX1.log' % problemName),
        #        'itrFileCSV': opath.join(log_dpath, '%s_itrEX1.csv' % problemName)
        #        }
        # EX1_run(prmt, etc)
        ###############################################################
        #
        ###############################################################
        # BNP
        # etc = {'solFilePKL': opath.join(sol_dpath, 'sol_%s_BNP.pkl' % problemName),
        #        'solFileCSV': opath.join(sol_dpath, 'sol_%s_BNP.csv' % problemName),
        #        'solFileTXT': opath.join(sol_dpath, 'sol_%s_BNP.txt' % problemName),
        #        'logFile': opath.join(log_dpath, '%s_BNP.log' % problemName),
        #        'itrFileCSV': opath.join(log_dpath, '%s_itrBNP.csv' % problemName),
        #        }
        # BNP_run(prmt, etc)
        ###############################################################

        ###############################################################
        # os.remove(ifpath)


def summary1():
    sum_fpath = reduce(opath.join, [exp_dpath, '_summary', 'experiment_summary.csv'])
    aprcs = ['GH'] + ['CWL%d' % cwl_no for cwl_no in range(5, 0, -1)]    
    with open(sum_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['pn', 'numPaths', 'numTasks', 'minTB', 'maxTB', 'thDetour', 'thWS']
        for aprc in aprcs:
            header += ['%s_objV' % aprc, '%s_cpuT' % aprc]
        writer.writerow(header)    
    #
    prmts_dpath = reduce(opath.join, [exp_dpath, '_summary', 'prmts'])
    sol_dpath = reduce(opath.join, [exp_dpath, '_summary', 'sol'])    
    log_dpath = reduce(opath.join, [exp_dpath, '_summary', 'log'])    
    for fn in os.listdir(prmts_dpath):        
        if not fn.endswith('.pkl'): continue    
        _, prefix = fn[:-len('.pkl')].split('_')
        #
        prmts_fpath = opath.join(prmts_dpath, fn)       
        with open(prmts_fpath, 'rb') as fp:
            prmt = pickle.load(fp)        
        K, T, cB_M, cB_P, _delta, cW = [prmt.get(k) for k in ['K', 'T', 'cB_M', 'cB_P', '_delta', 'cW']]
        new_row = [prefix, len(K), len(T), cB_M, cB_P, _delta, cW]
        #                
        for aprc in aprcs:
            sol_fpath = opath.join(sol_dpath, 'sol_%s_%s.csv' % (prefix, aprc))
            log_fpath = opath.join(log_dpath, '%s_itr%s.csv' % (prefix, aprc))
            if opath.exists(sol_fpath):
                with open(sol_fpath) as r_csvfile:
                    reader = csv.DictReader(r_csvfile)
                    for row in reader:                
                        objV, eliCpuTime = [row [cn] for cn in ['objV', 'eliCpuTime']]
                    new_row += [objV, eliCpuTime]  
            elif opath.exists(log_fpath):
                with open(log_fpath) as r_csvfile:
                    reader = csv.DictReader(r_csvfile)
                    for row in reader:                
                        pass
                    relObjV, eliCpuTime = [row[cn] for cn in ['relObjV', 'eliCpuTime']]
                new_row += ['[%s]' % relObjV, '[%s]' % eliCpuTime]
            else:                
                new_row += ['-', '-']
        #
        with open(sum_fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow(new_row)


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
    # gen_problems(opath.join(exp_dpath, 'm114'))
    # run_experiments(101)
    summary1()
    # gen_mrtProblems(opath.join(dpath['experiment'], 'tempProb'))
    # summary()