import os.path as opath
import os
import shutil
import pickle
from random import seed
from functools import reduce

#
from __path_organizer import exp_dpath


def gen_prmts(senAly_dpath):
    from mrtScenario import induce_otherInputs, convert_prob2prmt
    #
    assert opath.exists(senAly_dpath)
    dplym_dpath = opath.join(senAly_dpath, 'dplym')
    assert opath.exists(dplym_dpath)
    prmt_dpath = opath.join(senAly_dpath, 'prmt')
    if opath.exists(prmt_dpath):
        shutil.rmtree(prmt_dpath)
    os.mkdir(prmt_dpath)
    minTB = 2
    for fn in os.listdir(dplym_dpath):
        if not fn.endswith('.pkl'):
            continue
        _, problemName = fn[:-len('.pkl')].split('_')
        _, _nt, _, _mTB, _dp, _fp, _ = problemName.split('-')
        maxTB = int(_mTB[len('mTB'):])
        numTasks, detourPER, flowPER = list(map(int, [s[len('nt'):] for s in [_nt, _dp, _fp]]))
        numBundles = int(numTasks / ((minTB + maxTB) / 2)) + 1
        with open(opath.join(dplym_dpath, fn), 'rb') as fp:
            flow_oridest, task_ppdp = pickle.load(fp)
        flows, tasks, travel_time, thDetour, minWS = induce_otherInputs(flow_oridest, task_ppdp, detourPER, flowPER)
        problem = [problemName,
                   flows, tasks,
                   numBundles, minTB, maxTB,
                   travel_time, thDetour,
                   minWS]
        prmt = convert_prob2prmt(*problem)
        with open(reduce(opath.join, [prmt_dpath, 'prmt_%s.pkl' % problemName]), 'wb') as fp:
            pickle.dump(prmt, fp)


def run_experiments(machine_num):
    from EX1 import run as EX1_run
    from EX2 import run as EX2_run
    from CWL1 import run as CWL1_run
    from _util_cython import gen_cFile
    for prefix in ['CWL2', 'CWL3', 'CWL4', 'CWL5', 'GH']:
        gen_cFile(prefix)
    from CWL2 import run as CWL2_run
    from CWL3 import run as CWL3_run
    from CWL4 import run as CWL4_run
    from CWL5 import run as CWL5_run
    from GH import run as GH_run
    # from BNP import run as BNP_run
    cwl_functions = [None, CWL1_run, CWL2_run, CWL3_run, CWL4_run, CWL5_run]
    #
    _TimeLimit = 20 * 60 * 60
    machine_dpath = opath.join(exp_dpath, 'm%d' % machine_num)
    prmt_dpath = opath.join(machine_dpath, 'prmt')
    for path in [machine_dpath, prmt_dpath]:
        assert opath.exists(path), path
    log_dpath = opath.join(machine_dpath, 'log')
    sol_dpath = opath.join(machine_dpath, 'sol')
    for path in [log_dpath, sol_dpath]:
        if opath.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    problems_fpaths = [opath.join(prmt_dpath, fn) for fn in os.listdir(prmt_dpath)
                         if fn.endswith('.pkl')]
    problems_fpaths.sort()
    for fpath in problems_fpaths:
        with open(fpath, 'rb') as fp:
            prmt = pickle.load(fp)
        problemName = prmt['problemName']
        #
        ###############################################################
        # BNP
        # etc = {'solFilePKL': opath.join(sol_dpath, 'sol_%s_BNP.pkl' % problemName),
        #        'solFileCSV': opath.join(sol_dpath, 'sol_%s_BNP.csv' % problemName),
        #        'solFileTXT': opath.join(sol_dpath, 'sol_%s_BNP.txt' % problemName),
        #        'logFile': opath.join(log_dpath, '%s_BNP.log' % problemName),
        #        'itrFileCSV': opath.join(log_dpath, '%s_itrBNP.csv' % problemName),
        #        #
        #        'TimeLimit': _TimeLimit
        #        }
        # try:
        #     BNP_run(prmt, etc)
        # except:
        #     os.remove(fpath)
        #     continue
        ###############################################################
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
        for cwl_no in range(4, 5):
            etc = {'solFilePKL': opath.join(sol_dpath, 'sol_%s_CWL%d.pkl' % (problemName, cwl_no)),
                   'solFileCSV': opath.join(sol_dpath, 'sol_%s_CWL%d.csv' % (problemName, cwl_no)),
                   'solFileTXT': opath.join(sol_dpath, 'sol_%s_CWL%d.txt' % (problemName, cwl_no)),
                   'logFile': opath.join(log_dpath, '%s_CWL%d.log' % (problemName, cwl_no)),
                   'itrFileCSV': opath.join(log_dpath, '%s_itrCWL%d.csv' % (problemName, cwl_no)),
                   }
            cwl_functions[cwl_no](prmt, etc)
            # CWL1_run(prmt, etc)
        ###############################################################
        #
        ###############################################################
        # EX2
        # etc = {'solFilePKL': opath.join(sol_dpath, 'sol_%s_EX2.pkl' % problemName),
        #        'solFileCSV': opath.join(sol_dpath, 'sol_%s_EX2.csv' % problemName),
        #        'solFileTXT': opath.join(sol_dpath, 'sol_%s_EX2.txt' % problemName),
        #        'logFile': opath.join(log_dpath, '%s_EX2.log' % problemName),
        #        'itrFileCSV': opath.join(log_dpath, '%s_itrEX2.csv' % problemName),
        #        #
        #        'TimeLimit': _TimeLimit
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


if __name__ == '__main__':
    run_experiments(10000)
    # pass
    #
    # gen_prmts(opath.join(exp_dpath, 'm0'))
    # summaryPA()
