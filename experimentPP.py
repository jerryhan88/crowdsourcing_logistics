import os.path as opath
import multiprocessing
import os
from random import seed
import numpy as np
import pandas as pd
import csv, pickle
from functools import reduce
#
from __path_organizer import exp_dpath
from mrtScenario import PER25, PER75, STATIONS
from mrtScenario import gen_instance, inputConvertPickle

HOUR5 = 5 * 60 * 60
HOUR_INF = 1e400
TIME_LIMIT = HOUR_INF


def gen_problems4PP(problem_dpath):
    if not opath.exists(problem_dpath):
        os.mkdir(problem_dpath)
        for dname in ['dplym', 'prmt']:
            os.mkdir(opath.join(problem_dpath, dname))
    #
    min_durPD = 20
    minTB, maxTB = 2, 4
    flowPER, detourPER = PER75, PER25
    #
    #  '4small', '5out', '7inter', '11interOut'
    stationSel = '5out'
    stations = STATIONS[stationSel]
    for seedNum in range(10):
        for numTasks in [
                        # 50,
                        # 100,
                        # 200,
                        # 400,
                        800,
                         ]:
            numBundles = int(numTasks / ((minTB + maxTB) / 2)) + 1
            problemName = '%s-nt%d-mDP%d-mTB%d-dp%d-fp%d-sn%d' % (stationSel, numTasks,
                                                                  min_durPD, maxTB, detourPER, flowPER,
                                                                  seedNum)
            #
            flow_oridest = [(stations[i], stations[j])
                            for i in range(len(stations)) for j in range(len(stations)) if i != j]
            #
            seed(seedNum)
            task_ppdp, tasks, \
            flows, \
            travel_time, \
            thDetour, minWS = gen_instance(flow_oridest, numTasks, min_durPD, detourPER, flowPER)
            problem = [problemName,
                       flows, tasks,
                       numBundles, minTB, maxTB,
                       travel_time, thDetour,
                       minWS]
            inputConvertPickle(problem, flow_oridest, task_ppdp, problem_dpath)


def summaryRD_PP():
    def process_files(fns, wid, wsDict):
        rows = []
        for fn in fns:
            _, prefix = fn[:-len('.pkl')].split('_')
            #
            prmt_fpath = opath.join(prmt_dpath, fn)
            with open(prmt_fpath, 'rb') as fp:
                prmt = pickle.load(fp)
            K, T, cB_M, cB_P, _delta, cW = [prmt.get(k) for k in ['K', 'T', 'cB_M', 'cB_P', '_delta', 'cW']]
            new_row = [prefix, len(K), len(T), cB_M, cB_P, _delta, cW]
            #
            aprc_row = ['-' for _ in range(len(aprcs) * 2)]
            for i, aprc in enumerate(aprcs):
                sol_fpath = opath.join(sol_dpath, 'sol_%s_%s.csv' % (prefix, aprc))
                if opath.exists(sol_fpath):
                    with open(sol_fpath) as r_csvfile:
                        reader = csv.DictReader(r_csvfile)
                        for row in reader:
                            objV, eliCpuTime = [row[cn] for cn in ['objV', 'eliCpuTime']]
                        aprc_row[i] = objV
                        aprc_row[i + len(aprcs)] = eliCpuTime
            new_row += aprc_row
            #
            mip_comT_row = ['-' for _ in range(len(cwls))]
            numCols_row = ['-' for _ in range(len(cwls))]
            for i, aprc in enumerate(cwls):
                sol_fpath = opath.join(sol_dpath, 'sol_%s_%s.csv' % (prefix, aprc))
                log_fpath = opath.join(log_dpath, '%s_itr%s.csv' % (prefix, aprc))
                if opath.exists(sol_fpath):
                    with open(sol_fpath) as r_csvfile:
                        reader = csv.DictReader(r_csvfile)
                        for row in reader:
                            w_eliCpuTime = eval(row['eliCpuTime'])
                    with open(log_fpath) as r_csvfile:
                        reader = csv.DictReader(r_csvfile)
                        for row in reader:
                            pass
                        try:
                            cg_eliCpuTime, numCols = map(eval, [row[cn] for cn in ['eliCpuTime', 'numCols']])
                            mip_comT_row[i] = w_eliCpuTime - cg_eliCpuTime
                            numCols_row[i] = numCols
                        except:
                            pass
            new_row += mip_comT_row
            new_row += numCols_row
            rows.append(new_row)
        wsDict[wid] = rows
    #
    summaryPP_dpath = opath.join(exp_dpath, '_PracticalProblems')
    rd_fpath = reduce(opath.join, [summaryPP_dpath, 'rawDataPP.csv'])
    cwls = ['CWL%d' % cwl_no for cwl_no in range(5, 0, -1)]
    aprcs = ['GH'] + cwls
    with open(rd_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['pn', 'numPaths', 'numTasks', 'minTB', 'maxTB', 'thDetour', 'thWS']
        for aprc in aprcs:
            header += ['%s_objV' % aprc]
        for aprc in aprcs:
            header += ['%s_cpuT' % aprc]
        for aprc in cwls:
            header += ['%s_MIP_T' % aprc]
        for aprc in cwls:
            header += ['%s_numCols' % aprc]
        writer.writerow(header)
    #
    prmt_dpath = reduce(opath.join, [summaryPP_dpath, 'prmt'])
    sol_dpath = reduce(opath.join, [summaryPP_dpath, 'sol'])
    log_dpath = reduce(opath.join, [summaryPP_dpath, 'log'])
    numProcessors = multiprocessing.cpu_count()
    worker_fns = [[] for _ in range(numProcessors)]
    prmt_fns = sorted([fn for fn in os.listdir(prmt_dpath) if fn.endswith('.pkl')])
    for i, fn in enumerate(prmt_fns):
        worker_fns[i % numProcessors].append(fn)
    ps = []
    wsDict = multiprocessing.Manager().dict()
    for wid, fns in enumerate(worker_fns):
        p = multiprocessing.Process(target=process_files,
                                    args=(fns, wid, wsDict))
        ps.append(p)
        p.start()
    for p in ps:
        p.join()
    for _, rows in wsDict.items():
        for row in rows:
            with open(rd_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow(row)
    #
    df = pd.read_csv(rd_fpath)
    df['seedNum'] = df.apply(lambda row: int(row['pn'].split('-')[-1][len('sn'):]), axis=1)
    df = df.sort_values(by=['numPaths', 'numTasks', 'seedNum'])
    df = df.drop(['seedNum'], axis=1)
    df.to_csv(rd_fpath, index=False)


def summaryPP():
    summaryPP_dpath = opath.join(exp_dpath, '_PracticalProblems')
    rd_fpath = reduce(opath.join, [summaryPP_dpath, 'rawDataPP.csv'])
    sum_fpath = reduce(opath.join, [summaryPP_dpath, 'summaryPP.csv'])
    odf = pd.read_csv(rd_fpath)    
    odf = odf.drop(['pn', 'minTB', 'maxTB', 'thDetour', 'thWS'], axis=1)    
    odf = odf.replace('-', np.nan)    
    odf[odf.columns] = odf[odf.columns].apply(pd.to_numeric)    
    df = odf.groupby(['numPaths', 'numTasks']).mean().reset_index()    
    aprcs = ['CWL%d' % cwl_no for cwl_no in range(1, 6)] + ['GH']    
    sdf = odf.groupby(['numPaths', 'numTasks']).std().reset_index()
    for aprc in aprcs:
        df['%s_cpuT_sd' % aprc] = sdf['%s_cpuT' % aprc]
    # for aprc in aprcs:
    #     df['p_%s_cpuT' % aprc] = df.apply(lambda row: '%.f(%.f)' % (row['%s_cpuT' % aprc], row['%s_cpuT_sd' % aprc]), axis=1)
    # for aprc in aprcs:
    #     df['p_%s_objV' % aprc] = df.apply(lambda row: '%.2f' % row['%s_objV' % aprc], axis=1)
    #
    for aprc in aprcs:
        df['%s_objV' % aprc] = np.where(df['%s_cpuT' % aprc] > TIME_LIMIT, np.nan, df['%s_objV' % aprc])
        df['p_%s_objV' % aprc] = df.apply(lambda row: '%.2f' % row['%s_objV' % aprc], axis=1)
        #
        df['%s_cpuT_sd' % aprc] = np.where(df['%s_cpuT' % aprc] > TIME_LIMIT, np.nan, df['%s_cpuT_sd' % aprc])
        df['%s_cpuT' % aprc] = np.where(df['%s_cpuT' % aprc] > TIME_LIMIT, np.nan, df['%s_cpuT' % aprc])
    df.to_csv(sum_fpath, index=False)
    

if __name__ == '__main__':
    # gen_problems4PP(opath.join(exp_dpath, 'm18'))
    # summaryRD_PP()
    summaryPP()