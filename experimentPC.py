import os.path as opath
import os
from random import seed
import numpy as np
import pandas as pd
import csv, pickle
from functools import reduce
#
from __path_organizer import exp_dpath
from mrtScenario import gen_instance, inputConvertPickle
from mrtScenario import PER50, PER75


def gen_problems4PC(problem_dpath):
    if not opath.exists(problem_dpath):
        os.mkdir(problem_dpath)
        for dname in ['dplym', 'prmt']:
            os.mkdir(opath.join(problem_dpath, dname))
    min_durPD = 20
    minTB, maxTB = 2, 4
    flowPER, detourPER = PER75, PER50
    flow_oridest_W = [
        ('Tanjong Pagar', 'Tampines'),
        ('Raffles Place', 'Bishan'),
        ('Raffles Place', 'Ang Mo Kio'),
        ('Jurong East', 'Yew Tee'),
        ('Tanjong Pagar', 'Boon Lay'),
    ]
    seedNum = 0
    numTasks = 10

    thDetour = 65
    for i in range(len(flow_oridest_W)):
        flow_oridest = flow_oridest_W[:i + 1]
        if i != 4:
            continue
        for seedNum in range(2, 20):
            for numTasks in np.arange(15, 16, 5):
                # if i > 2 and numTasks > 20:
                #     continue
                numBundles = int(numTasks / ((minTB + maxTB) / 2)) + 1
                problemName = '%s-nt%d-mDP%d-mTB%d-dt%d-fp%d-sn%d' % ('bR%d' % len(flow_oridest), numTasks,
                                                                      min_durPD, maxTB, thDetour, flowPER,
                                                                      seedNum)
                #
                seed(seedNum)
                task_ppdp, tasks, \
                flows, \
                travel_time, \
                _, minWS = gen_instance(flow_oridest, numTasks, min_durPD, detourPER, flowPER)

                thDetour = 65

                problem = [problemName,
                           flows, tasks,
                           numBundles, minTB, maxTB,
                           travel_time, thDetour,
                           minWS]
                inputConvertPickle(problem, flow_oridest, task_ppdp, problem_dpath)


def summaryRD():
    summaryPC_dpath = opath.join(exp_dpath, '_PerformanceComparision')
    rd_fpath = reduce(opath.join, [summaryPC_dpath, 'rawDataPC.csv'])
    aprcs = ['EX2', 'BNP'] + ['GH'] + ['CWL1']
    with open(rd_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['pn', 'numPaths', 'numTasks', 'minTB', 'maxTB', 'thDetour', 'thWS']
        for aprc in aprcs:
            header += ['%s_objV' % aprc]
        for aprc in aprcs:
            header += ['%s_cpuT' % aprc]
        header += ['EX2_DG']
        writer.writerow(header)
    #
    prmt_dpath = reduce(opath.join, [summaryPC_dpath, 'prmt'])
    sol_dpath = reduce(opath.join, [summaryPC_dpath, 'sol'])
    log_dpath = reduce(opath.join, [summaryPC_dpath, 'log'])
    for fn in os.listdir(prmt_dpath):
        if not fn.endswith('.pkl'): continue
        _, prefix = fn[:-len('.pkl')].split('_')
        #
        prmt_fpath = opath.join(prmt_dpath, fn)
        with open(prmt_fpath, 'rb') as fp:
            prmt = pickle.load(fp)
        K, T, cB_M, cB_P, _delta, cW = [prmt.get(k) for k in ['K', 'T', 'cB_M', 'cB_P', '_delta', 'cW']]
        new_row = [prefix, len(K), len(T), cB_M, cB_P, _delta, cW]
        aprc_row = ['-' for _ in range(len(aprcs) * 2)]
        for i, aprc in enumerate(aprcs):
            sol_fpath = opath.join(sol_dpath, 'sol_%s_%s.csv' % (prefix, aprc))
            log_fpath = opath.join(log_dpath, '%s_itr%s.csv' % (prefix, aprc))
            if opath.exists(sol_fpath):
                with open(sol_fpath) as r_csvfile:
                    reader = csv.DictReader(r_csvfile)
                    for row in reader:
                        objV, eliCpuTime = [row[cn] for cn in ['objV', 'eliCpuTime']]
                        gap = eval(row['Gap'])
                    # if aprc != 'GH':
                    #     aprc_row[i] = objV if gap == 0 else '[%s]' % objV
                    #     aprc_row[i + len(aprcs)] = eliCpuTime if gap == 0 else '[%s]' % eliCpuTime
                    # else:
                    aprc_row[i] = objV
                    aprc_row[i + len(aprcs)] = eliCpuTime
            elif opath.exists(log_fpath):
                if aprc == 'BNP':
                    nid_obj = {}
                    bestObj, totalTime = -1.0, 0.0
                    with open(log_fpath) as r_csvfile:
                        reader = csv.DictReader(r_csvfile)
                        for row in reader:
                            nid, eliCpuTime, objVal, eventType = [row[cn] for cn in ['nid', 'eliCpuTime', 'objVal', 'eventType']]
                            if eventType == 'M' and objVal:
                                nid_obj[nid] = eval(objVal)
                            elif eventType == 'INT':
                                if nid_obj[nid] > bestObj:
                                    bestObj = nid_obj[nid]
                            if eliCpuTime:
                                totalTime += eval(eliCpuTime)
                    aprc_row[i] = '[%s]' % bestObj
                    aprc_row[i + len(aprcs)] = '[%s]' % totalTime
                elif aprc == 'EX2':
                    with open(log_fpath) as r_csvfile:
                        reader = csv.DictReader(r_csvfile)
                        for row in reader:
                            pass
                        objbst, eliCpuTime = [row[cn] for cn in ['objbst', 'eliCpuTime']]
                    aprc_row[i] = objbst
                    aprc_row[i + len(aprcs)] = eliCpuTime
                else:
                    with open(log_fpath) as r_csvfile:
                        reader = csv.DictReader(r_csvfile)
                        for row in reader:
                            pass
                        relObjV, eliCpuTime = [row[cn] for cn in ['relObjV', 'eliCpuTime']]
                    aprc_row[i] = '[%s]' % relObjV
                    aprc_row[i + len(aprcs)] = '[%s]' % eliCpuTime
        new_row += aprc_row
        exSol_fpath = opath.join(sol_dpath, 'sol_%s_%s.csv' % (prefix, 'EX2'))
        if opath.exists(exSol_fpath):
            with open(exSol_fpath) as r_csvfile:
                reader = csv.DictReader(r_csvfile)
                for row in reader:
                    gap = eval(row['Gap'])
            new_row += [gap]
        else:
            new_row += ['-']
        with open(rd_fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow(new_row)
    #
    df = pd.read_csv(rd_fpath)
    df['seedNum'] = df.apply(lambda row: int(row['pn'].split('-')[-1][len('sn'):]), axis=1)
    df = df.sort_values(by=['numPaths', 'numTasks', 'seedNum'])
    df = df.drop(['seedNum'], axis=1)
    df.to_csv(rd_fpath, index=False)


def summaryPC():
    summaryPC_dpath = opath.join(exp_dpath, '_PerformanceComparision')
    rd_fpath = reduce(opath.join, [summaryPC_dpath, 'rawDataPC.csv'])
    sum_fpath = reduce(opath.join, [summaryPC_dpath, 'summaryPC.csv'])
    odf = pd.read_csv(rd_fpath)
    #
    odf = odf[(odf['CWL1_objV'] != '-')]
    odf = odf[(odf['EX2_objV'] != '-')]
    odf = odf[(odf['EX2_DG'] != '-')]
    odf = odf.replace('-', np.nan)
    odf = odf.drop(['pn', 'minTB', 'maxTB', 'thDetour', 'thWS'], axis=1)
    odf[odf.columns] = odf[odf.columns].apply(pd.to_numeric)
    #
    df = odf.groupby(['numPaths', 'numTasks']).mean().reset_index()    
    objCols = ['EX2_objV', 'BNP_objV', 'GH_objV', 'CWL1_objV']
    comCols = ['EX2_cpuT', 'BNP_cpuT', 'GH_cpuT', 'CWL1_cpuT']
    decimals = pd.Series([2, 2, 2, 2], index=objCols)
    df = df.round(decimals)
    decimals = pd.Series([0, 0, 0, 0], index=comCols)
    df = df.round(decimals)    
    # df[comCols] = df[comCols].astype(int)
    df['EX2_DG'] = df['EX2_DG'] * 100
    df = df.round(pd.Series([0], index=['EX2_DG']))
    # df['EX2_DG'] = df['EX2_DG'].astype(int)
    #
    sdf = odf.groupby(['numPaths', 'numTasks']).std().reset_index()
    for cn in comCols:
        df['%s_sd' % cn] = sdf[cn]
    #
    df['EX2_sol'] = df.apply(lambda row: '%.2f(%d%%)' % (row['EX2_objV'], int(row['EX2_DG'])), axis=1)
    df['BNP_sol'] = df.apply(lambda row: '%.2f' % row['BNP_objV'], axis=1)
    df['CWL_sol'] = df.apply(lambda row: '%.f%%' % (100 * (row['BNP_objV'] - row['CWL1_objV']) / row['BNP_objV']), axis=1)
    df['GH_sol'] = df.apply(lambda row: '%.f%%' % (100 * (row['BNP_objV'] - row['GH_objV']) / row['BNP_objV']), axis=1)
    #
    df.to_csv(sum_fpath, index=False)


if __name__ == '__main__':
    # gen_problems4PC(opath.join(exp_dpath, 'm1001'))
    # summaryRD()
    summaryPC()