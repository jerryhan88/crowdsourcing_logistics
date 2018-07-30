import os.path as opath
import multiprocessing
import os
import csv, pickle
from itertools import chain
import pandas as pd
import numpy as np
from functools import reduce
#
from __path_organizer import exp_dpath

MIN60, SEC60 = 60.0, 60.0
MIN20 = 20
Meter1000 = 1000.0
WALKING_SPEED = 5.0  # km/hour


cP = 12
bundleReward = {
    2: 15,
    3: 20,
    4: 25
}


def summaryRD():
    def process_files(fns, wid, wsDict):
        rows = []
        for fn in fns:
            _, prefix = fn[:-len('.pkl')].split('_')
            prmt_fpath = opath.join(prmt_dpath, fn)
            with open(prmt_fpath, 'rb') as fp:
                prmt = pickle.load(fp)
            d2d_ratio = float(fn[:-len('.pkl')].split('-')[-3][len('d2dR'):]) / 100
            K, T, _delta, cW = [prmt.get(k) for k in ['K', 'T', '_delta', 'cW']]
            new_row = [prefix, len(K), len(T), d2d_ratio, _delta, cW]
            aprc = 'CWL4'
            sol_fpath = opath.join(sol_dpath, 'sol_%s_%s.csv' % (prefix, aprc))
            if opath.exists(sol_fpath):
                with open(sol_fpath) as r_csvfile:
                    reader = csv.DictReader(r_csvfile)
                    for row in reader:
                        objV, eliCpuTime = [row[cn] for cn in ['objV', 'eliCpuTime']]
                    new_row += [objV, eliCpuTime]
                with open(opath.join(sol_dpath, 'sol_%s_%s.pkl' % (prefix, aprc)), 'rb') as fp:
                    sol = pickle.load(fp)
                C, q_c = [sol.get(k) for k in ['C', 'q_c']]
                bundles = [C[c] for c in range(len(C)) if q_c[c] > 0.5]
                nbt = len(list(chain(*bundles)))
                nst = (len(T) - nbt)
                Revenue = nbt * cP
                Cost = 0
                for bc in bundles:
                    Cost += bundleReward[len(bc)]
                Profit = Revenue - Cost
                new_row += [nbt, nst, float(nbt) / len(T),
                            Revenue, Cost, Profit]
            else:
                new_row += ['-', '-']
                new_row += ['-', '-', '-']
                new_row += ['-', '-', '-']
            rows.append(new_row)
        wsDict[wid] = rows
    #
    summaryTTDF_dpath = opath.join(exp_dpath, '_TaskType_comDF')
    #
    rd_fpath = reduce(opath.join, [summaryTTDF_dpath, 'rawDataTT.csv'])
    with open(rd_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['pn', 'numPaths', 'numTasks', 'd2d_ratio', 'thDetour', 'thWS',
                  'objV', 'cpuT',
                  'nbt', 'nst', 'Fraction',
                  'Revenue', 'Cost', 'Profit']
        writer.writerow(header)
    #
    prmt_dpath = reduce(opath.join, [summaryTTDF_dpath, 'prmt'])
    sol_dpath = reduce(opath.join, [summaryTTDF_dpath, 'sol'])
    #
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
    df = df.sort_values(by=['d2d_ratio', 'seedNum'])
    df = df.drop(['seedNum'], axis=1)
    df.to_csv(rd_fpath, index=False)


def summaryTT():
    summaryTT_dpath = opath.join(exp_dpath, '_TaskType_comDF')
    rd_fpath = reduce(opath.join, [summaryTT_dpath, 'rawDataTT.csv'])
    sum_fpath = reduce(opath.join, [summaryTT_dpath, 'summaryTT.csv'])
    odf = pd.read_csv(rd_fpath)
    odf = odf.drop(['pn', 'thWS'], axis=1)
    odf = odf.replace('-', np.nan)
    odf[odf.columns] = odf[odf.columns].apply(pd.to_numeric)
    df = odf.groupby(['d2d_ratio', 'thDetour']).mean().reset_index()
    # aprcs = ['CWL%d' % cwl_no for cwl_no in range(1, 6)] + ['GH']
    # sdf = odf.groupby(['numPaths', 'numTasks']).std().reset_index()
    # for aprc in aprcs:
    #     df['%s_cpuT_sd' % aprc] = sdf['%s_cpuT' % aprc]

    df.to_csv(sum_fpath, index=False)




if __name__ == '__main__':
    # summaryRD()
    summaryTT()