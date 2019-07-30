import os.path as opath
import os
import csv, pickle
import pandas as pd
from functools import reduce
from itertools import chain
#
from __path_organizer import exp_dpath

#
# Post analysis
#


def summaryPA():
    summaryPA_dpath = opath.join(exp_dpath, '_summaryPA')
    rd_fpath = reduce(opath.join, [summaryPA_dpath, 'rawDataPA_MCS.csv'])
    with open(rd_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['pn', 'numTasks', 'allowTT',
                  'cpuT',
                  'NHT', 'Ratio', 'XHNT', 'NB',
                  'Revenue', 'Cost', 'Profit']
        writer.writerow(header)
    prmt_dpath = reduce(opath.join, [summaryPA_dpath, 'prmt'])
    sol_dpath = reduce(opath.join, [summaryPA_dpath, 'sol'])
    aprc = 'CWL4'
    cP = 12
    bundleReward = {
        2: 15,
        3: 20,
        4: 25
    }
    for fn in os.listdir(prmt_dpath):
        if not fn.endswith('.pkl'): continue
        _, prefix = fn[:-len('.pkl')].split('_')
        prmt_fpath = opath.join(prmt_dpath, fn)
        with open(prmt_fpath, 'rb') as fp:
            prmt = pickle.load(fp)
        T, _delta = [prmt.get(k) for k in ['T', '_delta']]
        new_row = [prefix, len(T), _delta]
        #
        sol_fpath = opath.join(sol_dpath, 'sol_%s_%s.csv' % (prefix, aprc))
        if not opath.exists(sol_fpath):
            new_row += ['-',
                        '-', '-', '-', '-',
                        '-', '-', '-']
        else:
            with open(sol_fpath) as r_csvfile:
                reader = csv.DictReader(r_csvfile)
                for row in reader:
                    objV, eliCpuTime = [eval(row[cn]) for cn in ['objV', 'eliCpuTime']]
            with open(opath.join(sol_dpath, 'sol_%s_%s.pkl' % (prefix, aprc)), 'rb') as fp:
                sol = pickle.load(fp)
            C, q_c = [sol.get(k) for k in ['C', 'q_c']]
            bundles = [C[c] for c in range(len(C)) if q_c[c] > 0.5]
            nb = len(bundles)
            NHT = len(list(chain(*bundles)))
            XHNT = (len(T) - NHT)
            Revenue = NHT * cP
            Cost = 0
            for bc in bundles:
                Cost += bundleReward[len(bc)]
            Profit = Revenue - Cost
            new_row += [eliCpuTime,
                        NHT, NHT / float(len(T)), XHNT, nb,
                        Revenue, Cost, Profit]
        with open(rd_fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow(new_row)
    df = pd.read_csv(rd_fpath)
    df['seedNum'] = df.apply(lambda row: int(row['pn'].split('-')[-1][len('sn'):]), axis=1)
    df = df.sort_values(by=['numTasks', 'seedNum'])
    df = df.drop(['seedNum'], axis=1)
    df.to_csv(rd_fpath, index=False)
    #
    sum_fpath = reduce(opath.join, [summaryPA_dpath, 'summaryPA_MCS.csv'])
    # if not df[(df['cpuT'] == '-')].empty:
    #     df = df[(df['cpuT'] != '-')]
    df = df.drop(['pn'], axis=1)
    cols = list(df.columns[1:])
    df[cols] = df[cols].apply(pd.to_numeric)
    df = df.groupby(['numTasks']).mean().reset_index()
    df.to_csv(sum_fpath, index=False)

