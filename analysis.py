import os.path as opath
import os
import csv, pickle
import pandas as pd
from scipy import stats
from functools import reduce
#
from __path_organizer import exp_dpath
from mrtScenario import get_travelTimeSG
from PD_IH import calc_detourTime


def diff_vc():
    # run experiments on different volume capacity
    anl_dpath = opath.join(exp_dpath, '__analysis')
    prefix = '11interOut-nt200-mDP20-mTB4-dp25-fp75-sn0'
    prmt_fpath = opath.join(anl_dpath, 'prmt_%s.pkl' % prefix)
    with open(prmt_fpath, 'rb') as fp:
        prmt = pickle.load(fp)
    for cB_M, cB_P in [(2,5)]:
        new_prefix = prefix[:].replace('-mDP20-mTB4-dp25-fp75', '-minT%d-maxT%d' % (cB_M, cB_P))
        prmt['cB_M'] = cB_M
        prmt['cB_P'] = cB_P
        prmt['problemName'] = new_prefix
        with open(reduce(opath.join, [anl_dpath, 'prmt_%s.pkl' % new_prefix]), 'wb') as fp:
            pickle.dump(prmt, fp)


def dataProcessing():
    anl_dpath = opath.join(exp_dpath, '__analysis')
    argmt_fpath = opath.join(anl_dpath, '_arrangement.csv')
    with open(argmt_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        writer.writerow(['pn',
                         'tid',
                         'pp_nMRT', 'dp_nMRT', 'detourTime', 'sW',
                         'assign'])

    MRTs_tt, locPD_MRT_tt = get_travelTimeSG()
    for fn in os.listdir(anl_dpath):
        if not fn.endswith('.pkl') or not fn.startswith('sol'):
            continue
        _, prefix, _ = fn[:-len('.pkl')].split('_')
        sol_fpath = opath.join(anl_dpath, fn)
        dplym_fpath = opath.join(anl_dpath, 'dplym_%s.pkl' % prefix)
        prmt_fpath = opath.join(anl_dpath, 'prmt_%s.pkl' % prefix)
        with open(sol_fpath, 'rb') as fp:
            sol = pickle.load(fp)
        with open(dplym_fpath, 'rb') as fp:
            flow_oridest, task_ppdp = pickle.load(fp)
        with open(prmt_fpath, 'rb') as fp:
            prmt = pickle.load(fp)
        T, K, w_k, t_ij, _delta = [prmt.get(k) for k in ['T', 'K', 'w_k', 't_ij', '_delta']]
        tb_assignment = [False for _ in range(len(T))]
        C, q_c = [sol.get(k) for k in ['C', 'q_c']]
        selBundles = [C[c] for c in range(len(C)) if q_c[c] > 0.5]
        for bc in selBundles:
            for i in bc:
                tb_assignment[i] = True
        task_nMRT_dt_sW = []
        for i in T:
            pp, dp = task_ppdp[i]
            pp_tt, pp_nMRT = locPD_MRT_tt[pp]
            dp_tt, dp_nMRT = locPD_MRT_tt[dp]
            iP, iM = 'p%d' % i, 'd%d' % i
            sW = 0.0
            for k in K:
                kP, kM = 'ori%d' % k, 'dest%d' % k
                seq = [kP, iP, iM, kM]
                dt = calc_detourTime(seq, t_ij)
                if dt <= _delta:
                    sW += w_k[k]
            task_nMRT_dt_sW.append([])
            with open(argmt_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([prefix,
                                 i,
                                 pp_nMRT, dp_nMRT, pp_tt + dp_tt, sW,
                                 1 if tb_assignment[i] else 0])


def summary():
    anl_dpath = opath.join(exp_dpath, '__analysis')
    argmt_fpath = opath.join(anl_dpath, '_arrangement.csv')
    if not opath.exists(argmt_fpath):
        dataProcessing()
    sum_fpath = opath.join(anl_dpath, '_summary.csv')
    with open(sum_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        writer.writerow(['pn', 'Measure', 'xbt', 'bt', 'tScore', 'pValue'])
    #
    odf = pd.read_csv(argmt_fpath)

    # for pn in set(odf['pn']):
    for pn in ['11interOut-nt200-mDP20-mTB4-dp25-fp75-sn0',
               '11interOut-nt800-mDP20-mTB4-dp25-fp75-sn0']:

        df = odf[(odf['pn'] == pn)]
        df0 = df[(df['assign']==0)]
        df1 = df[(df['assign']!=0)]
        with open(sum_fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            #
            tScore, pValue = stats.ttest_ind(df0['detourTime'], df1['detourTime'], equal_var=False)
            writer.writerow([pn, 'detourTime', df0['detourTime'].mean(), df1['detourTime'].mean(), tScore, pValue])
            #
            tScore, pValue = stats.ttest_ind(df0['sW'], df1['sW'], equal_var=False)
            writer.writerow([pn, 'sW', df0['sW'].mean(), df1['sW'].mean(), tScore, pValue])

            writer.writerow([pn, len(df0), len(df1)])



def num_tasksInBundles():
    anl_dpath = opath.join(exp_dpath, '__analysis')
    argmt_fpath = opath.join(anl_dpath, '_num_tasksInBundles.csv')
    with open(argmt_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        writer.writerow(['pn',
                         'minTB', 'maxTB',
                         'B2', 'B3', 'B4', 'B5', 'BTT', 'STT'])
    for fn in os.listdir(anl_dpath):
        if not fn.endswith('.pkl') or not fn.startswith('sol'):
            continue
        _, prefix, _ = fn[:-len('.pkl')].split('_')
        sol_fpath = opath.join(anl_dpath, fn)
        dplym_fpath = opath.join(anl_dpath, 'dplym_%s.pkl' % prefix)
        prmt_fpath = opath.join(anl_dpath, 'prmt_%s.pkl' % prefix)
        with open(sol_fpath, 'rb') as fp:
            sol = pickle.load(fp)
        with open(prmt_fpath, 'rb') as fp:
            prmt = pickle.load(fp)
        T, cB_M, cB_P = [prmt.get(k) for k in ['T','cB_M', 'cB_P']]
        C, q_c = [sol.get(k) for k in ['C', 'q_c']]
        selBundles = [C[c] for c in range(len(C)) if q_c[c] > 0.5]
        nt_tib = {i: 0 for i in range(2, 6)}
        nbt = 0
        for bc in selBundles:
            nt_tib[len(bc)] += 1
            nbt += len(bc)
            
        with open(argmt_fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_row = [prefix, cB_M, cB_P] + [nt_tib[i] for i in range(2, 6)] + [nbt, len(T) - nbt]
            writer.writerow(new_row)
    

def bundleAnalysis():
    anl_dpath = opath.join(exp_dpath, '__analysis')

    argmt_fpath = opath.join(anl_dpath, '_arrangement.csv')
    sol_fn = 'sol_11interOut-nt200-mDP20-mTB4-dp25-fp75-sn0_CWL4.pkl'
    #
    _, prefix, _ = sol_fn[:-len('.pkl')].split('_')
    sol_fpath = opath.join(anl_dpath, sol_fn)
    dplym_fpath = opath.join(anl_dpath, 'dplym_%s.pkl' % prefix)
    prmt_fpath = opath.join(anl_dpath, 'prmt_%s.pkl' % prefix)
    
    colFP_dpath = opath.join(opath.join(anl_dpath, 'selColFP'), 'scFP_%s' % '11interOut-nt200-mDP20-mTB4-dp25-fp75-sn0_CWL4')
    


    with open(sol_fpath, 'rb') as fp:
        sol = pickle.load(fp)
    with open(dplym_fpath, 'rb') as fp:
        flow_oridest, task_ppdp = pickle.load(fp)
    with open(prmt_fpath, 'rb') as fp:
        prmt = pickle.load(fp)
    C, q_c = [sol.get(k) for k in ['C', 'q_c']]
    selCols = [(c, C[c]) for c in range(len(C)) if q_c[c] > 0.5]

        
        
    
    numBT = sum(len(bc) for bc in selCols)
    bid = 28
    selCols[bid]

    
    
    
    colFP_fpath = opath.join(colFP_dpath, 'bid%d.pkl' % bid)
    with open(colFP_fpath, 'rb') as fp:
        c, Ts, feasiblePath = pickle.load(fp)
    print(bid, c, Ts, feasiblePath)
    for i in Ts:
        print(task_ppdp[i])
    for k in feasiblePath:
        print(flow_oridest[k])
    
    len(selCols)


if __name__ == '__main__':
    # dataProcessing()
    # summary()
    # diff_vc()
    num_tasksInBundles()
