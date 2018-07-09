import os.path as opath
import csv, pickle
import pandas as pd
from scipy import stats
#
from __path_organizer import exp_dpath
from mrtScenario import get_travelTimeSG
from PD_IH import calc_detourTime


def dataProcessing():
    anl_dpath = opath.join(exp_dpath, '__analysis')
    argmt_fpath = opath.join(anl_dpath, '_arrangement.csv')
    sol_fn = 'sol_11interOut-nt200-mDP20-mTB4-dp25-fp75-sn0_CWL4.pkl'
    #
    _, prefix, _ = sol_fn[:-len('.pkl')].split('_')
    sol_fpath = opath.join(anl_dpath, sol_fn)
    dplym_fpath = opath.join(anl_dpath, 'dplym_%s.pkl' % prefix)
    prmt_fpath = opath.join(anl_dpath, 'prmt_%s.pkl' % prefix)
    with open(sol_fpath, 'rb') as fp:
        sol = pickle.load(fp)
    with open(dplym_fpath, 'rb') as fp:
        flow_oridest, task_ppdp = pickle.load(fp)
    with open(prmt_fpath, 'rb') as fp:
        prmt = pickle.load(fp)
    #
    MRTs_tt, locPD_MRT_tt = get_travelTimeSG()
    T, K, w_k, t_ij, _delta = [prmt.get(k) for k in ['T', 'K', 'w_k', 't_ij', '_delta']]
    tb_assignment = [False for _ in range(len(T))]
    C, q_c = [sol.get(k) for k in ['C', 'q_c']]
    selBundles = [C[c] for c in range(len(C)) if q_c[c] > 0.5]
    for bc in selBundles:
        for i in bc:
            tb_assignment[i] = True

    with open(argmt_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        writer.writerow(['tid',
                         'pp_nMRT', 'dp_nMRT', 'detourTime', 'sW',
                         'assign'])
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
            writer.writerow([i,
                             pp_nMRT, dp_nMRT, pp_tt + dp_tt, sW,
                             1 if tb_assignment[i] else 0])


def summary():
    anl_dpath = opath.join(exp_dpath, '__analysis')
    argmt_fpath = opath.join(anl_dpath, '_arrangement.csv')
    if not opath.exists(argmt_fpath):
        dataProcessing()
    sum_fpath = opath.join(anl_dpath, '_summary.csv')
    #
    df = pd.read_csv(argmt_fpath)
    df0 = df[(df['assign']==0)]
    df1 = df[(df['assign']!=0)]

    with open(sum_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        writer.writerow([None, 'xbt', 'bt', 'tScore', 'pValue'])
        #
        tScore, pValue = stats.ttest_ind(df0['detourTime'], df1['detourTime'], equal_var=False)
        writer.writerow(['detourTime', df0['detourTime'].mean(), df1['detourTime'].mean(), tScore, pValue])
        #
        tScore, pValue = stats.ttest_ind(df0['sW'], df1['sW'], equal_var=False)
        writer.writerow(['sW', df0['sW'].mean(), df1['sW'].mean(), tScore, pValue])


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

FIGSIZE = (8, 6)
import matplotlib.pyplot as plt

_rgb = lambda r, g, b: (r / float(255), g / float(255), b / float(255))
clists = (
    'blue', 'green', 'red', 'magenta', 'black', 'cyan',
    _rgb(255, 165, 0), _rgb(238, 130, 238), _rgb(255, 228, 225),  # orange, violet, misty rose
    _rgb(127, 255, 212),  # aqua-marine
    'yellow',
    _rgb(220, 220, 220), _rgb(255, 165, 0),  # gray, orange
    'black'
)
mlists = (
    'o',  # circle
    '^',  # triangle_up
    'D',  # diamond
    '*',  # star
    '8',  # octagon


    'v',  #    triangle_down

    '<',  #    triangle_left
    '>',  #    triangle_right
    's',  #    square
    'p',  #    pentagon

    '+',  #    plus
    'x',  #    x

    'h',  #    hexagon1
    '1',  #    tri_down
    '2',  #    tri_up
    '3',  #    tri_left
    '4',  #    tri_right

    'H',  #    hexagon2
    'd',  #    thin_diamond
    '|',  #    vline
    '_',  #    hline
    '.',  #    point
    ',',  #    pixel

    '8',  #    octagon
    )

FONT_SIZE = 18

import numpy as np


def draw_charts():
    ratios = [[64, 68, 69, 69, 70, 70, 71, 72, 73],
         [75, 72, 70, 68, 67, 65, 64, 63, 62],
         [80, 77, 74, 72, 70, 68, 68, 67, 64],
         [85, 81, 78, 75, 73, 73, 70, 69, 68],
         [90, 85, 81, 79, 77, 76, 74, 73, 71], ]

    profits = [[292, 390, 477, 561, 654, 749, 842, 941, 1043],
        [225, 401, 580, 746, 929, 1078, 1244, 1407, 1542],
        [-376, -174, -5, 175, 349, 507, 694, 868, 953],
        [-972, -773, -585, -403, -232, -31, 117, 298, 474],
        [-1567, -1363, -1180, -974, -792, -596, -435, -243, -107],]


    tasks = list(np.arange(100, 301, 25))
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('# T', fontsize=FONT_SIZE)
    # ax.set_ylabel('%', fontsize=FONT_SIZE)
    ax.set_ylabel('$', fontsize=FONT_SIZE)
    for i, y in enumerate(profits):
        plt.plot(range(len(tasks)), y, color=clists[i], marker=mlists[i])

    plt.legend(['MCS'] + ['DF%d' % nv for nv in [5, 10, 15, 20]], ncol=1, fontsize=FONT_SIZE)
    plt.xticks(range(len(tasks)), tasks)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    plt.ylim((-2100, 1600))
    # img_ofpath = opath.join('_charts', '%s-%s-%s.pdf' % (lv, mea, ma_prefix))
    img_ofpath = 'temp.pdf'

    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    # summary()

    draw_charts()
