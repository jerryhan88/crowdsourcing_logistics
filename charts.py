import os.path as opath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from functools import reduce
#
from __path_organizer import exp_dpath

_rgb = lambda r, g, b: (r / float(255), g / float(255), b / float(255))
clists = (
    'blue', 'green', 'red', 'magenta', 'black', #'cyan',
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
    'v',  # triangle_down
    's',  #    square



    '8',  # octagon


    '<',  # triangle_left
    '>',  #    triangle_right

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


FIGSIZE = (8, 6)
FONT_SIZE = 18
MARKER_SIZE = 12

def TT_DF_PF():
    labels = ['MCS%d' % dt for dt in [30, 60, 90]] + ['DF%d' % nv for nv in [5, 10, 15]]
    d2d_ratio = [0, 0.25, 0.5, 0.75, 1]        

    profit = [[137.6, 86.5, 51.1, 26.1, 11.7], 
                [1102.4, 975.6, 831.3, 599.8, 408.9], 
                [1561.4, 1399, 1191.1, 1103.6, 1025.5], 
                [1564.835612, 1296.86525, 1487.289168, 1676.883939, 1589.128548],
                [1012.3785, 702.4398839, 941.695547, 1083.586061, 1045.651343],
                [428.695396, 145.1280603, 342.7044793, 495.4122872, 452.1762485],
                ]
    fraction = [[0.099333333, 0.063333333, 0.037666667, 0.019333333, 0.008666667],
                [0.759, 0.696, 0.611333333, 0.443, 0.302333333],
                [0.984, 0.906666667, 0.807666667, 0.766, 0.728333333],
                [0.632, 0.554761905, 0.607333333, 0.659333333, 0.633333333], 
                [0.666, 0.577, 0.643, 0.681666667, 0.668], 
                [0.693, 0.61, 0.664, 0.704666667, 0.689], 
                ]
    
    
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Profit ($)', fontsize=FONT_SIZE)
    ax.set_ylabel('Fraction', fontsize=FONT_SIZE)
    mask_data = [False, False, False, False, True]
    for i in range(len(profit)):
        x_data = np.array(profit[i])
        y_data = np.array(fraction[i])
        plt.scatter(x_data[mask_data], y_data[mask_data], s=120, color=clists[i], marker=mlists[i], label=labels[i])
    plt.legend(labels, ncol=2, fontsize=FONT_SIZE + 4, 
           handletextpad=0.11, columnspacing=0.1, loc='lower right')
    plt.text(1050, 0.35, 'Total: %d tasks' % 300, fontsize=FONT_SIZE + 4)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE + 4)
    plt.xlim((-100, 2000))
    plt.ylim((-0.1, 1.1))
    img_ofpath = 'TT_DF_PF_%d.pdf' % 100
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


def TT_DF_Profit():
    labels = ['MCS%d' % dt for dt in [30, 60, 90]] + ['DF%d' % nv for nv in [5, 10, 15]]
    d2d_ratio = [0, 0.25, 0.5, 0.75, 1]    
    profit = [[137.6, 86.5, 51.1, 26.1, 11.7], 
                [1102.4, 975.6, 831.3, 599.8, 408.9], 
                [1561.4, 1399, 1191.1, 1103.6, 1025.5], 
                [1564.835612, 1296.86525, 1487.289168, 1676.883939, 1589.128548],
                [1012.3785, 702.4398839, 941.695547, 1083.586061, 1045.651343],
                [428.695396, 145.1280603, 342.7044793, 495.4122872, 452.1762485],
                ]
    
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('D2D Ratio', fontsize=FONT_SIZE)
    ax.set_ylabel('Profit', fontsize=FONT_SIZE)        
    xs = np.arange(len(d2d_ratio))
    for i in range(len(profit)):
        plt.plot(xs, profit[i], color=clists[i], marker=mlists[i], markersize=MARKER_SIZE)        
#    plt.text(2.5, 1.2, 'Total: %d tasks' % 300, fontsize=FONT_SIZE)
    plt.legend(labels, ncol=2, fontsize=FONT_SIZE, 
               handletextpad=0.11, columnspacing=0.1, loc='upper left')
    plt.xticks(xs, d2d_ratio)
    plt.ylim((-100, 2400))
#    plt.yticks(np.arange(0, 1.1, step=0.2))
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    #
    img_ofpath = 'TT_DF_Profit.pdf'
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


def TT_DF_Fraction():
    labels = ['MCS%d' % dt for dt in [30, 60, 90]] + ['DF%d' % nv for nv in [5, 10, 15]]
    d2d_ratio = [0, 0.25, 0.5, 0.75, 1]    
    fraction = [[0.099333333, 0.063333333, 0.037666667, 0.019333333, 0.008666667],
                [0.759, 0.696, 0.611333333, 0.443, 0.302333333],
                [0.984, 0.906666667, 0.807666667, 0.766, 0.728333333],
                [0.632, 0.554761905, 0.607333333, 0.659333333, 0.633333333], 
                [0.666, 0.577, 0.643, 0.681666667, 0.668], 
                [0.693, 0.61, 0.664, 0.704666667, 0.689], 
                ]
    
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('D2D Ratio', fontsize=FONT_SIZE)
    ax.set_ylabel('Fraction', fontsize=FONT_SIZE)        
    xs = np.arange(len(d2d_ratio))
    for i in range(len(fraction)):
        plt.plot(xs, fraction[i], color=clists[i], marker=mlists[i], markersize=MARKER_SIZE)        
    plt.text(2.5, 1.2, 'Total: %d tasks' % 300, fontsize=FONT_SIZE)
    plt.legend(labels, ncol=2, fontsize=FONT_SIZE, 
               handletextpad=0.11, columnspacing=0.1, loc='upper left')
    plt.xticks(xs, d2d_ratio)
    plt.ylim((-0.1, 1.5))
    plt.yticks(np.arange(0, 1.1, step=0.2))
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    #
    img_ofpath = 'TT_DF_Fraction.pdf'
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)
    

def TT_profit_ratio():
    profits = [[292, 477, 654, 842, 1043, 1234, 1451, 1670, 1890],
           [200, 555, 904, 1231, 1500, 1826, 2117, 2328, 2660],
           [-426, -55, 299, 644, 903, 1284, 1602, 1938, 2175],
           [-1047, -660, -307, 42, 399, 725, 1076, 1404, 1656],
           [-1667, -1280, -892, -535, -207, 149, 512, 812, 1139],]    
    ratios = [[64, 69, 70, 71, 73, 73, 75, 76, 77],
          [75, 70, 67, 65, 61, 60, 59, 56, 56],
          [80, 74, 70, 68, 64, 64, 62, 62, 59],
          [85, 78, 73, 70, 68, 66, 66, 64, 62],
          [90, 81, 77, 74, 71, 69, 68, 66, 65],]        
    tasks = list(np.arange(100, 501, 50))    
    labels = ['MCS'] + ['DF%d' % nv for nv in [5, 10, 15, 20]]
    #
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_ylabel('Fraction', fontsize=FONT_SIZE)
    ax.set_xlabel('Profit ($)', fontsize=FONT_SIZE)
    mask_data = [True if tasks[i] in [100, 300, 500] else False for i in range(len(ratios[0]))]
    for i in range(len(ratios)):
        x_data = np.array(profits[i])
        y_data = np.array(ratios[i])
        y_data = y_data / 100
        plt.scatter(x_data[mask_data], y_data[mask_data], s=70, color=clists[i], marker=mlists[i], label=labels[i])
    
    plt.legend(labels, ncol=len(labels), fontsize=FONT_SIZE, 
           handletextpad=0.11, columnspacing=0.1, loc='upper center')
    
    for i in range(len(ratios[0])):
        if tasks[i] not in [100, 300, 500]:
            continue
        x_data = [p[i] for p in profits]
        y_data = [r[i] for r in ratios]
        y_data = np.array(y_data)
        y_data = y_data / 100
        plt.plot(x_data, y_data, color='black')
        if tasks[i] == 100:
            plt.text(x_data[0]-400, y_data[0] - 0.03, '#T=%d' % tasks[i], fontsize=FONT_SIZE)
        else:
            plt.text(x_data[0]-400, y_data[0] + 0.02, '#T=%d' % tasks[i], fontsize=FONT_SIZE)
    plt.ylim((0.5, 1.0))
    plt.yticks(np.arange(0.5, 1.01, step=0.1))

    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    img_ofpath = 'PA_RP.pdf'
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)
    

def TT_objV():
    summaryTT_dpath = opath.join(exp_dpath, '_TaskType')
    sum_fpath = reduce(opath.join, [summaryTT_dpath, 'summaryTT.csv'])
    odf = pd.read_csv(sum_fpath)
    
    assert len(set(odf['numTasks'])) == 1
    numTask = set(odf['numTasks']).pop()
    
    
    odf = odf.drop(['numPaths', 'numTasks'], axis=1)    

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('D2D Ratio', fontsize=FONT_SIZE)
    ax.set_ylabel('Fraction', fontsize=FONT_SIZE)        
    detours = sorted(list(set(odf['thDetour'])))
    ratios = sorted(list(set(odf['d2d_ratio'])))
    xs = np.arange(len(ratios))
    for i, dt in enumerate(detours):
        df = odf[(odf['thDetour'] == dt)]
        plt.plot(xs, df['objV'] / numTask, color=clists[i], marker=mlists[i], markersize=MARKER_SIZE)        
    plt.legend(['MCS%d' % dt for dt in detours], ncol=len(detours), fontsize=FONT_SIZE - 3, 
               handletextpad=0.11, columnspacing=0.1, loc='upper center')
    plt.xticks(xs, ratios)
    plt.ylim((-10 / numTask, 950 / numTask))    
    plt.text(1.5, 0.95, 'Total: %d tasks' % numTask, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    #
    img_ofpath = 'TT_objV.pdf'
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


def PP_numCols():
    aprcs = ['CWL%d' % cwl_no for cwl_no in range(1, 6)]
    summaryPP_dpath = opath.join(exp_dpath, '_PracticalProblems')
    sum_fpath = reduce(opath.join, [summaryPP_dpath, 'summaryPP.csv'])
    odf = pd.read_csv(sum_fpath)    
    for numPath in set(odf['numPaths']):
        df = odf[(odf['numPaths'] == numPath)]
        avgs = {}
        num_tasks = set()
        for nt, v_c1, v_c2, v_c3, v_c4, v_c5 in df[['numTasks'] + ['%s_numCols' % cn for cn in aprcs]].values:
            for k, v in zip(aprcs, [v_c1, v_c2, v_c3, v_c4, v_c5]):
#                avgs[nt, k] = v / HOUR1
                avgs[nt, k] = v
            num_tasks.add(nt)
            
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)
        ax.set_xlabel('# T', fontsize=FONT_SIZE)
        ax.set_ylabel('# of columns', fontsize=FONT_SIZE)
        tasks = list(np.arange(50, 801, 50))
        xs = np.arange(len(tasks))
        
        labels = ['CwL%d' % cwl_no for cwl_no in range(1, 6)]
        for i, cn in enumerate(aprcs):
            y_data = np.array([avgs[nt, cn] if (nt, cn) in avgs else np.nan for nt in tasks])
            ma_y = np.isfinite(y_data)
            plt.plot(xs[ma_y], y_data[ma_y], color=clists[i], marker=mlists[i], markersize=MARKER_SIZE)
#        ax.set_yscale('symlog')
        plt.legend(labels, ncol=3, fontsize=FONT_SIZE,
                   handletextpad=0.11, columnspacing=0.1, loc='upper center')
        x_label = np.array([nt if nt in num_tasks else np.nan for nt in tasks]).astype(np.int)
        plt.xticks(xs[ma_y], x_label[ma_y])
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
#        plt.ylim((-0.3, 5.0))
        plt.ylim((0, 35000))
        img_ofpath = 'PP_numCols_%d.pdf' % numPath
        plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


    

def PP_comT():
    HOUR1 = 3600
    #
    aprcs = ['CWL%d' % cwl_no for cwl_no in range(1, 6)] + ['GH']
    summaryPP_dpath = opath.join(exp_dpath, '_PracticalProblems')
    sum_fpath = reduce(opath.join, [summaryPP_dpath, 'summaryPP.csv'])
    odf = pd.read_csv(sum_fpath)    
    odf = odf.drop(['%s_objV' % aprc for aprc in aprcs], axis=1)    
    odf = odf.drop(['p_%s_objV' % aprc for aprc in aprcs], axis=1)        
    
    for numPath in set(odf['numPaths']):
        df = odf[(odf['numPaths'] == numPath)]
        avgs = {}
        num_tasks = set()
        for nt, v_c1, v_c2, v_c3, v_c4, v_c5, v_gh in df[['numTasks'] + ['%s_cpuT' % cn for cn in aprcs]].values:
            for k, v in zip(aprcs, [v_c1, v_c2, v_c3, v_c4, v_c5, v_gh]):
#                avgs[nt, k] = v / HOUR1
                avgs[nt, k] = v
            num_tasks.add(nt)
#        stds = {}
#        for nt, v_c1, v_c2, v_c3, v_c4, v_c5, v_gh in df[['numTasks'] + ['%s_cpuT_sd' % cn for cn in aprcs]].values:
#            for k, v in zip(aprcs, [v_c1, v_c2, v_c3, v_c4, v_c5, v_gh]):
#                stds[nt, k] = v / HOUR1                   
        #
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)
        ax.set_xlabel('# T', fontsize=FONT_SIZE)
        ax.set_ylabel('Computation time (sec.)', fontsize=FONT_SIZE)
        tasks = list(np.arange(50, 801, 50))
        xs = np.arange(len(tasks))
        
        labels = ['CwL%d' % cwl_no for cwl_no in range(1, 6)] + ['GH']
        for i, cn in enumerate(aprcs):
            y_data = np.array([avgs[nt, cn] if (nt, cn) in avgs else np.nan for nt in tasks])
            ma_y = np.isfinite(y_data)
#            y_error = np.array([stds[nt, cn] if (nt, cn) in avgs else np.nan for nt in tasks])
#            ax.errorbar(xs[ma_y], y_data[ma_y], yerr=y_error[ma_y], elinewidth=1.0, capsize=4, # fmt='-o', 
#                        color=clists[i], marker=mlists[i], markersize=MARKER_SIZE, label=labels[i])
            plt.plot(xs[ma_y], y_data[ma_y], color=clists[i], marker=mlists[i], markersize=MARKER_SIZE)
        ax.set_yscale('symlog')
        
#        handles, labels = ax.get_legend_handles_labels()
#        handles = [h[0] for h in handles]
#        plt.legend(handles, labels, ncol=1, fontsize=FONT_SIZE)
        plt.legend(labels, ncol=3, fontsize=FONT_SIZE,
                   handletextpad=0.11, columnspacing=0.1, loc='upper center')

        x_label = np.array([nt if nt in num_tasks else np.nan for nt in tasks]).astype(np.int)
        plt.xticks(xs[ma_y], x_label[ma_y])
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
#        plt.ylim((-0.3, 5.0))
        plt.ylim((0, 1500000000))
        img_ofpath = 'PP_comT_%d.pdf' % numPath
        plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


def PC_comT_fixPath():
    NUM_PATH, HOUR1 = 2, 3600
    #
    summaryPC_dpath = opath.join(exp_dpath, '_PerformanceComparison')
    sum_fpath = reduce(opath.join, [summaryPC_dpath, 'summaryPC.csv'])
    df = pd.read_csv(sum_fpath)
    df = df.drop(['EX2_objV', 'BNP_objV', 'GH_objV', 'CWL1_objV'], axis=1)
    df = df.drop(['EX2_sol', 'BNP_sol', 'CWL_sol', 'GH_sol'], axis=1)
    df = df.drop(['EX2_DG'], axis=1)    
    df = df[(df['numPaths'] == NUM_PATH)]    
    df['EX2_cpuT'][7] = np.nan; df['EX2_cpuT_sd'][7] = np.nan    
    #
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('# T', fontsize=FONT_SIZE)
    ax.set_ylabel('Computation time (sec.)', fontsize=FONT_SIZE)

    labels = ['ILP', 'BnP', 'CwL', 'GH']
    
    for i, cn in enumerate(['EX2_cpuT', 'BNP_cpuT', 'CWL1_cpuT', 'GH_cpuT']):
        sd_cn = cn + '_sd'
#        ax.errorbar(range(len(df)), df[cn] / HOUR1, yerr=df[sd_cn] / HOUR1, elinewidth=1.0, capsize=4,
#                 color=clists[i], marker=mlists[i], markersize=MARKER_SIZE,
#                 label=labels[i])
        plt.plot(range(len(df)), df[cn], color=clists[i], marker=mlists[i], markersize=MARKER_SIZE)
#    ax.set_yscale('log', basey=10)
    ax.set_yscale('symlog')
#    handles, labels = ax.get_legend_handles_labels()
#    handles = [h[0] for h in handles]
#    plt.legend(handles, labels, ncol=1, fontsize=FONT_SIZE)
    ax.legend(labels, loc=1, bbox_to_anchor=(1.0, 0.5), fontsize=FONT_SIZE)
    plt.xticks(range(len(df['numTasks'])), df['numTasks'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
#    plt.ylim((-1, 24))
    plt.ylim((-1, 150000))
    img_ofpath = 'PC_comT_FP.pdf'
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


def PC_comT_fixTask():
    NUM_TASK, HOUR1 = 10, 3600
    #
    summaryPC_dpath = opath.join(exp_dpath, '_PerformanceComparison')
    sum_fpath = reduce(opath.join, [summaryPC_dpath, 'summaryPC.csv'])
    df = pd.read_csv(sum_fpath)
    df = df.drop(['EX2_objV', 'BNP_objV', 'GH_objV', 'CWL1_objV'], axis=1)
    df = df.drop(['EX2_sol', 'BNP_sol', 'CWL_sol', 'GH_sol'], axis=1)
    df = df.drop(['EX2_DG'], axis=1)    
    df = df[(df['numTasks'] == NUM_TASK)]    
    #
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('# P', fontsize=FONT_SIZE)
    ax.set_ylabel('Computation time (sec.)', fontsize=FONT_SIZE)
    
    labels = ['ILP', 'BnP', 'CwL', 'GH']
    
    for i, cn in enumerate(['EX2_cpuT', 'BNP_cpuT', 'CWL1_cpuT', 'GH_cpuT']):
        sd_cn = cn + '_sd'
#        ax.errorbar(range(len(df)), df[cn] / HOUR1, yerr=df[sd_cn] / HOUR1, elinewidth=1.0, capsize=4,
#                     color=clists[i], marker=mlists[i], markersize=MARKER_SIZE,
#                     label=labels[i])
        plt.plot(range(len(df)), df[cn], color=clists[i], marker=mlists[i], markersize=MARKER_SIZE)
    ax.set_yscale('symlog')
#    handles, labels = ax.get_legend_handles_labels()
#    handles = [h[0] for h in handles]
#    plt.legend(handles, labels, ncol=1, fontsize=FONT_SIZE)
    plt.legend(labels, ncol=1, fontsize=FONT_SIZE)
    plt.xticks(range(len(df['numPaths'])), df['numPaths'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.ylim((-1, 150000))
    img_ofpath = 'PC_comT_FT.pdf'
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)
    

def PA_ratio():
    ratios = [[64, 69, 70, 71, 73, 73, 75, 76, 77],
              [75, 70, 67, 65, 61, 60, 59, 56, 56],
              [80, 74, 70, 68, 64, 64, 62, 62, 59],
              [85, 78, 73, 70, 68, 66, 66, 64, 62],
              [90, 81, 77, 74, 71, 69, 68, 66, 65],]
    #
    tasks = list(np.arange(100, 501, 50))
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('# T', fontsize=FONT_SIZE)
    ax.set_ylabel('%', fontsize=FONT_SIZE)
    for i, y in enumerate(ratios):
        plt.plot(range(len(tasks)), y, color=clists[i], marker=mlists[i], markersize=MARKER_SIZE)

    plt.legend(['MCS'] + ['DF%d' % nv for nv in [5, 10, 15, 20]], ncol=1, fontsize=FONT_SIZE)
    plt.xticks(range(len(tasks)), tasks)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    #
    img_ofpath = 'PA_ratio.pdf'
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


def PA_profit():
    profits = [[292, 477, 654, 842, 1043, 1234, 1451, 1670, 1890],
               [200, 555, 904, 1231, 1500, 1826, 2117, 2328, 2660],
               [-426, -55, 299, 644, 903, 1284, 1602, 1938, 2175],
               [-1047, -660, -307, 42, 399, 725, 1076, 1404, 1656],
               [-1667, -1280, -892, -535, -207, 149, 512, 812, 1139],]
    #
    tasks = list(np.arange(100, 501, 50))
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('# T', fontsize=FONT_SIZE)
    ax.set_ylabel('Profit ($)', fontsize=FONT_SIZE)
    for i, y in enumerate(profits):
        plt.plot(range(len(tasks)), y, color=clists[i], marker=mlists[i], markersize=MARKER_SIZE)

    plt.legend(['MCS'] + ['DF%d' % nv for nv in [5, 10, 15, 20]], ncol=1, fontsize=FONT_SIZE)
    plt.xticks(range(len(tasks)), tasks)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    #
    # plt.ylim((-2100, 1600))
    # img_ofpath = opath.join('_charts', '%s-%s-%s.pdf' % (lv, mea, ma_prefix))
    img_ofpath = 'PA_profit.pdf'
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


def PR_objV():
    summaryPR_dpath = opath.join(exp_dpath, '_ParticipationRate')
    sum_fpath = reduce(opath.join, [summaryPR_dpath, 'summaryPR.csv'])
    wdf = pd.read_csv(sum_fpath)
    TOTAL_PASSENGER = 2349832
    #
    for numTasks in set(wdf['numTasks']):
        odf = wdf[(wdf['numTasks'] == numTasks)]
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)
        prs = sorted(list(set(odf['alpha'])))
        thPs = sorted(list(set(odf['thP'])))
        xs = np.arange(len(prs))
        for i, thP in enumerate(thPs):
            df = odf[(odf['thP'] == thP)]
            plt.plot(xs, df['objV'] / numTasks, color=clists[i], marker=mlists[i], markersize=MARKER_SIZE)
        plt.legend([r'P = %d%%' % (thP * 100) for thP in thPs], ncol=1, fontsize=FONT_SIZE - 3,
               handletextpad=0.11, columnspacing=0.1, loc='upper left')
        plt.xticks(xs, [r'$10^{%d}$' % math.log10(pr) for pr in prs])
        plt.ylim((0.0, 1.0))
        plt.text(1.8, 0.4, '# tasks: %d' % numTasks, fontsize=FONT_SIZE)
        plt.text(1.8, 0.48, '# passengers: %s ' % "{:,}".format(TOTAL_PASSENGER), fontsize=FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        ax.set_xlabel('Participation probability', fontsize=FONT_SIZE)
        ax.set_ylabel('Ratio', fontsize=FONT_SIZE)
        #
        img_ofpath = 'PR_objV_%d.pdf' % numTasks
        plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


def PC_perGap():
    numTasks = [10, 15, 20, 25, 30]
    
    
    ydata = [
     [0, 0, 0 ,5 ,9],
     [0, 0, 0, 0, 0],
     [40, 40, 44, 40, 39],
    ]
    labels = ['ILP', 'CwL', 'GH']
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE)
    
    for ax in [ax1, ax2]:    
        ax.plot(range(len(numTasks)), ydata[0], color=clists[0], marker=mlists[0], markersize=MARKER_SIZE)
        ax.plot(range(len(numTasks)), ydata[1], color=clists[2], marker=mlists[2], markersize=MARKER_SIZE)
        ax.plot(range(len(numTasks)), ydata[2], color=clists[3], marker=mlists[3], markersize=MARKER_SIZE)
    
    ax1.set_ylim(35, 55)
    ax2.set_ylim(-1, 14)
    
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    for ax in [ax1, ax2]:    
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.xticks(range(len(numTasks)), numTasks)
    
    
    fig.text(0.5, 0.04, '# T', ha='center', fontsize=FONT_SIZE)
    fig.text(0.04, 0.5, 'Optimality gap (%)', va='center', rotation='vertical', fontsize=FONT_SIZE)
    ax1.legend(labels, bbox_to_anchor=(0.01, 0.315), fontsize=FONT_SIZE)
    fig.text(0.5, 0.5, '*Compared with BnP', ha='center', fontsize=FONT_SIZE)

    img_ofpath = 'PC_objV_FP.pdf'
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)


def PP_perGap():
    NUM_TASKS = 400
    aprcs = ['CWL%d' % cwl_no for cwl_no in range(1, 6)] + ['GH']
    summaryPP_dpath = opath.join(exp_dpath, '_PracticalProblems')
    sum_fpath = reduce(opath.join, [summaryPP_dpath, 'summaryPP.csv'])
    odf = pd.read_csv(sum_fpath)
    odf = odf[['numPaths', 'numTasks'] + ['%s_objV' % aprc for aprc in aprcs]]    
    odf = odf[(odf['numTasks'] == NUM_TASKS)]
    for aprc in aprcs:        
        odf['%s_gap' % aprc] = (odf['CWL1_objV'] - odf['%s_objV' % aprc]) / odf['CWL1_objV'] * 100
    np_gap = {}
    for _, row in odf.iterrows():
        np_gap[row['numPaths']] = [(aprc, row['%s_gap' % aprc]) for aprc in aprcs[1:]]
    
    aprc_xy = {aprc: [] for aprc in aprcs[1:]}
    for i, (num_path, aprc_gap) in enumerate(np_gap.items()):
        for j, (aprc, gap) in enumerate(aprc_gap):
            aprc_xy[aprc].append([0.25 + i + 0.1 * j, gap])
            
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.set_ylabel('Solution quality gap (%)', fontsize=FONT_SIZE)
    ax.set_xlabel('Scenario', fontsize=FONT_SIZE)
    labels = ['CwL%d' % cwl_no for cwl_no in range(2, 6)] + ['GH']
    for i, aprc in enumerate(aprcs[1:]):
        x_data, y_data = zip(*aprc_xy[aprc])
        plt.scatter(x_data, y_data, s=70, color=clists[i+1], marker=mlists[i+1], label=labels[i])
    plt.legend(labels, ncol=5, fontsize=FONT_SIZE, 
       handletextpad=0.11, columnspacing=0.05, bbox_to_anchor=(1.015, 0.6))

    plt.xticks([0.45 + i for i in range(3)], [r'$FC$', r'$ST$', r'$ER$'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    ax.set_xlim(-0.1, 3.1)
    fig.text(0.18, 0.6, '*# tasks: %d\n**Compared with CwL1' % NUM_TASKS, ha='left', fontsize=FONT_SIZE)

    img_ofpath = 'PP_perGap_%d.pdf' % NUM_TASKS
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)    
    
    
if __name__ == '__main__':
    # TT_objV()
    PP_comT()
    # PA_ratio()
    # PA_profit()
    # PR_objV()
    # PC_perGap()
    # PP_perGap()
