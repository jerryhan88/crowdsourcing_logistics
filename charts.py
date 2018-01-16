from init_project import *
#
import os.path as opath
import os, fnmatch
import csv
import numpy as np
import pandas as pd
import datetime
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

FIGSIZE = (8, 6)
FIGSIZE2 =(8, 4)

_fontsize = 14




def run():
    sum_fpath = opath.join(dpath['experiment'], 'experiment_summary.csv')
    #
    nts, tds = set(), set()
    nt_td_obj, nt_td_comT = {}, {}
    with open(sum_fpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            numTasks, thDetour = [int(row[cn]) for cn in ['numTasks', 'thDetour']]
            bnp_objV, bnp_wallT = [float(row[cn]) for cn in ['bnp_objV', 'bnp_wallT(h)']]
            #
            nts.add(numTasks)
            tds.add(thDetour)
            #
            nt_td_obj[numTasks, thDetour] = bnp_objV
            nt_td_comT[numTasks, thDetour] = bnp_wallT
    nts, tds = map(list, [nts, tds])
    nts.sort()
    tds.sort()
    #
    for measure, data in [('ObjV', nt_td_obj),
                          ('ComT', nt_td_comT)]:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)
        ax.set_xlabel('Allowable detour time', fontsize=_fontsize)
        ax.set_ylabel('%s' % measure, fontsize=_fontsize)
        plt.xticks(range(len(tds)), tds)
        for i, nt in enumerate(nts):
            y = []
            for td in tds:
                if (nt, td) in data:
                    y.append(data[nt, td])
                else:
                    y.append(None)
            plt.plot(range(len(tds)), y, color=clists[i], marker=mlists[i])
        #
        plt.legend(['%d tasks ' % nt for nt in nts], ncol=1, loc='upper right', fontsize=_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=_fontsize)
        #
        # plt.ylim(_ylim)
        img_ofpath = '%s.pdf' % measure
        plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    run()