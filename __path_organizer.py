import os.path as opath
import os
from functools import reduce


data_dpath = reduce(opath.join, ['..', '_data', 'crowdsourcing_logistics'])
ef_dpath = opath.join(data_dpath, 'ExternalFiles')
ez_dpath = opath.join(data_dpath, 'EZlinkData')
mrtNet_dpath = opath.join(data_dpath, 'MRT_Network')
#
pf_dpath = opath.join(data_dpath, 'ProcessedFiles')
viz_dpath = opath.join(data_dpath, 'Viz')
#
exp_dpath = opath.join(data_dpath, 'Experiments')

dir_paths = [data_dpath,
             ef_dpath, ez_dpath, mrtNet_dpath,
             pf_dpath, viz_dpath,
             #
             exp_dpath]


for dpath in dir_paths:
    if opath.exists(dpath):
        continue
    os.mkdir(dpath)