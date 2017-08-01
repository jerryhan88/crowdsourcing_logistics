from init_project import *
#
from problems import random_problem
from mathematicalModel import run_mip_eliSubTour, convert_input4MathematicalModel
from greedyHeuristic import run_greedyHeuristic, convert_input4greedyHeuristic
#
import csv
import pickle


def run(processorID, num_workers=11):
    for i, fn in enumerate(os.listdir(dpath['problem'])):
        if not fn.endswith('.pkl'):
            continue
        ifpath = opath.join(dpath['problem'], fn)
        inputs = None
        with open(ifpath, 'rb') as fp:
            inputs = pickle.load(fp)
        if i % num_workers != processorID:
            continue
        points, travel_time, \
        flows, paths, \
        tasks, rewards, volumes, \
        numBundles, thVolume, thDetour = inputs
        numTasks, numPaths = map(len, [tasks, paths])
        fn = 'nt%d-np%d-nb%d-tv%d-td%d.csv' % (numTasks, numPaths, numBundles, thVolume, thDetour)
        ofpath = opath.join(dpath['experiment'], fn)
        if opath.exists(ofpath):
            continue
        with open(ofpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            header = ['numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour',
                      'm_obj', 'h_obj', 'gapR_obj', 'm_time', 'h_time', 'gap_time', ]
            writer.writerow(header)
        try:
            m_obj, m_time = run_mip_eliSubTour(convert_input4MathematicalModel(*inputs))
            h_obj, h_time = run_greedyHeuristic(convert_input4greedyHeuristic(*inputs))
            gap_obj = (m_obj - h_obj) / float(m_obj)
            gap_time = (m_time - h_time)
            with open(ofpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([numTasks, numPaths, numBundles, thVolume, thDetour,
                                 m_obj, h_obj, gap_obj,
                                 m_time, h_time, gap_time])
        except:
            with open(ofpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([numTasks, numPaths, numBundles, thVolume, thDetour,
                                 -1, -1, -1,
                                 -1, -1, -1])

def single_run(fn):
    ifpath = opath.join(dpath['problem'], fn)
    inputs = None
    with open(ifpath, 'rb') as fp:
        inputs = pickle.load(fp)
    points, travel_time, \
    flows, paths, \
    tasks, rewards, volumes, \
    numBundles, thVolume, thDetour = inputs
    numTasks, numPaths = map(len, [tasks, paths])
    fn = 'nt%d-np%d-nb%d-tv%d-td%d.csv' % (numTasks, numPaths, numBundles, thVolume, thDetour)
    ofpath = opath.join(dpath['experiment'], fn)
    with open(ofpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour',
                  'm_obj', 'h_obj', 'gapR_obj', 'm_time', 'h_time', 'gap_time', ]
        writer.writerow(header)
    try:
        m_obj, m_time = run_mip_eliSubTour(convert_input4MathematicalModel(*inputs))
        h_obj, h_time = run_greedyHeuristic(convert_input4greedyHeuristic(*inputs))
        gap_obj = (m_obj - h_obj) / float(m_obj)
        gap_time = (m_time - h_time)
        with open(ofpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow([numTasks, numPaths, numBundles, thVolume, thDetour,
                             m_obj, h_obj, gap_obj,
                             m_time, h_time, gap_time])
    except:
        with open(ofpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow([numTasks, numPaths, numBundles, thVolume, thDetour,
                             -1, -1, -1,
                             -1, -1, -1])


if __name__ == '__main__':
    single_run('nt3-np72-nb3-tv3-td4.pkl')
    # run(0)
