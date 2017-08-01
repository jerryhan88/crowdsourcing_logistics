from init_project import *
#
from problems import random_problem
from mathematicalModel import run_mip_eliSubTour, convert_input4MathematicalModel
from greedyHeuristic import run_greedyHeuristic, convert_input4greedyHeuristic
#
import csv


def run(processorID, num_workers=11):
    maxFlow = 3
    minReward, maxReward = 1, 3
    minVolume, maxVolume = 1, 2
    thVolume = 3
    detourAlowProp = 0.5
    jobID = 0
    for i in range(2, 9):
        numCols = numRows = i
        for numTasks in range(5, 50, 5):
            for j in range(1, 10):
                bundleResidualProp = 1 + j / 10.0
                inputs, fn = random_problem(numCols, numRows, maxFlow,
                               numTasks, minReward, maxReward, minVolume, maxVolume,
                               thVolume, bundleResidualProp, detourAlowProp)
                if jobID % num_workers != processorID:
                    continue
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
                              'm_obj', 'h_obj', 'gap_obj', 'm_time', 'h_time', 'gap_time', ]
                    writer.writerow(header)
                try:
                    m_obj, m_time = run_mip_eliSubTour(convert_input4MathematicalModel(*inputs))
                    h_obj, h_time = run_greedyHeuristic(convert_input4greedyHeuristic(*inputs))
                    gap_obj = (m_obj - h_obj) / float(m_obj)
                    gap_time = (m_time - h_time) / float(m_time)
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
                jobID += 1


if __name__ == '__main__':
    run(0)
