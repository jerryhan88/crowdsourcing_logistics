from init_project import *
#
from cpuinfo import get_cpu_info
import platform
import shutil
import pickle, csv
#
from exactMM import run as exactMM_run
from colGenMM import run as colGenMM_run
from greedyHeuristic import run as gHeuristic_run
from problems import *


def gen_problems(problem_dpath):
    #
    # Generate problems
    #
    problem_summary_fpath = opath.join(problem_dpath, '__problem_summary.csv')
    with open(problem_summary_fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        new_headers = ['fn', 'numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour']
        writer.writerow(new_headers)
    #
    maxFlow = 3
    minReward, maxReward = 1, 3
    minVolume, maxVolume = 1, 3
    volumeAlowProp, detourAlowProp = 1.2, 1.2
    numCols, numRows = 1, 5
    for numTasks in range(4, 18, 2):
        for numBundles in range(3, max(4, int(numTasks / 4))):

            inputs = random_problem(numCols, numRows, maxFlow,
                                    numTasks, minReward, maxReward, minVolume, maxVolume,
                                    numBundles, volumeAlowProp, detourAlowProp)
            travel_time, \
            flows, paths, \
            tasks, rewards, volumes, \
            numBundles, thVolume, thDetour = inputs
            numTasks, numPaths = map(len, [tasks, paths])
            fn = 'nt%d-np%d-nb%d-tv%d-td%d.pkl' % (numTasks, numPaths, numBundles, thVolume, thDetour)
            with open(problem_summary_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([fn, numTasks, numPaths, numBundles, thVolume, thDetour])
            ofpath = opath.join(problem_dpath, fn)
            with open(ofpath, 'wb') as fp:
                pickle.dump(inputs, fp)


def init_expEnv(initEnv=False):
    node_name = platform.node()
    exp_dpath = opath.join(dpath['experiment'], node_name)
    problem_dpath = opath.join(exp_dpath, '__problem')
    log_dpath = opath.join(exp_dpath, 'log')
    res_dpath = opath.join(exp_dpath, 'res')
    if initEnv and opath.exists(exp_dpath):
        shutil.rmtree(exp_dpath)
    if not opath.exists(exp_dpath):
        for path in [exp_dpath, problem_dpath, log_dpath, res_dpath]:
            os.makedirs(path)
        #
        cpu_info = get_cpu_info()
        cpu_spec_fpath = opath.join(exp_dpath, '__cpuSpec.txt')
        with open(cpu_spec_fpath, 'w') as f:
            f.write('numProcessor: %d\n' % int(cpu_info['count']))
            f.write('bits: %d\n' % int(cpu_info['bits']))
            f.write('brand:%s' % str(cpu_info['brand']))
        gen_problems(problem_dpath)
    #
    return log_dpath, res_dpath, [opath.join(problem_dpath, fn) for fn in os.listdir(problem_dpath)
                       if fn.endswith('.pkl')]


def cluster_run(processorID, num_workers=11):
    _numThreads, _TimeLimit = 1, None
    #
    log_dpath, res_dpath, problemPaths = init_expEnv(True)
    for i, ifpath in enumerate(problemPaths):
        if i % num_workers != processorID:
            continue
        inputs = None
        with open(ifpath, 'rb') as fp:
            inputs = pickle.load(fp)
        prefix = opath.basename(ifpath)[:-len('.pkl')]
        nt, np, nb, tv, td = [int(v[len('xx'):]) for v in prefix.split('-')]

        #
        # exactMM
        #
        for m, func in [('exactMM', exactMM_run),
                        ('colGenMM', colGenMM_run)]:
            try:
                objV, eliTime = func(convert_input4MathematicalModel(*inputs),
                                    log_fpath=opath.join(log_dpath, '%s-%s.log' % (prefix, m)),
                                    numThreads=_numThreads, TimeLimit=_TimeLimit)
            except:
                objV, eliTime = -1, -1
            record_res(opath.join(res_dpath, '%s-%s.csv' % (prefix, m)),
                       nt, np, nb, tv, td, m, objV, eliTime)
        #
        # gHeuristic
        #
        m = 'gHeuristic'
        try:
            objV, eliTime = gHeuristic_run(convert_input4MathematicalModel(*inputs),
                                        log_fpath=opath.join(log_dpath, '%s-%s.log' % (prefix, m)))
        except:
            objV, eliTime = -1, -1
        record_res(opath.join(res_dpath, '%s-%s.csv' % (prefix, m)),
                   nt, np, nb, tv, td, m, objV, eliTime)


def record_res(fpath, nt, np, nb, tv, td, m, objV, eliTime):
    with open(fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour',
                  'method', 'objV', 'eliTime']
        writer.writerow(header)
        writer.writerow([nt, np, nb, tv, td, m, objV, eliTime])


# def run(processorID, num_workers=11):
#     for i, fn in enumerate(os.listdir(dpath['problem'])):
#         if not fn.endswith('.pkl'):
#             continue
#         ifpath = opath.join(dpath['problem'], fn)
#         inputs = None
#         with open(ifpath, 'rb') as fp:
#             inputs = pickle.load(fp)
#         if i % num_workers != processorID:
#             continue
#         travel_time, \
#         flows, paths, \
#         tasks, rewards, volumes, \
#         numBundles, thVolume, thDetour = inputs
#         numTasks, numPaths = map(len, [tasks, paths])
#         fn = 'nt%d-np%d-nb%d-tv%d-td%d.csv' % (numTasks, numPaths, numBundles, thVolume, thDetour)
#         ofpath = opath.join(dpath['experiment'], fn)
#         if opath.exists(ofpath):
#             continue
#         with open(ofpath, 'wt') as w_csvfile:
#             writer = csv.writer(w_csvfile, lineterminator='\n')
#             header = ['numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour',
#                       'm_obj', 'h_obj', 'gapR_obj', 'm_time', 'h_time', 'gap_time', ]
#             writer.writerow(header)
#         try:
#             # m_obj, m_time = run_mip_eliSubTour(convert_input4MathematicalModel(*inputs))
#             m_obj, m_time = masterProblem(convert_input4MathematicalModel(*inputs))
#             h_obj, h_time = run_greedyHeuristic(convert_input4greedyHeuristic(*inputs))
#             gap_obj = (m_obj - h_obj) / float(m_obj)
#             gap_time = (m_time - h_time)
#             with open(ofpath, 'a') as w_csvfile:
#                 writer = csv.writer(w_csvfile, lineterminator='\n')
#                 writer.writerow([numTasks, numPaths, numBundles, thVolume, thDetour,
#                                  m_obj, h_obj, gap_obj,
#                                  m_time, h_time, gap_time])
#         except:
#             with open(ofpath, 'a') as w_csvfile:
#                 writer = csv.writer(w_csvfile, lineterminator='\n')
#                 writer.writerow([numTasks, numPaths, numBundles, thVolume, thDetour,
#                                  -1, -1, -1,
#                                  -1, -1, -1])
#
# def single_run(fn):
#     ifpath = opath.join(dpath['problem'], fn)
#     inputs = None
#     with open(ifpath, 'rb') as fp:
#         inputs = pickle.load(fp)
#     points, travel_time, \
#     flows, paths, \
#     tasks, rewards, volumes, \
#     numBundles, thVolume, thDetour = inputs
#     numTasks, numPaths = map(len, [tasks, paths])
#     fn = 'nt%d-np%d-nb%d-tv%d-td%d.csv' % (numTasks, numPaths, numBundles, thVolume, thDetour)
#     ofpath = opath.join(dpath['experiment'], fn)
#     with open(ofpath, 'wt') as w_csvfile:
#         writer = csv.writer(w_csvfile, lineterminator='\n')
#         header = ['numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour',
#                   'm_obj', 'h_obj', 'gapR_obj', 'm_time', 'h_time', 'gap_time', ]
#         writer.writerow(header)
#     try:
#         # m_obj, m_time = run_mip_eliSubTour(convert_input4MathematicalModel(*inputs))
#         m_obj, m_time = masterProblem(convert_input4MathematicalModel(*inputs))
#         h_obj, h_time = run_greedyHeuristic(convert_input4greedyHeuristic(*inputs))
#         gap_obj = (m_obj - h_obj) / float(m_obj)
#         gap_time = (m_time - h_time)
#         with open(ofpath, 'a') as w_csvfile:
#             writer = csv.writer(w_csvfile, lineterminator='\n')
#             writer.writerow([numTasks, numPaths, numBundles, thVolume, thDetour,
#                              m_obj, h_obj, gap_obj,
#                              m_time, h_time, gap_time])
#     except:
#         with open(ofpath, 'a') as w_csvfile:
#             writer = csv.writer(w_csvfile, lineterminator='\n')
#             writer.writerow([numTasks, numPaths, numBundles, thVolume, thDetour,
#                              -1, -1, -1,
#                              -1, -1, -1])
#
#
# def local_run():
#     points, travel_time, \
#     flows, paths, \
#     tasks, rewards, volumes, \
#     numBundles, thVolume, thDetour = ex8()
#     inputs = [points, travel_time,
#                 flows, paths,
#                 tasks, rewards, volumes,
#                 numBundles, thVolume, thDetour]
#     numTasks, numPaths = map(len, [tasks, paths])
#     fn = 'nt%d-np%d-nb%d-tv%d-td%d.csv' % (numTasks, numPaths, numBundles, thVolume, thDetour)
#     ofpath = opath.join(dpath['experiment'], fn)
#     with open(ofpath, 'wt') as w_csvfile:
#         writer = csv.writer(w_csvfile, lineterminator='\n')
#         header = ['numTasks', 'numPaths', 'numBundles', 'thVolume', 'thDetour',
#                   'm_obj', 'h_obj', 'gapR_obj', 'm_time', 'h_time', 'gap_time', ]
#         writer.writerow(header)
#     try:
#         # m_obj, m_time = run_mip_eliSubTour(convert_input4MathematicalModel(*inputs))
#         m_obj, m_time = masterProblem(convert_input4MathematicalModel(*inputs))
#         h_obj, h_time = run_greedyHeuristic(convert_input4greedyHeuristic(*inputs))
#         gap_obj = (m_obj - h_obj) / float(m_obj)
#         gap_time = (m_time - h_time)
#         with open(ofpath, 'a') as w_csvfile:
#             writer = csv.writer(w_csvfile, lineterminator='\n')
#             writer.writerow([numTasks, numPaths, numBundles, thVolume, thDetour,
#                              m_obj, h_obj, gap_obj,
#                              m_time, h_time, gap_time])
#     except:
#         with open(ofpath, 'a') as w_csvfile:
#             writer = csv.writer(w_csvfile, lineterminator='\n')
#             writer.writerow([numTasks, numPaths, numBundles, thVolume, thDetour,
#                              -1, -1, -1,
#                              -1, -1, -1])




if __name__ == '__main__':
    cluster_run(0)

    # run(0, num_workers=8)
    # local_run()
    # single_run('nt6-np20-nb4-tv2-td4.pkl')
    # run(0)
