import os.path as opath
from gurobipy import *
#
# prefix = 'gh_sBundling'
# pyx_fn, c_fn = '%s.pyx' % prefix, '%s.c' % prefix
# if opath.exists(c_fn):
#     if opath.getctime(c_fn) < opath.getmtime(pyx_fn):
#         from setup import cythonize; cythonize(prefix)
# else:
#     from setup import cythonize; cythonize(prefix)
from gh_sBundling import run as ghS_run
#
from _utils.recording import *
from _utils.mm_utils import *


# PRICING_TIME_LIMIT = 60 * 10
PRICING_TIME_LIMIT = 60 * 2


def run(counter, inputs, grbSetting, use_ghS=False):
    pi_i, mu, B = [inputs.get(k) for k in ['pi_i', 'mu', 'B']]
    inclusiveC, exclusiveC = [inputs.get(k) for k in ['inclusiveC', 'exclusiveC']]
    #
    T, r_i, v_i, _lambda = [inputs.get(k) for k in ['T', 'r_i', 'v_i', '_lambda']]
    P, D, N = [inputs.get(k) for k in ['P', 'D', 'N']]
    K, w_k, t_ij, _delta = [inputs.get(k) for k in ['K', 'w_k', 't_ij', '_delta']]
    #
    rc, dvs = ghS_run(inputs)
    logContents = '\n\n'
    logContents += 'Initial solution\n'
    logContents += '\t rc: %.2f \n' % rc
    logContents += '\t b: %s \n' % str([i for i in T if dvs['_z_i'][i] > 0.5])
    record_log(grbSetting['LogFile'], logContents)
    #
    bestSols = []
    bestSols.append((rc, [i for i in T if dvs['_z_i'][i] > 0.5]))

    return bestSols



if __name__ == '__main__':
    pass