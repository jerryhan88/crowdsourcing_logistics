import os.path as opath
import pickle

#
from problems import *

#
prob_dir = opath.join('z_files', '_problems')


def ex_test():
    from exactMM import run as exactMM_run
    fn = 'nt05-np12-nb2-tv3-td4.pkl'
    prefix = fn[:-len('.pkl')]
    #
    ifpath = opath.join(prob_dir, fn)
    with open(ifpath, 'rb') as fp:
        problem = pickle.load(fp)
    exLogF = opath.join('z_files', '%s-%s.log' % (prefix, 'ex'))
    exResF = opath.join('z_files', '%s-%s.csv' % (prefix, 'ex'))
    #
    probSetting = {'problem': problem}
    grbSetting = {'LogFile': exLogF,
                  'Threads': 8,}
    etcSetting = {'exLogF': exLogF,
                  'exResF': exResF
                  }

    exactMM_run(probSetting, grbSetting, etcSetting)


def ghM_test():
    prefix = 'gh_mBundling'
    pyx_fn, c_fn = '%s.pyx' % prefix, '%s.c' % prefix
    if opath.exists(c_fn):
        if opath.getctime(c_fn) < opath.getmtime(pyx_fn):
            from setup import cythonize;
            cythonize(prefix)
    else:
        from setup import cythonize;
        cythonize(prefix)
    from gh_mBundling import run as ghM_run
    fn = 'nt05-np12-nb2-tv3-td4.pkl'
    prefix = fn[:-len('.pkl')]
    #
    ifpath = opath.join(prob_dir, fn)
    with open(ifpath, 'rb') as fp:
        problem = pickle.load(fp)

    objV, B = ghM_run(problem)
    print(objV, B)


def ghS_test():
    from gh_sBundling import run as ghS_run
    fn = 'nt05-np12-nb2-tv3-td4.pkl'
    prefix = fn[:-len('.pkl')]
    #
    ifpath = opath.join(prob_dir, fn)
    with open(ifpath, 'rb') as fp:
        problem = pickle.load(fp)
    pi_i = [1.179, 0.125, 0.482, 0.857, 0.0]
    mu = 3.429
    inclusiveC = [[3, 1]]
    exclusiveC = [[1, 2]]
    #
    inputs = convert_p2i(*problem)
    ghS_inputs = {}
    for k in ['T', 'r_i', 'v_i', '_lambda',
              'N',
              'K', 'w_k',
              't_ij', '_delta',
              '_N']:
        ghS_inputs[k] = inputs[k]
    ghS_inputs['pi_i'] = pi_i
    ghS_inputs['mu'] = mu
    ghS_inputs['inclusiveC'] = inclusiveC
    ghS_inputs['exclusiveC'] = exclusiveC
    #
    rc, dvs = ghS_run(ghS_inputs)
    print(rc, dvs)



def or_test():
    from gh_mBundling import run as ghM_run
    from optRouting import run as or_run
    fn = 'nt05-np12-nb2-tv3-td4.pkl'
    prefix = fn[:-len('.pkl')]
    #
    ifpath = opath.join(prob_dir, fn)
    with open(ifpath, 'rb') as fp:
        problem = pickle.load(fp)
    objV, B = ghM_run(problem)
    #
    inputs = convert_p2i(*problem)
    bB = inputs['bB']
    T, r_i, v_i, _lambda = list(map(inputs.get, ['T', 'r_i', 'v_i', '_lambda']))
    K, w_k = list(map(inputs.get, ['K', 'w_k']))
    t_ij, _delta = list(map(inputs.get, ['t_ij', '_delta']))

    logContents = ''
    grbSettingOP = {}
    ps = 0
    for b in B:
        br = sum([r_i[i] for i in b])
        logContents += '%s (%d) \n' % (str(b), br)
        p = 0
        for k, w in enumerate(w_k):
            probSetting = {'b': b, 'k': k, 't_ij': t_ij}
            detourTime, route = or_run(probSetting, grbSettingOP)
            if detourTime <= _delta:
                p += w * br
                logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w, detourTime, 1, str(route))
            else:
                logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w, detourTime, 0, str(route))
        logContents += '\t\t\t\t\t\t %.3f\n' % p
        ps += p

    print(logContents)
    print(ps)


def gBNP_test():
    from bnpTree import BnPTree
    fn = 'nt05-np12-nb2-tv3-td4.pkl'
    prefix = fn[:-len('.pkl')]
    #
    ifpath = opath.join(prob_dir, fn)
    with open(ifpath, 'rb') as fp:
        problem = pickle.load(fp)

    ghLogF, orLogF, cgLogF, bnpLogF = [opath.join('z_files', '%s-%s' % (prefix, fn)) for fn in
                                       ['gh.log', 'or.log', 'cg.log', 'bnp.log']]
    ghResF, orResF, cgResF, bnpResF = [opath.join('z_files', '%s-%s' % (prefix, fn)) for fn in
                                       ['gh.csv', 'or.csv', 'cg.csv', 'bnp.csv']]
    bptFile = opath.join('z_files', '%s-%s' % (prefix, 'Hbpt.csv'))
    emsgFile, epklFile = 'E.txt', 'E.pkl'
    #
    probSetting = {'problem': problem,
                   'inclusiveC': [], 'exclusiveC': []}
    grbSetting = {'LogFile': bnpLogF,
                  'Threads': 8}
    etcSetting = {'ghLogF': ghLogF, 'orLogF': orLogF, 'cgLogF': cgLogF, 'bnpLogF': bnpLogF,
                  #
                  'ghResF': ghResF, 'orResF': orResF, 'cgResF': cgResF, 'bnpResF': bnpResF,
                  #
                  'bptFile': bptFile,
                  #
                  'EpklFile': epklFile, 'EmsgFile': emsgFile,
                  'use_ghS': False
                  }
    bnbTree = BnPTree(probSetting, grbSetting, etcSetting)
    bnbTree.startBnP()


def ghS_BNP_test():
    from bnpTree import BnPTree
    fn = 'nt100-np12-nb40-tv3-td4.pkl'
    prefix = fn[:-len('.pkl')]
    #
    ifpath = opath.join(prob_dir, fn)
    with open(ifpath, 'rb') as fp:
        problem = pickle.load(fp)

    ghLogF, orLogF, cgLogF, bnpLogF = [opath.join('z_files', '%s-%s' % (prefix, fn)) for fn in
                                       ['gh.log', 'or.log', 'cg.log', 'bnp.log']]
    ghResF, orResF, cgResF, bnpResF = [opath.join('z_files', '%s-%s' % (prefix, fn)) for fn in
                                       ['gh.csv', 'or.csv', 'cg.csv', 'bnp.csv']]
    bptFile = opath.join('z_files', '%s-%s' % (prefix, 'Hbpt.csv'))
    emsgFile, epklFile = 'E.txt', 'E.pkl'
    #
    probSetting = {'problem': problem,
                   'inclusiveC': [], 'exclusiveC': []}
    grbSetting = {'LogFile': bnpLogF,
                  'Threads': 8,
                  'Method': 1}
    etcSetting = {'ghLogF': ghLogF, 'orLogF': orLogF, 'cgLogF': cgLogF, 'bnpLogF': bnpLogF,
                  #
                  'ghResF': ghResF, 'orResF': orResF, 'cgResF': cgResF, 'bnpResF': bnpResF,
                  #
                  'bptFile': bptFile,
                  #
                  'EpklFile': epklFile, 'EmsgFile': emsgFile,
                  'use_ghS': True
                  }
    bnbTree = BnPTree(probSetting, grbSetting, etcSetting)
    bnbTree.startBnP()


if __name__ == '__main__':
    # ex_test()
    # ghM_test()
    # or_test()
    # gBNP_test()
    # ghS_test()
    ghS_BNP_test()
