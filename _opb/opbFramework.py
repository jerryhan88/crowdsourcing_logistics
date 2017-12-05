import time

#
from _opb.opbTree import Tree
#
from _utils.recording import *
from optRouting import run as optR_run
from problems import *


#
def run(probSetting, grbSetting, etcSetting):
    startCpuTime, startWallTime = time.clock(), time.time()
    #
    bB, \
    T, r_i, v_i, _lambda, P, D, N, \
    K, w_k, t_ij, _delta = convert_p2i(*probSetting['problem'])
    #
    K_bundles = {}
    for k in K:
        probSetting['k'] = k
        probSetting['inclusiveC'] = []
        probSetting['exclusiveC'] = []
        probSetting['B'] = None
        probSetting['p_b'] = None
        probSetting['e_bi'] = None
        etcSetting['bbtFile'] = opath.join(etcSetting['bbtDir'], '%s-%d.csv' % (etcSetting['prefix'], k))
        etcSetting['opbLogF'] = opath.join(etcSetting['logDir'], '%s-%d.log' % (etcSetting['prefix'], k))
        grbSetting['LogFile'] = opath.join(etcSetting['logDir'], '%s-%d.log' % (etcSetting['prefix'], k))
        #
        bnbTree = Tree(probSetting, grbSetting, etcSetting)
        bnbTree.startBnP()
        chosenBundles = []
        for i, q_b in enumerate(bnbTree.incumbent.data.res['q_b']):
            if q_b > 0.5:
                chosenBundles.append(bnbTree.incumbent.data.B[i])
        #
        K_bundles[k] = chosenBundles
    #
    bundle_weight = {}
    for k, bundles in K_bundles.items():
        for b in map(tuple, bundles):
            if b in bundle_weight:
                bundle_weight[b] += w_k[k]
            else:
                bundle_weight[b] = w_k[k]
    weight_bundle = [(w * sum(r_i[i] for i in b), b) for b, w in bundle_weight.items()]
    weight_bundle.sort(reverse=True)
    #
    chosenB = []
    assignedT = set()
    for _, b in weight_bundle:
        set_b = set(b)
        if not set_b.intersection(assignedT) and len(chosenB) < bB - 1:
            chosenB.append(b)
            assignedT = assignedT.union(set_b)
    unassignedT = list(set(T).difference(assignedT))
    removedTasks = []
    if sum([v_i[i] for i in unassignedT]) <= _lambda:
        while _lambda < sum([v_i[i] for i in unassignedT]):
            t = unassignedT.pop()
            removedTasks.append(t)
    chosenB.append(unassignedT)
    #
    objVal = 0
    grbSettingOP = {'LogFile': etcSetting['lastLogF'],
                    'Threads': grbSetting['Threads']}
    for b in chosenB:
        br = sum([r_i[i] for i in b])
        p = 0
        for k in K:
            probSetting = {'b': b, 'k': k, 't_ij': t_ij}
            detourTime, route = optR_run(probSetting, grbSettingOP)
            if detourTime <= _delta:
                p += w_k[k] * br
        objVal += p

    endCpuTime, endWallTime = time.clock(), time.time()
    eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
    #
    logContents = '\n\n'
    logContents += 'Summary\n'
    logContents += '\t Cpu Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(startCpuTime)
    logContents += '\t\t End.Time: %s\n' % str(endCpuTime)
    logContents += '\t\t Eli.Time: %f\n' % eliCpuTime
    logContents += '\t Wall Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(startWallTime)
    logContents += '\t\t End.Time: %s\n' % str(endWallTime)
    logContents += '\t\t Eli.Time: %f\n' % eliWallTime
    logContents += '\t ObjV: %.3f\n' % objVal
    logContents += '\t chosen B.: %s\n' % str(chosenB)
    logContents += '\t removed T.: %s\n' % str(removedTasks)
    record_log(etcSetting['lastLogF'], logContents)
    record_res(etcSetting['opbResF'], objVal, None, eliCpuTime, eliWallTime)


def test():
    import os.path as opath
    import os
    import pickle

    ifpath = 'nt10-np12-nb4-tv3-td4.pkl'
    prefix = ifpath[:-len('.pkl')]
    with open(ifpath, 'rb') as fp:
        inputs = pickle.load(fp)


    file_dir = 'z_files'
    if not opath.exists(file_dir):
        os.mkdir(file_dir)
    opbResF = opath.join('z_files','%s-opb.csv' % prefix)
    bbtDir = opath.join('z_files', '%s-opbBBT' % prefix)
    logDir = opath.join('z_files', '%s-opbLog' % prefix)
    lastLogF = opath.join(logDir, '%s-obpLast.log' % prefix)
    for dirPath in [bbtDir, logDir]:
        if not opath.exists(dirPath):
            os.mkdir(dirPath)
    emsgFile, epklFile = [opath.join(file_dir, '%s-%s' % (prefix, fn)) for fn in ['E.txt', 'E.pkl']]
    #
    probSetting = {'problem': inputs}
    grbSetting = {
                  'Threads': 8}
    etcSetting = {
                  'opbResF': opbResF,
                  #
                  'prefix': prefix,
                  'bbtDir': bbtDir,
                  'logDir': logDir,
                  'lastLogF': lastLogF,
                  #
                  'EpklFile': epklFile, 'EmsgFile': emsgFile,
                  }

    run(probSetting, grbSetting, etcSetting)







if __name__ == '__main__':
    test()
    # calTime_test()
