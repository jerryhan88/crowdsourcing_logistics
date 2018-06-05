import os.path as opath
import multiprocessing
import time
import pickle, csv
import treelib
import numpy as np
from itertools import combinations
from heapq import heappush, heappop
from gurobipy import *
#
from _util_logging import write_log, res2file
from RMP import generate_RMP
from BNP_SP import run as SP_run


NUM_CORES = multiprocessing.cpu_count()
LOG_INTER_RESULTS = False
LOGGING_INTERVAL = 20
EPSILON = 0.000000001


def itr2file(fpath, contents=[]):
    if not contents:
        if opath.exists(fpath):
            os.remove(fpath)
        with open(fpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            header = ['nid',
                      'eliCpuTime', 'eliWallTime',
                      'objVal',
                      'eventType',
                      'contents']
            writer.writerow(header)
    else:
        with open(fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow(contents)


def run(prmt, etc=None):
    startCpuTime, startWallTime = time.clock(), time.time()
    if 'TimeLimit' not in etc:
        etc['TimeLimit'] = 1e400
    etc['startTS'] = startCpuTime
    etc['startCpuTime'] = startCpuTime
    etc['startWallTime'] = startWallTime
    itr2file(etc['itrFileCSV'])
    #
    bnp_inputs = {}
    for cond in ['inclusiveC', 'exclusiveC']:
        bnp_inputs[cond] = []
    #
    # Initialize a BNP tree and the root node
    #
    bnpTree = treelib.Tree()
    bnpTree.bestBound, bnpTree.incumbent = None, None
    bnpTree.leafNodes = []
    #
    tag, indentifier = '-', '*'
    bnpTree.create_node(tag, indentifier)
    rootNode = bnpTree.get_node('*')
    rootNode.ni = bnpNode(indentifier, prmt, bnp_inputs, etc)  # ni: node instance
    #
    is_terminated = rootNode.ni.startCG()
    #
    if is_terminated:
        handle_termination(bnpTree, rootNode)
        return None
    #
    branching(bnpTree, rootNode)
    #
    incumProb = bnpTree.incumbent.ni.bnp_inputs
    incumRes = bnpTree.incumbent.ni.res
    bbRes = bnpTree.bestBound.ni.res
    if incumRes['objVal'] != 0:
        BnPgap = abs(incumRes['objVal'] - bbRes['objVal']) / incumRes['objVal']
    else:
        BnPgap = None

    if 'solFileCSV' in etc:
        assert 'solFilePKL' in etc
        assert 'solFileTXT' in etc
        endCpuTime, endWallTime = time.clock(), time.time()
        eliCpuTime = endCpuTime - etc['startCpuTime']
        eliWallTime = endWallTime - etc['startWallTime']
        #
        chosenB = [incumProb['C'][c] for c in range(len(incumProb['C'])) if incumRes['q_c'][c] > 0.5]
        #
        with open(etc['solFileTXT'], 'w') as f:
            logContents = 'End Branch and Pricing from the root\n'
            logContents += 'Summary\n'
            logContents += '\t Cpu Time: %f\n' % eliCpuTime
            logContents += '\t Wall Time: %f\n' % eliWallTime
            logContents += '\t ObjV: %.3f\n' % incumRes['objVal']
            if type(BnPgap) == float:
                logContents += '\t Gap: %.3f\n' % BnPgap
            else:
                assert BnPgap is None
                logContents += '\t Gap: NA\n'
            logContents += '\t chosen B.: %s\n' % str(chosenB)
            f.write(logContents)
            f.write('\n')
        res2file(etc['solFileCSV'], incumRes['objVal'], BnPgap, eliCpuTime, eliWallTime)
        #
        _q_c = {{c: incumRes['q_c'][c] for c in range(len(incumProb['C']))}}
        sol = {
            'C': incumProb['C'], 'p_c': incumProb['p_c'], 'e_ci': incumProb['e_ci'],
            'q_c': _q_c
        }
        with open(etc['solFilePKL'], 'wb') as fp:
            pickle.dump(sol, fp)


def branching(bnpTree, curNode):
    etc = curNode.ni.etc
    bestBound, incumbent = bnpTree.bestBound, bnpTree.incumbent
    #
    logContents = 'Try Branching; bnbNode(%s)\n' % curNode.identifier
    write_log(etc['logFile'], logContents)
    itr2file(etc['itrFileCSV'], [curNode.identifier,
                                 None, None,
                                 curNode.ni.res['objVal'],
                                 'TB', None])
    #
    curProb = curNode.ni.bnp_inputs
    curRes = curNode.ni.res
    if bestBound is None:
        bnpTree.bestBound = curNode
    # Pruning
    if incumbent:
        incumRes = incumbent.ni.res
        if curRes['objVal'] < incumRes['objVal']:
            logContents = 'bnbNode(%s) was pruned\n' % curNode.identifier
            write_log(etc['logFile'], logContents)
            itr2file(etc['itrFileCSV'], [curNode.identifier,
                                         None, None,
                                         None,
                                         'PR', None])
            branching_dfs_lcp(bnpTree)
            return None
    #
    fracOrdered = []
    for c, q_c_var in enumerate(curRes['q_c']):
        if len(curProb['C'][c]) == 1:
            continue
        deviationFH = abs(q_c_var - 0.5)
        fracOrdered.append((deviationFH, curProb['C'][c]))
    fracOrdered.sort()
    mostFracVal, _ = fracOrdered[0]
    if abs(mostFracVal - 0.5) < EPSILON:
        #
        # Integral solution
        #
        logContents = 'Found a integral solution\n'
        write_log(etc['logFile'], logContents)
        itr2file(etc['itrFileCSV'], [curNode.identifier,
                                     None, None,
                                     None,
                                     'INT',
                                     None])
        if incumbent is None:
            logContents = 'The first incumbent, bnbNode(%s)\n' % curNode.identifier
            write_log(etc['logFile'], logContents)
            itr2file(etc['itrFileCSV'], [curNode.identifier,
                                         None, None,
                                         None,
                                         'INT',
                                         'First'])
            bnpTree.incumbent = curNode
        else:
            incumRes = incumbent.ni.res
            if incumRes['objVal'] < curRes['objVal']:
                logContents = 'The incumbent was changed\n'
                logContents += 'bnbNode(%s) -> bnbNode(%s)\n' % (incumbent.identifier, curNode.identifier)
                write_log(etc['logFile'], logContents)
                itr2file(etc['itrFileCSV'], [curNode.identifier,
                                             None, None,
                                             None,
                                             'IC',
                                             '%s -> %s' % (incumbent.identifier, curNode.identifier)])
                bnpTree.incumbent = curNode
            else:
                logContents = 'No change about the incumbent\n'
                write_log(etc['logFile'], logContents)
                itr2file(etc['itrFileCSV'], [curNode.identifier,
                                             None, None,
                                             None,
                                             'NI',
                                             None])
        branching_dfs_lcp(bnpTree)
    else:
        #
        # Choose branching criteria
        #
        pTag, pIdentifier = curNode.tag, curNode.identifier
        pOri_inputs, pBNP_inputs = curNode.ni.ori_inputs, curNode.ni.bnp_inputs
        chosenTaskPairs = set(pBNP_inputs['inclusiveC']).union(pBNP_inputs['exclusiveC'])
        for _, candiBundle in fracOrdered:
            foundPairs = False
            for i0, i1 in combinations(candiBundle, 2):
                if (i0, i1) not in chosenTaskPairs:
                    foundPairs = True
                    break
            if foundPairs:
                break
        else:
            logContents = 'No suitable pairs\n'
            write_log(etc['logFile'], logContents)
            assert False
        logContents = 'Start Branching; bnbNode(%s)\n' % curNode.identifier
        logContents += '\t All bundles %s\n' % str(curProb['C'])
        logContents += '\t Chosen bundle %s\n' % str(candiBundle)
        logContents += '\t Chosen tasks %s\n' % str((i0, i1))
        write_log(etc['logFile'], logContents)
        #
        # Left child
        #
        lTag, lIndentifier = pTag + 'L', pIdentifier + '0'
        lBNP_inputs = duplicate_BNP_inputs(pBNP_inputs)
        lBNP_inputs['inclusiveC'] += [(i0, i1)]
        bnpTree.create_node(lTag, lIndentifier, parent=pIdentifier)
        lcNode = bnpTree.get_node(lIndentifier)
        lcNode.ni = bnpNode(lIndentifier, pOri_inputs, lBNP_inputs, etc)
        is_terminated = lcNode.ni.startCG()
        if is_terminated:
            handle_termination(bnpTree, lcNode)
            return None
        heappush(bnpTree.leafNodes, (-(bnpTree.depth(lcNode) + 0.1), lcNode))
        #
        # Right child
        #
        rTag, rIndentifier = pTag + 'R', pIdentifier + '1'
        rBNP_inputs = duplicate_BNP_inputs(pBNP_inputs)
        rBNP_inputs['exclusiveC'] += [(i0, i1)]
        bnpTree.create_node(rTag, rIndentifier, parent=pIdentifier)
        rcNode = bnpTree.get_node(rIndentifier)
        rcNode.ni = bnpNode(rIndentifier, pOri_inputs, rBNP_inputs, etc)
        is_terminated = rcNode.ni.startCG()
        if is_terminated:
            handle_termination(bnpTree, rcNode)
            return None
        heappush(bnpTree.leafNodes, (-bnpTree.depth(rcNode), rcNode))
        #
        if bnpTree.bestBound == curNode:
            #
            # Update the best bound
            #
            bestBoundVal, bestBoundNode = -1e400, None
            for nid, candiNode in bnpTree.nodes.items():
                if candiNode.is_leaf() and candiNode.ni.res and bestBoundVal < candiNode.ni.res['objVal']:
                    bestBoundVal, bestBoundNode = candiNode.ni.res['objVal'], candiNode
            bnpTree.bestBound = bestBoundNode
        #
        branching_dfs_lcp(bnpTree)


def branching_dfs_lcp(bnpTree):
    # Depth First Search and left child priority
    if bnpTree.leafNodes:
        _, nextNode = heappop(bnpTree.leafNodes)
        branching(bnpTree, nextNode)
    else:
        bnpTree.bestBound = bnpTree.incumbent

def duplicate_BNP_inputs(pBNP_inputs):
    cBNP_inputs = {}
    cBNP_inputs['C'] = [c[:] for c in pBNP_inputs['C']]
    cBNP_inputs['p_c'] = pBNP_inputs['p_c'][:]
    cBNP_inputs['e_ci'] = [vec[:] for vec in pBNP_inputs['e_ci']]
    cBNP_inputs['inclusiveC'] = pBNP_inputs['inclusiveC'][:]
    cBNP_inputs['exclusiveC'] = pBNP_inputs['exclusiveC'][:]
    return cBNP_inputs


def handle_termination(bnpTree, curNode):
    etcSetting = curNode.ni.etcSetting
    logContents = 'BNP model reach to the time limit, while solving node %s\n' % curNode.identifier
    write_log(etcSetting['logFile'], logContents)
    endCpuTimeBnP, endWallTimeBnP = time.clock(), time.time()
    eliCpuTimeBnP = endCpuTimeBnP - etcSetting['startCpuTimeBnP']
    eliWallTimeBnP = endWallTimeBnP - etcSetting['startWallTimeBnP']
    if bnpTree.bestBound and bnpTree.incument:
        incumRes = bnpTree.incumbent.ni.res
        bbRes = bnpTree.bestBound.ni.res
        BnPgap = abs(incumRes['objVal'] - bbRes['objVal']) / incumRes['objVal']
        res2file(etcSetting['ResFile'], incumRes['objVal'], BnPgap, eliCpuTimeBnP, eliWallTimeBnP)
    else:
        res2file(etcSetting['ResFile'], -1, None, eliCpuTimeBnP, eliWallTimeBnP)


class bnpNode(object):
    def __init__(self, nid, prmt, bnp_inputs, etc):
        self.nid = nid
        self.prmt, self.bnp_inputs = prmt, bnp_inputs
        self.etc = etc
        if self.nid == '*':
            for cond in ['inclusiveC', 'exclusiveC']:
                assert not self.bnp_inputs[cond]
            for key in ['C', 'p_c', 'e_ci']:
                assert key not in self.bnp_inputs
            C, p_c, e_ci = self.gen_initColumns()
            self.bnp_inputs['C'] = C
            self.bnp_inputs['p_c'] = p_c
            self.bnp_inputs['e_ci'] = e_ci
        self.res = {}

    def __repr__(self):
        return 'bnbNode(%s)' % self.nid

    def gen_initColumns(self):
        #
        # Generate initial singleton bundles
        #
        T = self.prmt['T']
        C, p_c, e_ci = [], [], []
        for i in T:
            Ts = [i]
            C.append(Ts)
            #
            p_c.append(0)
            #
            vec = [0 for _ in range(len(T))]
            vec[i] = 1
            e_ci.append(vec)
        #
        return C, p_c, e_ci

    def startCG(self):
        startCpuTimeCG, startWallTimeCG = time.clock(), time.time()
        logContents = 'Start column generation of bnbNode(%s)\n' % self.nid
        write_log(self.etc['logFile'], logContents)
        #
        T = self.prmt['T']
        C = self.bnp_inputs['C']
        RMP, q_c, taskAC, numBC = generate_RMP(self.prmt, self.bnp_inputs)
        #
        counter, is_terminated = 0, False
        while True:
            if len(C) == len(T) ** 2 - 1:
                break
            LRMP = RMP.relax()
            LRMP.setParam('Threads', NUM_CORES)
            LRMP.setParam('OutputFlag', False)
            LRMP.optimize()
            if LRMP.status == GRB.Status.INFEASIBLE:
                logContents = 'Relaxed model is infeasible!!\n'
                logContents += 'No solution!\n'
                write_log(self.etc['logFile'], logContents)
                #
                LRMP.write('%s.lp' % self.prmt['problemName'])
                LRMP.computeIIS()
                LRMP.write('%s.ilp' % self.prmt['problemName'])
                assert False
            #
            counter += 1
            pi_i = [LRMP.getConstrByName("taskAC[%d]" % i).Pi for i in T]
            mu = LRMP.getConstrByName("numBC").Pi
            self.bnp_inputs['pi_i'] = pi_i
            self.bnp_inputs['mu'] = mu
            #
            if LOG_INTER_RESULTS:
                logContents = 'Start %dth iteration\n' % counter
                logContents += '\t Columns\n'
                logContents += '\t\t # of columns %d\n' % len(self.bnp_inputs['C'])
                logContents += '\t\t %s\n' % str(self.bnp_inputs['C'])
                logContents += '\t\t %s\n' % str(['%.2f' % v for v in self.bnp_inputs['p_c']])
                logContents += '\t Relaxed objVal\n'
                logContents += '\t\t z: %.2f\n' % LRMP.objVal
                logContents += '\t\t RC: %s\n' % str(
                    ['%.2f' % LRMP.getVarByName("q[%d]" % c).RC for c in range(len(C))])
                logContents += '\t Dual V\n'
                logContents += '\t\t Pi: %s\n' % str(['%.2f' % v for v in pi_i])
                logContents += '\t\t mu: %.2f\n' % mu
                write_log(self.etc['logFile'], logContents)
            #
            objV_newC = SP_run(self.prmt, self.bnp_inputs, self.etc)
            if objV_newC == 'terminated':
                logContents = '%dth iteration of bnbNode(%s)\n' % (counter, self.nid)
                logContents += 'Terminated because of the time limit!\n'
                write_log(self.etc['logFile'], logContents)
                is_terminated = True
                break
            elif objV_newC is None:
                logContents = '%dth iteration of bnbNode(%s)\n' % (counter, self.nid)
                logContents += 'No solution!\n'
                write_log(self.etc['logFile'], logContents)
                break
            eliCpuTimeP, eliWallTimeP = time.clock() - self.etc['startCpuTime'], time.time() - self.etc['startWallTime']
            itr2file(self.etc['itrFileCSV'], [self.nid,
                                         '%.2f' % eliCpuTimeP, '%.2f' % eliWallTimeP,
                                         None,
                                         'S',
                                         {'numIter': counter, 'objV_newC': objV_newC}])
            #
            objV, newC = objV_newC
            if objV < 0:
                logContents = 'The reduced cost of the generated column is a negative number\n'
                write_log(self.etc['logFile'], logContents)
                break
            else:
                logContents = '\t New column\n'
                logContents += '\t\t Tasks %s\n' % str(newC)
                logContents += '\t\t red. C. %.3f\n' % objV
                write_log(self.etc['logFile'], logContents)
                vec = [0 for _ in range(len(T))]
                for i in newC:
                    vec[i] = 1
                p = objV + (np.array(vec) * np.array(pi_i)).sum() + mu
                C, p_c, e_ci = list(map(self.bnp_inputs.get, ['C', 'p_c', 'e_ci']))
                e_ci.append(vec)
                p_c.append(p)
                #
                col = Column()
                for i in range(len(T)):
                    if e_ci[len(C)][i] > 0:
                        col.addTerms(e_ci[len(C)][i], taskAC[i])
                col.addTerms(1, numBC)
                #
                q_c[len(C)] = RMP.addVar(obj=p_c[len(C)], vtype=GRB.BINARY, name="q[%d]" % len(C), column=col)
                C.append(newC)
                RMP.update()
        #
        if not is_terminated:
            LRMP = RMP.relax()
            LRMP.setParam('Threads', NUM_CORES)
            LRMP.setParam('OutputFlag', False)
            LRMP.optimize()
            C, p_c, e_ci = list(map(self.bnp_inputs.get, ['C', 'p_c', 'e_ci']))
            q_c = [LRMP.getVarByName("q[%d]" % c).x for c in range(len(C))]
            #
            endCpuTimeCG, endWallTimeCG = time.clock(), time.time()
            eliCpuTimeCG, eliWallTimeCG = endCpuTimeCG - startCpuTimeCG, endWallTimeCG - startWallTimeCG
            chosenC = [(C[c], '%.2f' % q_c[c]) for c in range(len(C)) if q_c[c] > 0]
            #
            logContents = 'Column generation summary\n'
            logContents += '\t Cpu Time: %f\n' % eliCpuTimeCG
            logContents += '\t Wall Time: %f\n' % eliWallTimeCG
            logContents += '\t ObjV: %.3f\n' % LRMP.objVal
            logContents += '\t chosen B.: %s\n' % str(chosenC)
            write_log(self.etc['logFile'], logContents)
            #
            self.bnp_inputs['C'] = C
            self.bnp_inputs['p_c'] = p_c
            self.bnp_inputs['e_ci'] = e_ci
            #
            self.res['objVal'] = LRMP.objVal
            self.res['q_c'] = q_c
            #
            itr2file(self.etc['itrFileCSV'], [self.nid,
                                         '%.2f' % eliCpuTimeCG, '%.2f' % eliWallTimeCG,
                                         self.res['objVal'],
                                         'M',
                                        {'objVal': LRMP.objVal,
                                         'inclusiveC': str(self.bnp_inputs['inclusiveC']),
                                         'exclusiveC': str(self.bnp_inputs['exclusiveC'])}])
        #
        return is_terminated


if __name__ == '__main__':
    from mrtScenario import mrtS1, mrtS2
    #
    prmt = mrtS1()
    # prmt = mrtS2()
    problemName = prmt['problemName']
    #
    etc = {'solFilePKL': opath.join('_temp', 'sol_%s_BNP.pkl' % problemName),
           'solFileCSV': opath.join('_temp', 'sol_%s_BNP.csv' % problemName),
           'solFileTXT': opath.join('_temp', 'sol_%s_BNP.txt' % problemName),
           'logFile': opath.join('_temp', '%s_BNP.log' % problemName),
           'itrFileCSV': opath.join('_temp', '%s_itrBNP.csv' % problemName),
           }
    #
    run(prmt, etc)
