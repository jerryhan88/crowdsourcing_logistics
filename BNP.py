from sys import exit
import datetime, time
import treelib
import numpy as np
from itertools import combinations
from heapq import heappush, heappop
from gurobipy import *
#
from _util import log2file, bpt2file, res2file
from _util import set_grbSettings
from RMP import generate_RMP
from SP import run as SP_run
from PD import run as PD_run
from problems import *


EPSILON = 0.000000001


def run(probSetting, etcSetting, grbSetting):
    startCpuTimeBnP, startWallTimeBnP = time.clock(), time.time()
    if 'TimeLimit' not in etcSetting:
        etcSetting['TimeLimit'] = 1e400
    etcSetting['startTS'] = startCpuTimeBnP
    etcSetting['startCpuTimeBnP'] = startCpuTimeBnP
    etcSetting['startWallTimeBnP'] = startWallTimeBnP
    bpt2file(etcSetting['bptFile'])
    assert 'problem' in probSetting
    probSetting['inputs'] = convert_p2i(*probSetting['problem'])
    for cond in ['inclusiveC', 'exclusiveC']:
        probSetting[cond] = []
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
    rootNode.ni = bnpNode(indentifier, probSetting, etcSetting, grbSetting)
    #
    logContents = '\n\n'
    logContents += '======================================================================================\n'
    logContents += '%s\n' % str(datetime.datetime.now())
    logContents += 'Start Branch and Pricing from the root\n'
    logContents += '======================================================================================\n'
    log2file(etcSetting['LogFile'], logContents)
    is_terminated = rootNode.ni.startCG()
    #
    if is_terminated:
        handle_termination(bnpTree, rootNode)
        return None
    #
    branching(bnpTree, rootNode)
    #
    incumProb = bnpTree.incumbent.ni.probSetting
    incumRes = bnpTree.incumbent.ni.res
    bbRes = bnpTree.bestBound.ni.res
    if incumRes['objVal'] != 0:
        BnPgap = abs(incumRes['objVal'] - bbRes['objVal']) / incumRes['objVal']
    else:
        BnPgap = None
    chosenB = [incumProb['C'][c] for c in range(len(incumProb['C'])) if incumRes['q_c'][c] > 0.5]
    endCpuTimeBnP, endWallTimeBnP = time.clock(), time.time()
    eliCpuTimeBnP, eliWallTimeBnP = endCpuTimeBnP - startCpuTimeBnP, endWallTimeBnP - startWallTimeBnP
    #
    logContents = '\n\n'
    logContents += '======================================================================================\n'
    logContents += '%s\n' % str(datetime.datetime.now())
    logContents += 'End Branch and Pricing from the root\n'
    logContents += 'Summary\n'
    logContents += '\t Cpu Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeBnP)
    logContents += '\t\t End.Time: %s\n' % str(endCpuTimeBnP)
    logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeBnP
    logContents += '\t Wall Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeBnP)
    logContents += '\t\t End.Time: %s\n' % str(endWallTimeBnP)
    logContents += '\t\t Eli.Time: %f\n' % eliWallTimeBnP
    logContents += '\t ObjV: %.3f\n' % incumRes['objVal']
    if type(BnPgap) == float:
        logContents += '\t Gap: %.3f\n' % BnPgap
    else:
        assert BnPgap is None
        logContents += '\t Gap: NA\n'
    logContents += '\t chosen B.: %s\n' % str(chosenB)
    logContents += '======================================================================================\n'
    log2file(etcSetting['LogFile'], logContents)
    res2file(etcSetting['ResFile'], incumRes['objVal'], BnPgap, eliCpuTimeBnP, eliWallTimeBnP)


def branching(bnpTree, curNode):
    etcSetting, grbSetting = curNode.ni.etcSetting, curNode.ni.grbSetting
    bestBound, incumbent = bnpTree.bestBound, bnpTree.incumbent
    #
    now = datetime.datetime.now()
    logContents = '\n\n'
    logContents += '======================================================================================\n'
    logContents += '%s\n' % str(now)
    logContents += 'Try Branching; bnbNode(%s)\n' % curNode.identifier
    logContents += '======================================================================================\n'
    log2file(etcSetting['LogFile'], logContents)
    bpt2file(etcSetting['bptFile'], [curNode.identifier, now, None, None, 'TB', None])
    #
    curProb = curNode.ni.probSetting
    curRes = curNode.ni.res
    if bestBound is None:
        bnpTree.bestBound = curNode
    # Pruning
    if incumbent:
        incumRes = incumbent.ni.res
        if curRes['objVal'] < incumRes['objVal']:
            now = datetime.datetime.now()
            logContents = '\n\n'
            logContents += '======================================================================================\n'
            logContents += '%s\n' % str(now)
            logContents += 'bnbNode(%s) was pruned\n' % curNode.identifier
            logContents += '======================================================================================\n'
            log2file(etcSetting['LogFile'], logContents)
            bpt2file(etcSetting['bptFile'], [curNode.identifier, now, None, None, 'PR', None])
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
        now = datetime.datetime.now()
        logContents = '\n\n'
        logContents += '======================================================================================\n'
        logContents += '%s\n' % str(now)
        logContents += 'Found a integral solution\n'
        logContents += '======================================================================================\n'
        log2file(etcSetting['LogFile'], logContents)
        bpt2file(etcSetting['bptFile'], [curNode.identifier, now, None, None, 'INT', None])
        if incumbent is None:
            logContents = '\n\n'
            logContents += '======================================================================================\n'
            logContents += 'The first incumbent, bnbNode(%s)\n' % curNode.identifier
            logContents += '======================================================================================\n'
            log2file(etcSetting['LogFile'], logContents)
            bpt2file(etcSetting['bptFile'], [curNode.identifier, now, None, None, 'IC', 'First'])
            bnpTree.incumbent = curNode
        else:
            incumRes = incumbent.ni.res
            if incumRes['objVal'] < curRes['objVal']:
                logContents = '\n\n'
                logContents += '======================================================================================\n'
                logContents += 'The incumbent was changed\n'
                logContents += 'bnbNode(%s) -> bnbNode(%s)\n' % (incumbent.identifier, curNode.identifier)
                logContents += '======================================================================================\n'
                log2file(etcSetting['LogFile'], logContents)
                bpt2file(etcSetting['bptFile'], [curNode.identifier,
                                                        now,
                                                        None, None,
                                                        'IC',
                                                        '%s -> %s' % (incumbent.identifier, curNode.identifier)])
                bnpTree.incumbent = curNode
            else:
                logContents = '\n\n'
                logContents += '======================================================================================\n'
                logContents += 'No change about the incumbent\n'
                logContents += '======================================================================================\n'
                log2file(etcSetting['LogFile'], logContents)
                bpt2file(etcSetting['bptFile'], [curNode.identifier,
                                                        now,
                                                        None, None,
                                                        'NI',
                                                        None])
        branching_dfs_lcp(bnpTree)
    else:
        #
        # Choose branching criteria
        #
        pTag, pIdentifier = curNode.tag, curNode.identifier
        pProbSetting = curNode.ni.probSetting
        chosenTaskPairs = set(pProbSetting['inclusiveC']).union(pProbSetting['exclusiveC'])
        for _, candiBundle in fracOrdered:
            foundPairs = False
            for i0, i1 in combinations(candiBundle, 2):
                if (i0, i1) not in chosenTaskPairs:
                    foundPairs = True
                    break
            if foundPairs:
                break
        else:
            logContents = '\n\n'
            logContents += '======================================================================================\n'
            logContents += '%s\n' % str(datetime.datetime.now())
            logContents += 'No suitable pairs\n'
            logContents += '======================================================================================\n'
            log2file(etcSetting['LogFile'], logContents)
            assert False
        logContents = '\n\n'
        logContents += '======================================================================================\n'
        logContents += '%s\n' % str(datetime.datetime.now())
        logContents += 'Start Branching; bnbNode(%s)\n' % curNode.identifier
        logContents += '\t All bundles %s\n' % str(curProb['C'])
        logContents += '\t Chosen bundle %s\n' % str(candiBundle)
        logContents += '\t Chosen tasks %s\n' % str((i0, i1))
        logContents += '======================================================================================\n'
        log2file(etcSetting['LogFile'], logContents)
        #
        # Left child
        #
        lTag, lIndentifier = pTag + 'L', pIdentifier + '0'
        lProbSetting = duplicate_probSetting(pProbSetting)
        lProbSetting['inclusiveC'] += [(i0, i1)]
        bnpTree.create_node(lTag, lIndentifier, parent=pIdentifier)
        lcNode = bnpTree.get_node(lIndentifier)
        lcNode.ni = bnpNode(lIndentifier, lProbSetting, etcSetting, grbSetting)
        is_terminated = lcNode.ni.startCG()
        if is_terminated:
            handle_termination(bnpTree, lcNode)
            return None
        heappush(bnpTree.leafNodes, (-(bnpTree.depth(lcNode) + 0.1), lcNode))
        #
        # Right child
        #
        rTag, rIndentifier = pTag + 'R', pIdentifier + '1'
        rProbSetting = duplicate_probSetting(pProbSetting)
        rProbSetting['exclusiveC'] += [(i0, i1)]
        bnpTree.create_node(rTag, rIndentifier, parent=pIdentifier)
        rcNode = bnpTree.get_node(rIndentifier)
        rcNode.ni = bnpNode(rIndentifier, rProbSetting, etcSetting, grbSetting)
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

def duplicate_probSetting(pProbSetting):
    cProbSetting = {'inputs': pProbSetting['inputs']}
    cProbSetting['C'] = [bc[:] for bc in pProbSetting['C']]
    cProbSetting['p_c'] = pProbSetting['p_c'][:]
    cProbSetting['e_ci'] = [vec[:] for vec in pProbSetting['e_ci']]
    cProbSetting['inclusiveC'] = pProbSetting['inclusiveC'][:]
    cProbSetting['exclusiveC'] = pProbSetting['exclusiveC'][:]
    return cProbSetting


def handle_termination(bnpTree, curNode):
    etcSetting = curNode.ni.etcSetting
    logContents = '\n\n'
    logContents += '======================================================================================\n'
    logContents += '%s\n' % str(datetime.datetime.now())
    logContents += 'BNP model reach to the time limit, while solving node %s\n' % curNode.identifier
    logContents += '======================================================================================\n'
    log2file(etcSetting['LogFile'], logContents)
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
    def __init__(self, nid, probSetting, etcSetting, grbSetting):
        self.nid = nid
        self.probSetting, self.etcSetting, self.grbSetting = probSetting, etcSetting, grbSetting
        if self.nid == '*':
            for cond in ['inclusiveC', 'exclusiveC']:
                assert not self.probSetting[cond]
            for key in ['C', 'p_c', 'e_ci']:
                assert key not in self.probSetting
            C, p_c, e_ci = self.gen_initColumns()
            self.probSetting['C'] = C
            self.probSetting['p_c'] = p_c
            self.probSetting['e_ci'] = e_ci
        self.res = {}

    def __repr__(self):
        return 'bnbNode(%s)' % self.nid

    def gen_initColumns(self):
        #
        # Generate initial singleton bundles
        #
        inputs = self.probSetting['inputs']
        T, r_i = list(map(inputs.get, ['T', 'r_i']))
        K, w_k = list(map(inputs.get, ['K', 'w_k']))
        t_ij, _delta = list(map(inputs.get, ['t_ij', '_delta']))
        C, p_c, e_ci = [], [], []
        for i in T:
            bc = [i]
            C.append(bc)
            #
            br = sum([r_i[i] for i in bc])
            p = 0
            for k in K:
                probSetting = {'bc': bc, 'k': k, 't_ij': t_ij}
                detourTime, route = PD_run(probSetting, self.grbSetting)
                if detourTime <= _delta:
                    p += w_k[k] * br
            p_c.append(p)
            #
            vec = [0 for _ in range(len(T))]
            vec[i] = 1
            e_ci.append(vec)
        #
        return C, p_c, e_ci

    def startCG(self):
        startCpuTimeCG, startWallTimeCG = time.clock(), time.time()
        logContents = '\n\n'
        logContents += '======================================================================================\n'
        logContents += '%s\n' % str(datetime.datetime.now())
        logContents += 'Start column generation of bnbNode(%s)\n' % self.nid
        logContents += '======================================================================================\n'
        log2file(self.etcSetting['LogFile'], logContents)
        #
        C = self.probSetting['C']
        T = self.probSetting['inputs']['T']
        RMP, q_c, taskAC, numBC = generate_RMP(self.probSetting)
        #
        counter, is_terminated = 0, False
        while True:
            if len(C) == len(T) ** 2 - 1:
                break
            LRMP = RMP.relax()
            set_grbSettings(LRMP, self.grbSetting)
            LRMP.optimize()
            if LRMP.status == GRB.Status.INFEASIBLE:
                logContents = '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                logContents += 'Relaxed model is infeasible!!\n'
                logContents += 'No solution!\n'
                logContents += '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                log2file(self.etcSetting['LogFile'], logContents)
                LRMP.computeIIS()
                import os.path as opath
                LRMP.write('%s.ilp' % opath.basename(self.etcSetting['LogFile']).split('.')[0])
                LRMP.write('%s.lp' % opath.basename(self.etcSetting['LogFile']).split('.')[0])
                assert False
            #
            counter += 1
            pi_i = [LRMP.getConstrByName("taskAC[%d]" % i).Pi for i in T]
            mu = LRMP.getConstrByName("numBC").Pi
            self.probSetting['pi_i'] = pi_i
            self.probSetting['mu'] = mu
            #
            startCpuTimeP, startWallTimeP = time.clock(), time.time()
            logContents = '\n\n'
            logContents += '======================================================================================\n'
            logContents += '%s\n' % str(datetime.datetime.now())
            logContents += 'Start %dth iteration of bnbNode(%s)\n' % (counter, self.nid)
            logContents += '\t Columns\n'
            logContents += '\t\t # of columns %d\n' % len(self.probSetting['C'])
            logContents += '\t\t %s\n' % str(self.probSetting['C'])
            logContents += '\t\t %s\n' % str(self.probSetting['p_c'])
            logContents += '\t Relaxed objVal\n'
            logContents += '\t\t z: %.3f\n' % LRMP.objVal
            logContents += '\t Dual V\n'
            logContents += '\t\t Pi: %s\n' % str([round(v, 3) for v in pi_i])
            logContents += '\t\t mu: %.3f\n' % mu
            logContents += '======================================================================================\n'
            log2file(self.etcSetting['LogFile'], logContents)
            objV_bc = SP_run(self.probSetting, self.etcSetting, self.grbSetting)
            if objV_bc == 'terminated':
                logContents = '\n\n'
                logContents += '======================================================================================\n'
                logContents += '%s\n' % str(datetime.datetime.now())
                logContents += '%dth iteration of bnbNode(%s)\n' % (counter, self.nid)
                logContents += 'Terminated because of the time limit!\n'
                logContents += '======================================================================================\n'
                log2file(self.etcSetting['LogFile'], logContents)
                is_terminated = True
                break
            elif objV_bc is None:
                logContents = '\n\n'
                logContents += '======================================================================================\n'
                logContents += '%s\n' % str(datetime.datetime.now())
                logContents += '%dth iteration of bnbNode(%s)\n' % (counter, self.nid)
                logContents += 'No solution!\n'
                logContents += '======================================================================================\n'
                log2file(self.etcSetting['LogFile'], logContents)
                break
            endCpuTimeP, endWallTimeP = time.clock(), time.time()
            eliCpuTimeP, eliWallTimeP = endCpuTimeP - startCpuTimeP, endWallTimeP - startWallTimeP
            #
            logContents = '\n\n'
            logContents += '======================================================================================\n'
            logContents += '%dth iteration (%s)\n' % (counter, str(datetime.datetime.now()))
            logContents += '\t Cpu Time\n'
            logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeP)
            logContents += '\t\t End.Time: %s\n' % str(endCpuTimeP)
            logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeP
            logContents += '\t Wall Time\n'
            logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeP)
            logContents += '\t\t End.Time: %s\n' % str(endWallTimeP)
            logContents += '\t\t Eli.Time: %f\n' % eliWallTimeP
            #
            objV, bc = objV_bc
            if objV < 0:
                logContents += '\n'
                logContents += 'The reduced cost of the generated column is a negative number\n'
                logContents += '======================================================================================\n'
                log2file(self.etcSetting['LogFile'], logContents)
                break
            else:
                logContents += '\n'
                logContents += '\t New column\n'
                logContents += '\t\t Tasks %s\n' % str(bc)
                logContents += '\t\t red. C. %.3f\n' % objV
                logContents += '======================================================================================\n'
                log2file(self.etcSetting['LogFile'], logContents)
                vec = [0 for _ in range(len(T))]
                for i in bc:
                    vec[i] = 1
                p = objV + (np.array(vec) * np.array(pi_i)).sum() + mu
                C, p_c, e_ci = list(map(self.probSetting.get, ['C', 'p_c', 'e_ci']))
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
                C.append(bc)
                RMP.update()
                bpt2file(self.etcSetting['bptFile'], [self.nid,
                                             datetime.datetime.fromtimestamp(startWallTimeP),
                                             eliWallTimeP, eliCpuTimeP,
                                             'S',
                                             {'numIter': counter, 'bc': bc}])
        if not is_terminated:
            LRMP = RMP.relax()
            set_grbSettings(LRMP, self.grbSetting)
            LRMP.optimize()
            C, p_c, e_ci = list(map(self.probSetting.get, ['C', 'p_c', 'e_ci']))
            q_c = [LRMP.getVarByName("q[%d]" % c).x for c in range(len(C))]
            #
            endCpuTimeCG, endWallTimeCG = time.clock(), time.time()
            eliCpuTimeCG, eliWallTimeCG = endCpuTimeCG - startCpuTimeCG, endWallTimeCG - startWallTimeCG
            chosenC = [(C[c], '%.2f' % q_c[c]) for c in range(len(C)) if q_c[c] > 0]
            #
            logContents = '\n\n'
            logContents += '======================================================================================\n'
            logContents += 'Column generation summary\n'
            logContents += '\t Cpu Time\n'
            logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeCG)
            logContents += '\t\t End.Time: %s\n' % str(endCpuTimeCG)
            logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeCG
            logContents += '\t Wall Time\n'
            logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeCG)
            logContents += '\t\t End.Time: %s\n' % str(endWallTimeCG)
            logContents += '\t\t Eli.Time: %f\n' % eliWallTimeCG
            logContents += '\t ObjV: %.3f\n' % LRMP.objVal
            logContents += '\t chosen B.: %s\n' % str(chosenC)
            logContents += '======================================================================================\n'
            log2file(self.etcSetting['LogFile'], logContents)
            #
            self.probSetting['C'] = C
            self.probSetting['p_c'] = p_c
            self.probSetting['e_ci'] = e_ci
            #
            self.res['objVal'] = LRMP.objVal
            self.res['q_c'] = q_c
            #
            bpt2file(self.etcSetting['bptFile'], [self.nid,
                                                    datetime.datetime.fromtimestamp(startWallTimeCG),
                                                    eliWallTimeCG, eliCpuTimeCG,
                                                    'M',
                                                    {'objVal': LRMP.objVal,
                                                     'inclusiveC': str(self.probSetting['inclusiveC']),
                                                     'exclusiveC': str(self.probSetting['exclusiveC'])}])
        #
        return is_terminated


if __name__ == '__main__':
    import os.path as opath
    from problems import paperExample, ex2
    #
    problem = paperExample()
    probSetting = {'problem': problem}
    bnpLogF = opath.join('_temp', 'paperExample_BNP.log')
    bnpResF = opath.join('_temp', 'paperExample_BNP.csv')
    bptFile = opath.join('_temp', 'paperExample_bnpTree.csv')

    # problem = ex2()
    # probSetting = {'problem': problem}
    # bnpLogF = opath.join('_temp', 'ex2_BNP.log')
    # bnpResF = opath.join('_temp', 'ex2_BNP.csv')
    # bptFile = opath.join('_temp', 'ex2_bnpTree.csv')



    etcSetting = {'LogFile': bnpLogF,
                  'ResFile': bnpResF,
                  'bptFile': bptFile,
                  # 'TimeLimit': 1

                  }
    grbSetting = {'LogFile': bnpLogF}
    #
    run(probSetting, etcSetting, grbSetting)
