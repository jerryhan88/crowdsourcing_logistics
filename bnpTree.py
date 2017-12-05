import datetime
import time
from heapq import heappush, heappop
from itertools import combinations

import treelib

from _utils.mm_utils import *
from _utils.recording import *
#
from bnpNode import BnPNode


class BnPTree(treelib.Tree):
    def __init__(self, probSetting, grbSetting, etcSetting):
        treelib.Tree.__init__(self)
        self.probSetting, self.grbSetting, self.etcSetting = probSetting, grbSetting, etcSetting
        #
        tag, indentifier = '-', '*'
        self.create_node(tag, indentifier,
                         data=BnPNode(indentifier, self.probSetting, self.grbSetting, self.etcSetting))
        self.bestBound, self.incumbent = None, None
        self.leafNodes = []

    def startBnP(self):
        rootNode = self.get_node('*')
        startCpuTimeBnP, startWallTimeBnP = time.clock(), time.time()
        logContents = '\n===========================================================\n'
        logContents += '%s\n' % str(datetime.datetime.now())
        logContents += 'Start Branch and Pricing from the root\n'
        logContents += '===========================================================\n'
        record_log(self.etcSetting['bnpLogF'], logContents)
        #
        is_feasible = rootNode.data.startCG()
        if not is_feasible:
            logContents = '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
            logContents += '%s\n' % str(datetime.datetime.now())
            logContents += 'The original (root) problem is infeasible\n'
            logContents += '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
            record_log(self.etcSetting['bnpLogF'], logContents)
            assert False
        self.branching(rootNode)
        incumProb = self.incumbent.data.probSetting
        incumRes = self.incumbent.data.res
        bbRes = self.bestBound.data.res
        BnPgap = abs(incumRes['objVal'] - bbRes['objVal']) / incumRes['objVal']
        chosenB = [incumProb['B'][b] for b in range(len(incumProb['B'])) if incumRes['q_b'][b] > 0.5]
        endCpuTimeBnP, endWallTimeBnP = time.clock(), time.time()
        eliCpuTimeBnP, eliWallTimeBnP = endCpuTimeBnP - startCpuTimeBnP, endWallTimeBnP - startWallTimeBnP
        #
        logContents = '\n\n===========================================================\n'
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
        logContents += '\t Gap: %.3f\n' % BnPgap
        logContents += '\t chosen B.: %s\n' % str(chosenB)
        logContents += '===========================================================\n'
        record_log(self.etcSetting['bnpLogF'], logContents)
        record_res(self.etcSetting['bnpResF'], incumRes['objVal'], BnPgap, eliCpuTimeBnP, eliWallTimeBnP)
        #
        rootNode.data.solveRootMIP()

    def branching_dfs_lcp(self):
        # Depth First Search and left child priority
        if self.leafNodes:
            _, nextNode = heappop(self.leafNodes)
            self.branching(nextNode)
        else:
            self.bestBound = self.incumbent

    def branching(self, curNode):
        now = datetime.datetime.now()
        logContents = '\n===========================================================\n'
        logContents += '%s\n' % str(now)
        logContents += 'Try Branching; bnbNode(%s)\n' % curNode.identifier
        logContents += '===========================================================\n'
        record_log(self.etcSetting['bnpLogF'], logContents)
        record_bpt(self.etcSetting['bptFile'], [curNode.identifier,
                                                now,
                                                None, None,
                                                'TB',
                                                None])
        #
        curProb = curNode.data.probSetting
        curRes = curNode.data.res
        if self.bestBound is None:
            self.bestBound = curNode
        # Pruning
        if self.incumbent:
            incumRes = self.incumbent.data.res
            if curRes['objVal'] < incumRes['objVal']:
                now = datetime.datetime.now()
                logContents = '\n===========================================================\n'
                logContents += '%s\n' % str(now)
                logContents += 'bnbNode(%s) was pruned\n' % curNode.identifier
                logContents += '===========================================================\n'
                record_log(self.etcSetting['bnpLogF'], logContents)
                record_bpt(self.etcSetting['bptFile'], [curNode.identifier,
                                                        now,
                                                        None, None,
                                                        'PR',
                                                        None])
                self.branching_dfs_lcp()
                return None
        #
        fracOrdered = []
        for b, q_b_var in enumerate(curRes['q_b']):
            if len(curProb['B'][b]) == 1:
                continue
            deviationFH = abs(q_b_var - 0.5)
            fracOrdered.append((deviationFH, curProb['B'][b]))
        fracOrdered.sort()
        mostFracVal, _ = fracOrdered[0]
        if abs(mostFracVal - 0.5) < EPSILON:
            #
            # Integral solution
            #
            now = datetime.datetime.now()
            logContents = '\n===========================================================\n'
            logContents += '%s\n' % str(now)
            logContents += 'Found a integral solution\n'
            record_bpt(self.etcSetting['bptFile'], [curNode.identifier,
                                                    now,
                                                    None, None,
                                                    'INT',
                                                    None])
            if self.incumbent is None:
                self.incumbent = curNode
                logContents += 'The first incumbent, bnbNode(%s)\n' % self.incumbent.identifier
                record_bpt(self.etcSetting['bptFile'], [curNode.identifier,
                                                        now,
                                                        None, None,
                                                        'IC',
                                                        'First'])
            else:
                incumRes = self.incumbent.data.res
                if incumRes['objVal'] < curRes['objVal']:
                    logContents += 'The incumbent was changed\n'
                    logContents += 'bnbNode(%s) -> bnbNode(%s)\n' % (self.incumbent.identifier, curNode.identifier)
                    record_bpt(self.etcSetting['bptFile'], [curNode.identifier,
                                                            now,
                                                            None, None,
                                                            'IC',
                                                            '%s -> %s' % (self.incumbent.identifier, curNode.identifier)])
                    self.incumbent = curNode
                else:
                    logContents += 'No change about the incumbent\n'
                    record_bpt(self.etcSetting['bptFile'], [curNode.identifier,
                                                            now,
                                                            None, None,
                                                            'NI',
                                                            None])
            logContents += '===========================================================\n'
            record_log(self.etcSetting['bnpLogF'], logContents)
            self.branching_dfs_lcp()
        else:
            #
            # Choose branching criteria
            #
            pTag, pIdentifier = curNode.tag, curNode.identifier
            pProbSetting = curNode.data.probSetting
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
                logContents = '\n===========================================================\n'
                logContents += '%s\n' % str(datetime.datetime.now())
                logContents += 'No suitable pairs\n'
                logContents += '===========================================================\n'
                record_log(self.etcSetting['bnpLogF'], logContents)
                assert False
            logContents = '\n===========================================================\n'
            logContents += '%s\n' % str(datetime.datetime.now())
            logContents += 'Start Branching; bnbNode(%s)\n' % curNode.identifier
            logContents += '\t All bundles %s\n' % str(curProb['B'])
            logContents += '\t Chosen bundle %s\n' % str(candiBundle)
            logContents += '\t Chosen tasks %s\n' % str((i0, i1))
            logContents += '===========================================================\n'
            record_log(self.etcSetting['bnpLogF'], logContents)
            #
            # Left child
            #
            lTag, lIndentifier = pTag + 'L', pIdentifier + '0'
            lProbSetting = self.duplicate_probSetting(pProbSetting)
            lProbSetting['inclusiveC'] += [(i0, i1)]
            self.create_node(lTag, lIndentifier, parent=pIdentifier,
                             data=BnPNode(lIndentifier, lProbSetting, self.grbSetting, self.etcSetting))
            lcNode = self.get_node(lIndentifier)
            lcNode_feasibility = lcNode.data.startCG()
            if lcNode_feasibility:
                heappush(self.leafNodes, (-(self.depth(lcNode) + 0.1), lcNode))
            #
            # Right child
            #
            rTag, rIndentifier = pTag + 'R', pIdentifier + '1'
            rProbSetting = self.duplicate_probSetting(pProbSetting)
            rProbSetting['exclusiveC'] += [(i0, i1)]
            self.create_node(rTag, rIndentifier, parent=pIdentifier,
                             data=BnPNode(rIndentifier, rProbSetting, self.grbSetting, self.etcSetting))
            rcNode = self.get_node(rIndentifier)
            rcNode_feasibility = rcNode.data.startCG()
            if rcNode_feasibility:
                heappush(self.leafNodes, (-self.depth(rcNode), rcNode))
            #
            if self.bestBound == curNode:
                #
                # Update the best bound
                #
                bestBoundNode, bestBoundVal = None, -1e400
                for nid, candiNode in self.nodes.items():
                    if candiNode.is_leaf() and candiNode.data.res and bestBoundVal < candiNode.data.res['objVal']:
                        bestBoundNode, bestBoundVal = candiNode, candiNode.data.res['objVal']
            #
            self.branching_dfs_lcp()

    def duplicate_probSetting(self, pProbSetting):
        nProbSetting = {'problem': pProbSetting['problem']}
        nProbSetting['B'] = [bundle[:] for bundle in pProbSetting['B']]
        nProbSetting['p_b'] = pProbSetting['p_b'][:]
        nProbSetting['e_bi'] = [vec[:] for vec in pProbSetting['e_bi']]
        nProbSetting['inclusiveC'] = pProbSetting['inclusiveC'][:]
        nProbSetting['exclusiveC'] = pProbSetting['exclusiveC'][:]
        return nProbSetting
