from sys import exit
import datetime, time
import treelib
import numpy as np
from itertools import combinations
from heapq import heappush, heappop
from gurobipy import *
#
from _util import write_log, bpt2file, res2file
from _util import set_grbSettings
from RMP import generate_RMP
from SP import run as SP_run
from PD_MM import calc_expectedProfit
from problems import convert_prob2prmt


EPSILON = 0.000000001


def run(problem, etcSetting, grbSetting):
    startCpuTimeBnP, startWallTimeBnP = time.clock(), time.time()
    if 'TimeLimit' not in etcSetting:
        etcSetting['TimeLimit'] = 1e400
    etcSetting['startTS'] = startCpuTimeBnP
    etcSetting['startCpuTimeBnP'] = startCpuTimeBnP
    etcSetting['startWallTimeBnP'] = startWallTimeBnP
    bpt2file(etcSetting['bptFile'])
    #
    ori_inputs = convert_prob2prmt(*problem)
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
    rootNode.ni = bnpNode(indentifier, ori_inputs, bnp_inputs, etcSetting, grbSetting)  # ni: node instance
    #
    logContents = 'Start Branch and Pricing from the root\n'
    write_log(etcSetting['LogFile'], logContents)
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
    chosenB = [incumProb['C'][c] for c in range(len(incumProb['C'])) if incumRes['q_c'][c] > 0.5]
    endCpuTimeBnP, endWallTimeBnP = time.clock(), time.time()
    eliCpuTimeBnP, eliWallTimeBnP = endCpuTimeBnP - startCpuTimeBnP, endWallTimeBnP - startWallTimeBnP
    #
    logContents = 'End Branch and Pricing from the root\n'
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
    write_log(etcSetting['LogFile'], logContents)
    res2file(etcSetting['ResFile'], incumRes['objVal'], BnPgap, eliCpuTimeBnP, eliWallTimeBnP)


def branching(bnpTree, curNode):
    etcSetting, grbSetting = curNode.ni.etcSetting, curNode.ni.grbSetting
    bestBound, incumbent = bnpTree.bestBound, bnpTree.incumbent
    #
    logContents = 'Try Branching; bnbNode(%s)\n' % curNode.identifier
    write_log(etcSetting['LogFile'], logContents)
    bpt2file(etcSetting['bptFile'], [curNode.identifier, datetime.datetime.now(),
                                     None, None, 'TB', None])
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
            write_log(etcSetting['LogFile'], logContents)
            bpt2file(etcSetting['bptFile'], [curNode.identifier, datetime.datetime.now(),
                                             None, None, 'PR', None])
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
        write_log(etcSetting['LogFile'], logContents)
        bpt2file(etcSetting['bptFile'], [curNode.identifier, datetime.datetime.now(),
                                         None, None, 'INT', None])
        if incumbent is None:
            logContents = 'The first incumbent, bnbNode(%s)\n' % curNode.identifier
            write_log(etcSetting['LogFile'], logContents)
            bpt2file(etcSetting['bptFile'], [curNode.identifier, datetime.datetime.now(),
                                             None, None, 'IC', 'First'])
            bnpTree.incumbent = curNode
        else:
            incumRes = incumbent.ni.res
            if incumRes['objVal'] < curRes['objVal']:
                logContents = 'The incumbent was changed\n'
                logContents += 'bnbNode(%s) -> bnbNode(%s)\n' % (incumbent.identifier, curNode.identifier)
                write_log(etcSetting['LogFile'], logContents)
                bpt2file(etcSetting['bptFile'], [curNode.identifier, datetime.datetime.now(),
                                                        None, None,
                                                        'IC',
                                                        '%s -> %s' % (incumbent.identifier, curNode.identifier)])
                bnpTree.incumbent = curNode
            else:
                logContents = 'No change about the incumbent\n'
                write_log(etcSetting['LogFile'], logContents)
                bpt2file(etcSetting['bptFile'], [curNode.identifier, datetime.datetime.now(),
                                                        None, None,
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
            write_log(etcSetting['LogFile'], logContents)
            assert False
        logContents = 'Start Branching; bnbNode(%s)\n' % curNode.identifier
        logContents += '\t All bundles %s\n' % str(curProb['C'])
        logContents += '\t Chosen bundle %s\n' % str(candiBundle)
        logContents += '\t Chosen tasks %s\n' % str((i0, i1))
        write_log(etcSetting['LogFile'], logContents)
        #
        # Left child
        #
        lTag, lIndentifier = pTag + 'L', pIdentifier + '0'
        lBNP_inputs = duplicate_BNP_inputs(pBNP_inputs)
        lBNP_inputs['inclusiveC'] += [(i0, i1)]
        bnpTree.create_node(lTag, lIndentifier, parent=pIdentifier)
        lcNode = bnpTree.get_node(lIndentifier)
        lcNode.ni = bnpNode(lIndentifier, pOri_inputs, lBNP_inputs, etcSetting, grbSetting)
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
        rcNode.ni = bnpNode(rIndentifier, pOri_inputs, rBNP_inputs, etcSetting, grbSetting)
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
    write_log(etcSetting['LogFile'], logContents)
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
    def __init__(self, nid, ori_inputs, bnp_inputs, etcSetting, grbSetting):
        self.nid = nid
        self.ori_inputs, self.bnp_inputs = ori_inputs, bnp_inputs
        self.etcSetting, self.grbSetting = etcSetting, grbSetting
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
        T = self.ori_inputs['T']
        C, p_c, e_ci = [], [], []
        for i in T:
            Ts = [i]
            C.append(Ts)
            #
            ep = calc_expectedProfit(self.ori_inputs, grbSetting, Ts)
            p_c.append(ep)
            #
            vec = [0 for _ in range(len(T))]
            vec[i] = 1
            e_ci.append(vec)
        #
        return C, p_c, e_ci

    def startCG(self):
        startCpuTimeCG, startWallTimeCG = time.clock(), time.time()
        logContents = 'Start column generation of bnbNode(%s)\n' % self.nid
        write_log(self.etcSetting['LogFile'], logContents)
        #
        T = self.ori_inputs['T']
        C = self.bnp_inputs['C']
        RMP, q_c, taskAC, numBC = generate_RMP(self.ori_inputs, self.bnp_inputs)
        #
        counter, is_terminated = 0, False
        while True:
            if len(C) == len(T) ** 2 - 1:
                break
            LRMP = RMP.relax()
            set_grbSettings(LRMP, self.grbSetting)
            LRMP.optimize()
            if LRMP.status == GRB.Status.INFEASIBLE:
                logContents = 'Relaxed model is infeasible!!\n'
                logContents += 'No solution!\n'
                write_log(self.etcSetting['LogFile'], logContents)
                LRMP.computeIIS()
                import os.path as opath
                LRMP.write('%s.ilp' % opath.basename(self.etcSetting['LogFile']).split('.')[0])
                LRMP.write('%s.lp' % opath.basename(self.etcSetting['LogFile']).split('.')[0])
                assert False
            #
            counter += 1
            pi_i = [LRMP.getConstrByName("taskAC[%d]" % i).Pi for i in T]
            mu = LRMP.getConstrByName("numBC").Pi
            self.bnp_inputs['pi_i'] = pi_i
            self.bnp_inputs['mu'] = mu
            #
            startCpuTimeP, startWallTimeP = time.clock(), time.time()
            logContents = 'Start %dth iteration of bnbNode(%s)\n' % (counter, self.nid)
            logContents += '\t Columns\n'
            logContents += '\t\t # of columns %d\n' % len(self.bnp_inputs['C'])
            logContents += '\t\t %s\n' % str(self.bnp_inputs['C'])
            logContents += '\t\t %s\n' % str(self.bnp_inputs['p_c'])
            logContents += '\t Relaxed objVal\n'
            logContents += '\t\t z: %.2f\n' % LRMP.objVal
            logContents += '\t\t RC: %s\n' % str(['%.2f' % LRMP.getVarByName("q[%d]" % c).RC for c in range(len(C))])
            logContents += '\t Dual V\n'
            logContents += '\t\t Pi: %s\n' % str(['%.2f' % v for v in pi_i])
            logContents += '\t\t mu: %.2f\n' % mu
            write_log(self.etcSetting['LogFile'], logContents)
            #
            objV_newC = SP_run(self.ori_inputs, self.bnp_inputs, self.etcSetting, self.grbSetting)
            if objV_newC == 'terminated':
                logContents = '%dth iteration of bnbNode(%s)\n' % (counter, self.nid)
                logContents += 'Terminated because of the time limit!\n'
                write_log(self.etcSetting['LogFile'], logContents)
                is_terminated = True
                break
            elif objV_newC is None:
                logContents = '%dth iteration of bnbNode(%s)\n' % (counter, self.nid)
                logContents += 'No solution!\n'
                write_log(self.etcSetting['LogFile'], logContents)
                break
            endCpuTimeP, endWallTimeP = time.clock(), time.time()
            eliCpuTimeP, eliWallTimeP = endCpuTimeP - startCpuTimeP, endWallTimeP - startWallTimeP
            #
            logContents = '%dth iteration (%s)\n' % (counter, str(datetime.datetime.now()))
            logContents += '\t Cpu Time: %f\n' % eliCpuTimeP
            logContents += '\t Wall Time: %f\n' % eliWallTimeP
            #
            objV, newC = objV_newC
            if objV < 0:
                logContents += '\n'
                logContents += 'The reduced cost of the generated column is a negative number\n'
                write_log(self.etcSetting['LogFile'], logContents)
                break
            else:
                logContents += '\n'
                logContents += '\t New column\n'
                logContents += '\t\t Tasks %s\n' % str(newC)
                logContents += '\t\t red. C. %.3f\n' % objV
                write_log(self.etcSetting['LogFile'], logContents)
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
                bpt2file(self.etcSetting['bptFile'], [self.nid,
                                             datetime.datetime.fromtimestamp(startWallTimeP),
                                             eliWallTimeP, eliCpuTimeP,
                                             'S',
                                             {'numIter': counter, 'bc': newC}])
        if not is_terminated:
            LRMP = RMP.relax()
            set_grbSettings(LRMP, self.grbSetting)
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
            write_log(self.etcSetting['LogFile'], logContents)
            #
            self.bnp_inputs['C'] = C
            self.bnp_inputs['p_c'] = p_c
            self.bnp_inputs['e_ci'] = e_ci
            #
            self.res['objVal'] = LRMP.objVal
            self.res['q_c'] = q_c
            #
            bpt2file(self.etcSetting['bptFile'], [self.nid,
                                                    datetime.datetime.fromtimestamp(startWallTimeCG),
                                                    eliWallTimeCG, eliCpuTimeCG,
                                                    'M',
                                                    {'objVal': LRMP.objVal,
                                                     'inclusiveC': str(self.bnp_inputs['inclusiveC']),
                                                     'exclusiveC': str(self.bnp_inputs['exclusiveC'])}])
        #
        return is_terminated


if __name__ == '__main__':
    import os.path as opath
    from problems import paperExample, ex1
    #
    problem = paperExample()
    # problem = ex1()
    problemName = problem[0]
    log_fpath = opath.join('_temp', '%s_BNP.log' % problemName)
    res_fpath = opath.join('_temp', '%s_BNP.csv' % problemName)
    bpt_fpath = opath.join('_temp', '%s_bnpTree.csv' % problemName)
    etcSetting = {'LogFile': log_fpath,
                  'ResFile': res_fpath,
                  'bptFile': bpt_fpath,
                  # 'TimeLimit': 1
                  }
    grbSetting = {'LogFile': log_fpath}
    #
    run(problem, etcSetting, grbSetting)
