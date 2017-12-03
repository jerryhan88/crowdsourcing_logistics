from gurobipy import *
from datetime import datetime
from itertools import combinations
import numpy as np
import time
import treelib
from heapq import heappush, heappop
#
from problems import *
#
prefix = 'greedyHeuristic'
pyx_fn, c_fn = '%s.pyx' % prefix, '%s.c' % prefix
if opath.exists(c_fn):
    if opath.getctime(c_fn) < opath.getmtime(pyx_fn):
        from setup import cythonize; cythonize(prefix)
else:
    from setup import cythonize; cythonize(prefix)
from greedyHeuristic import run as gHeuristic_run
from optRouting import run as minTimePD_run
from pricing import run as pricingWIE_run
#
from _utils.recording import *
from _utils.mm_utils import *


class BnBNode(object):
    def __init__(self, nid, probSetting, paraSetting, grbSetting):
        self.nid = nid
        self.probSetting, self.paraSetting, self.grbSetting = probSetting, paraSetting, grbSetting
        assert 'problem' in self.probSetting
        assert 'pfCst' in self.paraSetting
        self.problem, self.inclusiveC, self.exclusiveC = list(map(self.probSetting.get,
                                                                  ['problem', 'inclusiveC', 'exclusiveC']))
        self.pfCst = self.paraSetting['pfCst']
        if not self.inclusiveC and not self.exclusiveC:
            assert 'B' not in self.probSetting
            assert 'p_b' not in self.probSetting
            assert 'e_bi' not in self.probSetting
            #
            # Generate initial bundles
            #
            self.B, self.p_b, self.e_bi = self.gen_initBundles()
            self.probSetting['B'] = self.B
            self.probSetting['p_b'] = self.p_b
            self.probSetting['e_bi'] = self.e_bi
        else:
            self.B, self.p_b, self.e_bi = list(map(self.probSetting.get,
                                                   ['B', 'p_b', 'e_bi']))
        self.res = {}

    def __repr__(self):
        return 'bnbNode(%s)' % self.nid

    def solve_cgM(self):
        startCpuTimeM, startWallTimeM = time.clock(), time.time()
        logContents = '\n\n'
        logContents += '===========================================================\n'
        logContents += '%s\n' % str(datetime.now())
        logContents += 'Start column generation of bnbNode(%s)\n' % self.nid
        record_log(self.grbSetting['LogFile'], logContents)
        #
        problem, B, p_b, e_bi = self.problem, self.B, self.p_b, self.e_bi
        bB, \
        T, r_i, v_i, _lambda, P, D, N, \
        K, w_k, t_ij, _delta = convert_input4MathematicalModel(*problem)
        input4subProblem = [T, r_i, v_i, _lambda, P, D, N, K, w_k, t_ij, _delta]
        B_i0i1 = {}
        for i0, i1 in set(self.inclusiveC).union(set(self.exclusiveC)):
            for b in range(len(B)):
                if i0 in B[b] and i1 in B[b]:
                    if (i0, i1) not in B_i0i1:
                        B_i0i1[i0, i1] = []
                    B_i0i1[i0, i1].append(b)
        #
        # Define decision variables
        #
        masterM = Model('materProblem')
        q_b = {}
        for b in range(len(B)):
            q_b[b] = masterM.addVar(vtype=GRB.BINARY, name="q[%d]" % b)
        masterM.update()
        #
        # Define objective
        #
        obj = LinExpr()
        for b in range(len(B)):
            obj += p_b[b] * q_b[b]
        masterM.setObjective(obj, GRB.MAXIMIZE)
        #
        # Define constrains
        #
        taskAC = {}
        for i in T:  # eq:taskA
            taskAC[i] = masterM.addConstr(quicksum(e_bi[b][i] * q_b[b] for b in range(len(B))) == 1,
                                          name="taskAC[%d]" % i)
        numBC = masterM.addConstr(quicksum(q_b[b] for b in range(len(B))) == bB,
                                  name="numBC")

        try:
            for i, (i0, i1) in enumerate(self.inclusiveC):
                masterM.addConstr(quicksum(q_b[b] for b in B_i0i1[i0, i1]) >= 1,
                                  name="mIC[%d]" % i)
        except KeyError:
            import pickle
            with open(path, 'wb') as fp:
                pickle.dump(self.probSetting, self.grbSetting['LogFile'][:-len('.log')] + '.pkl')
            assert False


        for i, (i0, i1) in enumerate(self.exclusiveC):
            masterM.addConstr(quicksum(q_b[b] for b in B_i0i1[i0, i1]) <= 0,
                              name="mEC[%d]" % i)
        masterM.update()
        #
        counter, is_feasible = 0, True
        while True:
            if len(B) == len(T) ** 2 - 1:
                break
            relaxM = masterM.relax()
            set_grbSettings(relaxM, self.grbSetting)
            relaxM.optimize()
            if relaxM.status == GRB.Status.INFEASIBLE:
                # relaxM.computeIIS()
                # relaxM.write('relexM.ilp')
                # relaxM.write('relexM.lp')
                logContents = '\n\n'
                logContents += 'Relaxed model is infeasible!!\n'
                logContents += 'No solution!\n'
                is_feasible = False
                break
            #
            counter += 1
            pi_i = [relaxM.getConstrByName("taskAC[%d]" % i).Pi for i in T]
            mu = relaxM.getConstrByName("numBC").Pi
            startCpuTimeP, startWallTimeP = time.clock(), time.time()
            bestSols = pricingWIE_run(counter,
                                      pi_i, mu, B, input4subProblem,
                                      self.inclusiveC, self.exclusiveC,
                                      self.pfCst, self.grbSetting)
            if bestSols is None:
                logContents = '\n\n'
                logContents += '%dth iteration (%s)\n' % (counter, str(datetime.now()))
                logContents += 'No solution!\n'
                break
            endCpuTimeP, endWallTimeP = time.clock(), time.time()
            eliCpuTimeP, eliWallTimeP = endCpuTimeP - startCpuTimeP, endWallTimeP - startWallTimeP
            logContents = '\n\n'
            logContents += '%dth iteration (%s)\n' % (counter, str(datetime.now()))
            logContents += '\t Cpu Time\n'
            logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeP)
            logContents += '\t\t End.Time: %s\n' % str(endCpuTimeP)
            logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeP
            logContents += '\t Wall Time\n'
            logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeP)
            logContents += '\t\t End.Time: %s\n' % str(endWallTimeP)
            logContents += '\t\t Eli.Time: %f\n' % eliWallTimeP
            logContents += '\t Dual V\n'
            logContents += '\t\t Pi: %s\n' % str([round(v, 3) for v in pi_i])
            logContents += '\t\t mu: %.3f\n' % mu
            numPRC = 0
            for c_b, bundle in bestSols:
                if c_b < EPSILON:
                    continue
                numPRC += 1
                logContents += '\t New B. %d\n' % numPRC
                logContents += '\t\t Tasks %s\n' % str(bundle)
                logContents += '\t\t red. C. %.3f\n' % c_b
                vec = [0 for _ in range(len(T))]
                for i in bundle:
                    vec[i] = 1
                p = c_b + (np.array(vec) * np.array(pi_i)).sum() + mu
                e_bi.append(vec)
                p_b.append(p)
                #
                col = Column()
                for i in range(len(T)):
                    if e_bi[len(B)][i] > 0:
                        col.addTerms(e_bi[len(B)][i], taskAC[i])
                col.addTerms(1, numBC)
                #
                q_b[len(B)] = masterM.addVar(obj=p_b[len(B)], vtype=GRB.BINARY, name="q[%d]" % len(B), column=col)
                B.append(bundle)
                masterM.update()
            if numPRC == 0:
                logContents += '\t No new bundles\n'
                record_log(self.grbSetting['LogFile'], logContents)
                break
            else:
                record_log(self.grbSetting['LogFile'], logContents)
        #
        if is_feasible:
            relaxM = masterM.relax()
            set_grbSettings(relaxM, self.grbSetting)
            relaxM.optimize()
            q_b = [relaxM.getVarByName("q[%d]" % b).x for b in range(len(B))]
            #
            endCpuTimeM, endWallTimeM = time.clock(), time.time()
            eliCpuTimeM, eliWallTimeM = endCpuTimeM - startCpuTimeM, endWallTimeM - startWallTimeM
            chosenB = [(B[b], '%.2f' % q_b[b]) for b in range(len(B)) if q_b[b] > 0]
            #
            logContents = '\n\n'
            logContents += 'Column generation summary\n'
            logContents += '\t Cpu Time\n'
            logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeM)
            logContents += '\t\t End.Time: %s\n' % str(endCpuTimeM)
            logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeM
            logContents += '\t Wall Time\n'
            logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeM)
            logContents += '\t\t End.Time: %s\n' % str(endWallTimeM)
            logContents += '\t\t Eli.Time: %f\n' % eliWallTimeM
            logContents += '\t ObjV: %.3f\n' % relaxM.objVal
            logContents += '\t chosen B.: %s\n' % str(chosenB)
            logContents += '===========================================================\n'
            record_log(self.grbSetting['LogFile'], logContents)
            #
            self.res['objVal'] = relaxM.objVal
            self.res['B'] = B
            self.res['p_b'] = p_b
            self.res['e_bi'] = e_bi
            self.res['q_b'] = q_b
        #
        return is_feasible

    def gen_initBundles(self):
        bB, \
        T, r_i, v_i, _lambda, P, D, N, \
        K, w_k, t_ij, _delta = convert_input4MathematicalModel(*self.problem)
        #
        # generate initial bundles with the greedy heuristic
        #
        _, _, B = gHeuristic_run(self.problem)
        e_bi = []
        for b in B:
            vec = [0 for _ in range(len(T))]
            for i in b:
                vec[i] = 1
            e_bi.append(vec)
        #
        logContents = 'Initial bundles\n'
        for b in B:
            logContents += '\t %s\n' % str(b)
        if self.grbSetting['LogFile']:
            with open(self.grbSetting['LogFile'], 'wt') as f:
                f.write(logContents)
        else:
            print(logContents)
        #
        p_b = []
        logContents = 'Bundle-Path feasibility\n'
        for b in range(len(B)):
            logContents += '%s\n' % str(B[b])
            bundle = [i for i, v in enumerate(e_bi[b]) if v == 1]
            p = 0
            br = sum([r_i[i] for i in bundle])
            for k, w in enumerate(w_k):
                detourTime, route = minTimePD_run(bundle, k, t_ij)
                if detourTime <= _delta:
                    p += w * br
                    logContents += '\t k%d, dt %.2f; %d;\t %s\n' % (k, detourTime, 1, str(route))
                else:
                    logContents += '\t k%d, dt %.2f; %d;\t %s\n' % (k, detourTime, 0, str(route))
            p_b.append(p)
            logContents += '\t\t\t\t\t\t %.3f\n' % p
        if LOGGING_FEASIBILITY:
            record_log(self.grbSetting['LogFile'], logContents)
        return B, p_b, e_bi


class BnBTree(treelib.Tree):
    def __init__(self, probSetting, paraSetting, grbSetting):
        treelib.Tree.__init__(self)
        self.probSetting, self.paraSetting, self.grbSetting = probSetting, paraSetting, grbSetting
        #
        tag, indentifier = '-', '*'
        self.create_node(tag, indentifier,
                         data=BnBNode(indentifier, self.probSetting, self.paraSetting, self.grbSetting))
        self.bestBound, self.incumbent = None, None
        self.leafNodes = []

    def startBnP(self):
        rootNode = self.get_node('*')
        startCpuTimeBnP, startWallTimeBnP = time.clock(), time.time()
        logContents = '\n===========================================================\n'
        logContents += '%s\n' % str(datetime.now())
        logContents += 'Start Branch and Pricing from the root\n'
        logContents += '===========================================================\n'
        record_log(self.grbSetting['LogFile'], logContents)
        #
        is_feasible = rootNode.data.startCG()
        if not is_feasible:
            logContents = '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
            logContents += '%s\n' % str(datetime.now())
            logContents += 'The original (root) problem is infeasible\n'
            logContents += '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
            record_log(self.grbSetting['LogFile'], logContents)
            assert False
        self.branching(rootNode)
        incumRes = self.incumbent.data.res
        bbRes = self.bestBound.data.res
        BnPgap = abs(incumRes['objVal'] - bbRes['objVal']) / incumRes['objVal']
        chosenB = [incumRes['B'][b] for b in range(len(incumRes['B'])) if incumRes['q_b'][b] > 0.5]
        #
        endCpuTimeBnP, endWallTimeBnP = time.clock(), time.time()
        eliCpuTimeBnP, eliWallTimeBnP = endCpuTimeBnP - startCpuTimeBnP, endWallTimeBnP - startWallTimeBnP
        logContents = '\n\n===========================================================\n'
        logContents += '%s\n' % str(datetime.now())
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
        record_log(self.grbSetting['LogFile'], logContents)
        return incumRes['objVal'], BnPgap, eliCpuTimeBnP, eliWallTimeBnP

    def branching_dfs_lcp(self):
        # Depth First Search and left child priority
        if self.leafNodes:
            _, nextNode = heappop(self.leafNodes)
            self.branching(nextNode)
        else:
            self.bestBound = self.incumbent

    def branching(self, curNode):
        logContents = '\n===========================================================\n'
        logContents += '%s\n' % str(datetime.now())
        logContents += 'Try Branching; bnbNode(%s)\n' % curNode.identifier
        logContents += '===========================================================\n'
        record_log(self.grbSetting['LogFile'], logContents)
        curRes = curNode.data.res
        if self.bestBound is None:
            self.bestBound = curNode
        # Pruning
        if self.incumbent:
            incumRes = self.incumbent.data.res
            if curRes['objVal'] < incumRes['objVal']:
                logContents = '\n===========================================================\n'
                logContents += '%s\n' % str(datetime.now())
                logContents += 'bnbNode(%s) was pruned\n' % curNode.identifier
                logContents += '===========================================================\n'
                record_log(self.grbSetting['LogFile'], logContents)
                self.branching_dfs_lcp()
                return None
        #
        fracOrdered = []
        for b, q_b_var in enumerate(curRes['q_b']):
            if len(curRes['B'][b]) == 1:
                continue
            deviationFH = abs(q_b_var - 0.5)
            fracOrdered.append((deviationFH, curRes['B'][b]))
        fracOrdered.sort()
        mostFracVal, _ = fracOrdered[0]
        if abs(mostFracVal - 0.5) < EPSILON:
            #
            # Integral solution
            #
            logContents = '\n===========================================================\n'
            logContents += '%s\n' % str(datetime.now())
            logContents += 'Found a integral solution\n'
            if self.incumbent is None:
                self.incumbent = curNode
                logContents += 'The first incumbent, bnbNode(%s)\n' % self.incumbent.tag
            else:
                incumRes = self.incumbent.data.res
                if incumRes['objVal'] < curRes['objVal']:
                    logContents += 'The incumbent was changed\n'
                    logContents += 'bnbNode(%s) -> bnbNode(%s)\n' % (self.incumbent.tag,
                                                                     curNode.tag)
                    self.incumbent = curNode
                else:
                    logContents += 'No change about the incumbent\n'
            logContents += '===========================================================\n'
            record_log(self.grbSetting['LogFile'], logContents)
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
                logContents += '%s\n' % str(datetime.now())
                logContents += 'No suitable pairs\n'
                logContents += '===========================================================\n'
                record_log(self.grbSetting['LogFile'], logContents)
                assert False
            logContents = '\n===========================================================\n'
            logContents += '%s\n' % str(datetime.now())
            logContents += 'Start Branching; bnbNode(%s)\n' % curNode.identifier
            logContents += '\t Chosen bundle %s\n' % str(candiBundle)
            logContents += '\t Chosen tasks %s\n' % str((i0, i1))
            logContents += '===========================================================\n'
            record_log(self.grbSetting['LogFile'], logContents)
            #
            # Left child
            #
            lTag, lIndentifier = pTag + 'L', pIdentifier + '0'
            lProbSetting = self.duplicate_probSetting(pProbSetting)
            lProbSetting['inclusiveC'] += [(i0, i1)]
            self.create_node(lTag, lIndentifier, parent=pIdentifier,
                             data=BnBNode(lIndentifier, lProbSetting, self.paraSetting, self.grbSetting))
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
                             data=BnBNode(rIndentifier, rProbSetting, self.paraSetting, self.grbSetting))
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
        nProbSetting['B'] = [bundle[:] for bundle in self.probSetting['B']]
        nProbSetting['p_b'] = self.probSetting['p_b'][:]
        nProbSetting['e_bi'] = [vec[:] for vec in self.probSetting['e_bi']]
        nProbSetting['inclusiveC'] = pProbSetting['inclusiveC'][:]
        nProbSetting['exclusiveC'] = pProbSetting['exclusiveC'][:]
        return nProbSetting

#
ifpath = None
def test():
    import pickle

    with open(ifpath, 'rb') as fp:
        inputs = pickle.load(fp)
    _pfCst = 1.2

    probSetting = {'problem': inputs,
                   'inclusiveC': [], 'exclusiveC': []}


    paraSetting = {'pfCst': _pfCst}
    grbSetting = {'LogFile': 'test(%.2f).log' % _pfCst,
                  'Threads': 8}

    bnbTree = BnBTree(probSetting, paraSetting, grbSetting)
    bnbTree.startBnP()


if __name__ == '__main__':
    test()
    # calTime_test()
