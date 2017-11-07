from gurobipy import *
from datetime import datetime
from random import sample
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
from minTimePD import run as minTimePD_run
from pricingWIE import run as pricingWIE_run
#
from _utils.logging import *
from _utils.mm_utils import *


import pickle
ifpath = 'nt10-np20-nb4-tv3-td7.pkl'
prefix = ifpath[:-len('.csv')]
tsl_fpath = '%s.tsl' % prefix


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
        logContents += 'Start column generation of %s\n' % self.nid
        record_logs(self.grbSetting['LogFile'], logContents)
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
        for i, (i0, i1) in enumerate(self.inclusiveC):
            masterM.addConstr(quicksum(q_b[b] for b in B_i0i1[i0, i1]) >= 1,
                              name="mIC[%d]" % i)
        for i, (i0, i1) in enumerate(self.exclusiveC):
            masterM.addConstr(quicksum(q_b[b] for b in B_i0i1[i0, i1]) <= 0,
                              name="mEC[%d]" % i)
        masterM.update()
        #
        counter = 0
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
                logContents += 'Relaxed model is infeasible (%s)\n' % (counter, str(datetime.now()))
                logContents += 'No solution!\n'
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
                record_logs(self.grbSetting['LogFile'], logContents)
                break
            else:
                record_logs(self.grbSetting['LogFile'], logContents)
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
        record_logs(self.grbSetting['LogFile'], logContents)
        #
        self.res['objVal'] = relaxM.objVal
        self.res['B'] = B
        self.res['p_b'] = p_b
        self.res['e_bi'] = e_bi
        self.res['q_b'] = q_b

    def gen_initBundles(self):
        if not opath.exists(tsl_fpath):
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
                record_logs(self.grbSetting['LogFile'], logContents)
        else:
            with open(tsl_fpath, 'rb') as fp:
                B, p_b, e_bi = pickle.load(fp)
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
        logContents = '===========================================================\n'
        record_logs(self.grbSetting['LogFile'], logContents)
        #
        rootNode.data.solve_cgM()
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
        logContents = '===========================================================\n'
        record_logs(self.grbSetting['LogFile'], logContents)
        return incumRes['objVal'], BnPgap, eliCpuTimeBnP, eliWallTimeBnP

    def branchBestBound(self):
        # Branch the next node whose bound is the highest value
        if self.leafNodes:
            _, nextNode = heappop(self.leafNodes)
            self.branching(nextNode)

    def branching(self, curNode):
        curRes = curNode.data.res
        if self.bestBound is None:
            self.bestBound = curNode
        # Pruning
        if self.incumbent:
            incumRes = self.incumbent.data.res
            if curRes['objVal'] < incumRes['objVal']:
                self.branchBestBound()
                return None
        #
        mostFracArg, mostFracVal = None, 0.5
        for b, q_b_var in enumerate(curRes['q_b']):
            if len(curRes['B'][b]) == 1:
                continue
            deviationFH = abs(q_b_var - 0.5)
            if deviationFH < mostFracVal:
                mostFracArg, mostFracVal = b, deviationFH
        if abs(mostFracVal - 0.5) < EPSILON:
            # Integral a solution
            if self.incumbent is None:
                self.incumbent = curNode
            else:
                incumRes = self.incumbent.data.res
                if incumRes['objVal'] < curRes['objVal']:
                    self.incumbent = curNode
            self.branchBestBound()
        else:
            #
            # Choose branching criteria
            #
            i0, i1 = sample(curRes['B'][mostFracArg], 2)
            #
            # Start branching
            #
            pTag, pIdentifier = curNode.tag, curNode.identifier
            pProbSetting = curNode.data.probSetting
            #
            # Left child
            #
            lTag, lIndentifier = pTag + 'L', pIdentifier + '0'
            lProbSetting = self.duplicate_probSetting(pProbSetting)
            lProbSetting['inclusiveC'] += [(i0, i1)]
            self.create_node(lTag, lIndentifier, parent=pIdentifier,
                             data=BnBNode(lIndentifier, lProbSetting, self.paraSetting, self.grbSetting))
            lcNode = self.get_node(lIndentifier)
            lcNode.data.solve_cgM()
            lcRes = lcNode.data.res
            heappush(self.leafNodes, (-lcRes['objVal'], lcNode))
            #
            # Right child
            #
            rTag, rIndentifier = pTag + 'R', pIdentifier + '1'
            rProbSetting = self.duplicate_probSetting(pProbSetting)
            rProbSetting['exclusiveC'] += [(i0, i1)]
            self.create_node(rTag, rIndentifier, parent=pIdentifier,
                             data=BnBNode(rIndentifier, rProbSetting, self.paraSetting, self.grbSetting))
            rcNode = self.get_node(rIndentifier)
            rcNode.data.solve_cgM()
            rcRes = rcNode.data.res
            heappush(self.leafNodes, (-rcRes['objVal'], lcNode))
            #
            self.branchBestBound()

    def duplicate_probSetting(self, pProbSetting):
        nProbSetting = {'problem': pProbSetting['problem']}
        nProbSetting['B'] = [bundle[:] for bundle in self.probSetting['B']]
        nProbSetting['p_b'] = self.probSetting['p_b'][:]
        nProbSetting['e_bi'] = [vec[:] for vec in self.probSetting['e_bi']]
        nProbSetting['inclusiveC'] = pProbSetting['inclusiveC'][:]
        nProbSetting['exclusiveC'] = pProbSetting['exclusiveC'][:]
        return nProbSetting






#     leftChdCond, rightChdCond = [], []
#
#
#     # branchingTarget = sample(B[mostDeviArg], 2)
#     # branchingTarget = sample(B[22], 2)
#     branchingTarget = [0, 5]
#     for branchingTarget in [[0, 5],
#                             [7, 4],
#                             [2, 8], [2, 9], [8, 9],
#                             [0, 6], [0, 8], [6, 8],
#                             [0, 2], [0, 3], [2, 3],
#                             [4, 6], [4, 9], [6, 9], ]:
#         leftChdCond.append(branchingTarget)
#         rightChdCond.append(branchingTarget)
#         #
#         # Left Child
#         #
#
#         bestSolsL = pricingWIE_run(pi_i, mu, B, input4subProblem,
#                                          counter, log_fpath, numThreads, TimeLimit, pfCst,
#                                          leftChdCond, [])
#         bestSolsR = pricingWIE_run(pi_i, mu, B, input4subProblem,
#                                          counter, log_fpath, numThreads, TimeLimit, pfCst,
#                                          [], rightChdCond)
#         print('\n')
#         print(branchingTarget)
#         print('Left Child')
#         print(bestSolsL)
#         print('Right Child')
#         print(bestSolsR)




def test():
    with open(ifpath, 'rb') as fp:
        inputs = pickle.load(fp)
    _pfCst = 1.5

    probSetting = {'problem': inputs,
                   'inclusiveC': [], 'exclusiveC': []}
    paraSetting = {'pfCst': _pfCst}
    grbSetting = {'LogFile': 'temp(%.2f).log' % _pfCst}

    bnbTree = BnBTree(probSetting, paraSetting, grbSetting)
    bnbTree.startBnP()
    # objV, gap, eliCpuTime, eliWallTime = run(inputs, log_fpath=, pfCst=)


if __name__ == '__main__':
    test()
    # calTime_test()






# def run(problem, log_fpath=None, numThreads=None, TimeLimit=None, pfCst=None):
#     def gurobi_settings(mm):
#         if TimeLimit is not None:
#             mm.setParam('TimeLimit', TimeLimit)
#         if numThreads is not None:
#             mm.setParam('Threads', numThreads)
#         if log_fpath is not None:
#             mm.setParam('LogFile', log_fpath)
#             mm.setParam('OutputFlag', X_GL)
#     #
#     startCpuTimeM, startWallTimeM = time.clock(), time.time()
#     #
#     # Solve a master problem
#     #
#     bB, \
#     T, r_i, v_i, _lambda, P, D, N, \
#     K, w_k, t_ij, _delta = convert_input4MathematicalModel(*problem)
#     input4subProblem = [T, r_i, v_i, _lambda, P, D, N, K, w_k, t_ij, _delta]
#
#
#     if not opath.exists(tsl_fpath):
#         #
#         # generate initial bundles with the greedy heuristic
#         #
#         _, _, B = gHeuristic_run(problem)
#         e_bi = []
#         for b in B:
#             vec = [0 for _ in range(len(T))]
#             for i in b:
#                 vec[i] = 1
#             e_bi.append(vec)
#         #
#         logContents = 'Initial bundles\n'
#         for b in B:
#             logContents += '\t %s\n' % str(b)
#         if log_fpath:
#             with open(log_fpath, 'wt') as f:
#                 f.write(logContents)
#         else:
#             print(logContents)
#         #
#         p_b = []
#         logContents = 'Bundle-Path feasibility\n'
#         for b in range(len(B)):
#             logContents += '%s\n' % str(B[b])
#             bundle = [i for i, v in enumerate(e_bi[b]) if v == 1]
#             p = 0
#             br = sum([r_i[i] for i in bundle])
#             for k, w in enumerate(w_k):
#                 detourTime, route = minTimePD_run(bundle, k, t_ij)
#                 if detourTime <= _delta:
#                     p += w * br
#                     logContents += '\t k%d, dt %.2f; %d;\t %s\n' % (k, detourTime, 1, str(route))
#                 else:
#                     logContents += '\t k%d, dt %.2f; %d;\t %s\n' % (k, detourTime, 0, str(route))
#             p_b.append(p)
#             logContents += '\t\t\t\t\t\t %.3f\n' % p
#         if LOGGING_FEASIBILITY:
#             record_logs(log_fpath, logContents)
#     else:
#         B, p_b, e_bi = None, None, None
#         with open(tsl_fpath, 'rb') as fp:
#             B, p_b, e_bi = pickle.load(fp)
#     #
#     # Define decision variables
#     #
#     cgMM = Model('materProblem')
#     q_b = {}
#     for b in range(len(B)):
#         q_b[b] = cgMM.addVar(vtype=GRB.BINARY, name="q[%d]" % b)
#     cgMM.update()
#     #
#     # Define objective
#     #
#     obj = LinExpr()
#     for b in range(len(B)):
#         obj += p_b[b] * q_b[b]
#     cgMM.setObjective(obj, GRB.MAXIMIZE)
#     #
#     # Define constrains
#     #
#     taskAC = {}
#     for i in T:  # eq:taskA
#         taskAC[i] = cgMM.addConstr(quicksum(e_bi[b][i] * q_b[b] for b in range(len(B))) == 1, name="taskAC[%d]" % i)
#     numBC = cgMM.addConstr(quicksum(q_b[b] for b in range(len(B))) == bB, name="numBC")
#     cgMM.update()
#     #
#     counter = 0
#     while True:
#         if len(B) == len(T) ** 2 - 1:
#             break
#         counter += 1
#         relax = cgMM.relax()
#         relax.Params.OutputFlag = X_GL
#         relax.optimize()
#         #
#         pi_i = [relax.getConstrByName("taskAC[%d]" % i).Pi for i in T]
#         mu = relax.getConstrByName("numBC").Pi
#         startCpuTimeS, startWallTimeS = time.clock(), time.time()
#         bestSols = pricingWIE_run(pi_i, mu, B, input4subProblem, counter, log_fpath, numThreads, TimeLimit, pfCst)
#         if bestSols is None:
#             logContents = '\n\n'
#             logContents += '%dth iteration (%s)\n' % (counter, str(datetime.now()))
#             logContents += 'No solution!\n' % (counter, str(datetime.now()))
#             break
#         endCpuTimeS, endWallTimeS = time.clock(), time.time()
#         eliCpuTimeS, eliWallTimeS = endCpuTimeS - startCpuTimeS, endWallTimeS - startWallTimeS
#         logContents = '\n\n'
#         logContents += '%dth iteration (%s)\n' % (counter, str(datetime.now()))
#         logContents += '\t Cpu Time\n'
#         logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeS)
#         logContents += '\t\t End.Time: %s\n' % str(endCpuTimeS)
#         logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeS
#         logContents += '\t Wall Time\n'
#         logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeS)
#         logContents += '\t\t End.Time: %s\n' % str(endWallTimeS)
#         logContents += '\t\t Eli.Time: %f\n' % eliWallTimeS
#         logContents += '\t Dual V\n'
#         logContents += '\t\t Pi: %s\n' % str([round(v, 3) for v in pi_i])
#         logContents += '\t\t mu: %.3f\n' % mu
#         numPRC = 0
#         for c_b, bundle in bestSols:
#             if c_b < EPSILON:
#                 continue
#             numPRC += 1
#             logContents += '\t New B. %d\n' % numPRC
#             logContents += '\t\t Tasks %s\n' % str(bundle)
#             logContents += '\t\t red. C. %.3f\n' % c_b
#             vec = [0 for _ in range(len(T))]
#             for i in bundle:
#                 vec[i] = 1
#             p = c_b + (np.array(vec) * np.array(pi_i)).sum() + mu
#             e_bi.append(vec)
#             p_b.append(p)
#             #
#             col = Column()
#             for i in range(len(T)):
#                 if e_bi[len(B)][i] > 0:
#                     col.addTerms(e_bi[len(B)][i], taskAC[i])
#             col.addTerms(1, numBC)
#             #
#             q_b[len(B)] = cgMM.addVar(obj=p_b[len(B)], vtype=GRB.BINARY, name="q[%d]" % len(B), column=col)
#             B.append(bundle)
#             cgMM.update()
#         if numPRC == 0:
#             logContents += '\t No bundles\n'
#             record_logs(log_fpath, logContents)
#             break
#         else:
#             record_logs(log_fpath, logContents)
#
#
#     if not opath.exists(tsl_fpath):
#         with open(tsl_fpath, 'wb') as fp:
#             pickle.dump([B, p_b, e_bi], fp)
#
#
#     #
#     # Run Gurobi (Optimization)
#     #
#     gurobi_settings(cgMM)
#     cgMM.optimize()
#     #
#     endCpuTimeM, endWallTimeM = time.clock(), time.time()
#     eliCpuTimeM, eliWallTimeM = endCpuTimeM - startCpuTimeM, endWallTimeM - startWallTimeM
#     chosenB = [B[b] for b in range(len(B)) if q_b[b].x > 0.5]
#     #
#     logContents = '\n\n'
#     logContents += 'Summary\n'
#     logContents += '\t Cpu Time\n'
#     logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeM)
#     logContents += '\t\t End.Time: %s\n' % str(endCpuTimeM)
#     logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeM
#     logContents += '\t Wall Time\n'
#     logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeM)
#     logContents += '\t\t End.Time: %s\n' % str(endWallTimeM)
#     logContents += '\t\t Eli.Time: %f\n' % eliWallTimeM
#     logContents += '\t ObjV: %.3f\n' % cgMM.objVal
#     logContents += '\t Gap: %.3f\n' % cgMM.MIPGap
#     logContents += '\t chosen B.: %s\n' % str(chosenB)
#     record_logs(log_fpath, logContents)
#     #
#     relax = cgMM.relax()
#     relax.Params.OutputFlag = X_GL
#     relax.optimize()
#
#     print('\n')
#     print(B)
#     print('relax', [(b, relax.getVarByName("q[%d]" % b).x) for b in range(len(B))])
#     q_b_frac = []
#     mostDeviArg, mostDeviV = None, 1
#     for b in range(len(B)):
#         q_b_var = relax.getVarByName("q[%d]" % b).x
#         if not(abs(q_b_var - 0) < EPSILON or abs(q_b_var - 1) < EPSILON):
#             q_b_frac.append(b)
#             devi = abs(q_b_var - 0.5)
#             if len(B[b]) == 1:
#                 continue
#             if devi < mostDeviV:
#                 mostDeviArg, mostDeviV = b, devi
#
#     print([(b, B[b]) for b in q_b_frac])
#     print(B[mostDeviArg], mostDeviArg, mostDeviV)
#
#     leftChdCond, rightChdCond = [], []
#
#
#     # branchingTarget = sample(B[mostDeviArg], 2)
#     # branchingTarget = sample(B[22], 2)
#     branchingTarget = [0, 5]
#     for branchingTarget in [[0, 5],
#                             [7, 4],
#                             [2, 8], [2, 9], [8, 9],
#                             [0, 6], [0, 8], [6, 8],
#                             [0, 2], [0, 3], [2, 3],
#                             [4, 6], [4, 9], [6, 9], ]:
#         leftChdCond.append(branchingTarget)
#         rightChdCond.append(branchingTarget)
#         #
#         # Left Child
#         #
#
#         bestSolsL = pricingWIE_run(pi_i, mu, B, input4subProblem,
#                                          counter, log_fpath, numThreads, TimeLimit, pfCst,
#                                          leftChdCond, [])
#         bestSolsR = pricingWIE_run(pi_i, mu, B, input4subProblem,
#                                          counter, log_fpath, numThreads, TimeLimit, pfCst,
#                                          [], rightChdCond)
#         print('\n')
#         print(branchingTarget)
#         print('Left Child')
#         print(bestSolsL)
#         print('Right Child')
#         print(bestSolsR)
#
#
#
#
#     return cgMM.objVal, cgMM.MIPGap, eliCpuTimeM, eliWallTimeM