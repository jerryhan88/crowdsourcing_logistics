import os.path as opath
import time, datetime
import numpy as np
from gurobipy import *
#

from optRouting import run as optR_run
from pricing import run as pricing_run
prefix = 'greedyHeuristic'
pyx_fn, c_fn = '%s.pyx' % prefix, '%s.c' % prefix
if opath.exists(c_fn):
    if opath.getctime(c_fn) < opath.getmtime(pyx_fn):
        from setup import cythonize; cythonize(prefix)
else:
    from setup import cythonize; cythonize(prefix)
from greedyHeuristic import run as gHeuristic_run
#
from problems import *
from _utils.mm_utils import *
from _utils.recording import *


class BnPNode(object):
    def __init__(self, nid, probSetting, grbSetting, etcSetting):
        self.nid = nid
        self.probSetting, self.grbSetting, self.etcSetting = probSetting, grbSetting, etcSetting
        assert 'problem' in self.probSetting
        self.problem, self.inclusiveC, self.exclusiveC = list(map(self.probSetting.get,
                                                                  ['problem', 'inclusiveC', 'exclusiveC']))
        if not self.inclusiveC and not self.exclusiveC:
            assert 'B' not in self.probSetting
            assert 'p_b' not in self.probSetting
            assert 'e_bi' not in self.probSetting
            #
            # Generate initial bundles
            #
            self.B, self.p_b, self.e_bi = self.gen_initBundles()
            #
            record_bpt(self.etcSetting['bptFile'])
        else:
            self.B, self.p_b, self.e_bi = list(map(self.probSetting.get,
                                                   ['B', 'p_b', 'e_bi']))
        self.res = {}

    def __repr__(self):
        return 'bnbNode(%s)' % self.nid

    def gen_initBundles(self):
        bB, \
        T, r_i, v_i, _lambda, P, D, N, \
        K, w_k, t_ij, _delta = convert_input4MathematicalModel(*self.problem)
        #
        # Run the greedy heuristic
        #
        startCpuTime, startWallTime = time.clock(), time.time()
        objV, B, unassigned_tasks = gHeuristic_run(self.problem)
        gap, eliWallTime = None, None
        endCpuTime, endWallTime = time.clock(), time.time()
        eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
        #
        logContents = '\n\n'
        logContents += 'Summary\n'
        logContents += '\t Sta.Time: %s\n' % str(startCpuTime)
        logContents += '\t End.Time: %s\n' % str(endCpuTime)
        logContents += '\t Eli.Time: %f\n' % eliCpuTime
        logContents += '\t ObjV: %.3f\n' % objV
        logContents += '\t chosen B.: %s\n' % str(B)
        logContents += '\t unassigned T.: %d\n' % len(unassigned_tasks)
        record_logs(self.etcSetting['ghLogF'], logContents)
        record_res(self.etcSetting['ghResF'], objV, gap, eliCpuTime, eliWallTime)
        #
        logContents = '\n\n'
        logContents += '===========================================================\n'
        logContents = 'Initial bundles\n'
        for b in B:
            logContents += '\t %s\n' % str(b)
        logContents += '===========================================================\n'
        record_logs(self.etcSetting['bnpLogF'], logContents)
        #
        e_bi = []
        for b in B:
            vec = [0 for _ in range(len(T))]
            for i in b:
                vec[i] = 1
            e_bi.append(vec)
        #
        # Run the optimal routing
        #
        grbSettingOP = {'LogFile': self.etcSetting['orLogF'],
                         'Threads': self.grbSetting['Threads']}
        startCpuTime, startWallTime = time.clock(), time.time()
        p_b = []
        logContents = 'Bundle-Path feasibility\n'
        for b in range(len(B)):
            bundle = [i for i, v in enumerate(e_bi[b]) if v == 1]
            br = sum([r_i[i] for i in bundle])
            logContents += '%s (%d) \n' % (str(bundle), br)
            p = 0
            for k, w in enumerate(w_k):
                probSetting = {'b': bundle, 'k': k, 't_ij': t_ij}
                detourTime, route = optR_run(probSetting, grbSettingOP)
                if detourTime <= _delta:
                    p += w * br
                    logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w, detourTime, 1, str(route))
                else:
                    logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w, detourTime, 0, str(route))
            p_b.append(p)
            logContents += '\t\t\t\t\t\t %.3f\n' % p
        objV, gap = sum(p_b), None
        endCpuTime, endWallTime = time.clock(), time.time()
        eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
        #
        logContents += '\n\n'
        logContents += 'Summary\n'
        logContents += '\t Sta.Time: %s\n' % str(startCpuTime)
        logContents += '\t End.Time: %s\n' % str(endCpuTime)
        logContents += '\t Eli.Time: %f\n' % eliCpuTime
        logContents += '\t ObjV: %.3f\n' % objV
        logContents += '\t chosen B.: %s\n' % str(B)
        record_logs(self.etcSetting['orLogF'], logContents)
        record_res(self.etcSetting['orResF'], objV, gap, eliCpuTime, eliWallTime)
        #
        return B, p_b, e_bi

    def startCG(self):
        startCpuTimeM, startWallTimeM = time.clock(), time.time()
        logContents = '\n\n'
        logContents += '===========================================================\n'
        logContents += '%s\n' % str(datetime.datetime.now())
        logContents += 'Start column generation of bnbNode(%s)\n' % self.nid
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
        counter, is_feasible = 0, True
        while True:
            if len(B) == len(T) ** 2 - 1:
                break
            relaxM = masterM.relax()
            set_grbSettings(relaxM, self.grbSetting)
            relaxM.optimize()
            if relaxM.status == GRB.Status.INFEASIBLE:
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
            bestSols = pricing_run(counter,
                                   pi_i, mu, B, input4subProblem,
                                   self.inclusiveC, self.exclusiveC,
                                   self.grbSetting)
            if bestSols is None:
                logContents = '\n\n'
                logContents += '%dth iteration (%s)\n' % (counter, str(datetime.datetime.now()))
                logContents += 'No solution!\n'
                break
            endCpuTimeP, endWallTimeP = time.clock(), time.time()
            eliCpuTimeP, eliWallTimeP = endCpuTimeP - startCpuTimeP, endWallTimeP - startWallTimeP
            #
            logContents = '\n\n'
            logContents += '%dth iteration (%s)\n' % (counter, str(datetime.datetime.now()))
            logContents += '\t Cpu Time\n'
            logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeP)
            logContents += '\t\t End.Time: %s\n' % str(endCpuTimeP)
            logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeP
            logContents += '\t Wall Time\n'
            logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeP)
            logContents += '\t\t End.Time: %s\n' % str(endWallTimeP)
            logContents += '\t\t Eli.Time: %f\n' % eliWallTimeP
            logContents += '\t Relaxed objVal\n'
            logContents += '\t\t z: %.3f\n' % relaxM.objVal
            logContents += '\t Dual V\n'
            logContents += '\t\t Pi: %s\n' % str([round(v, 3) for v in pi_i])
            logContents += '\t\t mu: %.3f\n' % mu
            posBundles = []
            for c_b, bundle in bestSols:
                if c_b < EPSILON:
                    continue
                posBundles.append(bundle)
                logContents += '\t New B. %d\n' % len(posBundles)
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
            record_bpt(self.etcSetting['bptFile'], [self.nid,
                                         datetime.datetime.fromtimestamp(startWallTimeP),
                                         eliWallTimeP, eliCpuTimeP,
                                         'aI',
                                         {'numIter': counter, 'numGBs': len(posBundles), 'GBs': posBundles}])
            if not posBundles:
                logContents += '\t No new bundles\n'
                record_logs(self.grbSetting['LogFile'], logContents)
                break
            else:
                record_logs(self.grbSetting['LogFile'], logContents)
        #
        if not self.inclusiveC and not self.exclusiveC:
            self.rootMasterM = masterM
            self.rootB = B
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
            record_logs(self.grbSetting['LogFile'], logContents)
            #
            self.probSetting['B'] = B
            self.probSetting['p_b'] = p_b
            self.probSetting['e_bi'] = e_bi
            #
            self.res['objVal'] = relaxM.objVal
            self.res['q_b'] = q_b
            #
            record_bpt(self.etcSetting['bptFile'], [self.nid,
                                         datetime.datetime.fromtimestamp(startWallTimeM),
                                         eliWallTimeM, eliCpuTimeM,
                                         'CG',
                                         {'objVal': relaxM.objVal,
                                          'inclusiveC': str(self.inclusiveC),
                                          'exclusiveC': str(self.exclusiveC)}])
        else:
            endCpuTimeM, endWallTimeM = time.clock(), time.time()
            eliCpuTimeM, eliWallTimeM = endCpuTimeM - startCpuTimeM, endWallTimeM - startWallTimeM
            record_bpt(self.etcSetting['bptFile'], [self.nid,
                                         datetime.datetime.fromtimestamp(startWallTimeM),
                                         eliWallTimeM, eliCpuTimeM,
                                         'CG',
                                         {'objVal': -1,
                                          'inclusiveC': str(self.inclusiveC),
                                          'exclusiveC': str(self.exclusiveC)}])
        #
        return is_feasible

    def solveRootMIP(self):
        masterM, B = self.rootMasterM, self.rootB
        startCpuTime, startWallTime = time.clock(), time.time()
        grbSettingR = {'LogFile': self.etcSetting['cgLogF'],
                       'Threads': self.grbSetting['Threads']}
        set_grbSettings(masterM, grbSettingR)
        masterM.optimize()
        q_b = [masterM.getVarByName("q[%d]" % b).x for b in range(len(B))]
        chosenB = [B[b] for b in range(len(B)) if q_b[b] > 0.5]
        endCpuTime, endWallTime = time.clock(), time.time()
        eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
        #
        logContents = '\n\n'
        logContents += 'Summary\n'
        logContents += '\t Sta.Time: %s\n' % str(startCpuTime)
        logContents += '\t End.Time: %s\n' % str(endCpuTime)
        logContents += '\t Eli.Time: %f\n' % eliCpuTime
        logContents += '\t ObjV: %.3f\n' % masterM.objVal
        logContents += '\t chosen B.: %s\n' % str(chosenB)
        record_logs(self.etcSetting['cgLogF'], logContents)
        record_res(self.etcSetting['cgResF'], masterM.objVal, masterM.MIPGap, eliCpuTime, eliWallTime)

