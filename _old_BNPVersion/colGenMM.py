import os.path as opath
import time
import numpy as np
from datetime import datetime
from gurobipy import *
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
from pricing import run as pricing_run
#
from _utils.recording import *
from _utils.mm_utils import *


def run(problem, log_fpath=None, numThreads=None, TimeLimit=None, pfCst=None):
    startCpuTimeM, startWallTimeM = time.clock(), time.time()
    #
    # Solve a master problem
    #
    bB, \
    T, r_i, v_i, _lambda, P, D, N, \
    K, w_k, t_ij, _delta = convert_input4MathematicalModel(*problem)
    input4subProblem = [T, r_i, v_i, _lambda, P, D, N, K, w_k, t_ij, _delta]
    #
    # generate initial bundles with the greedy heuristic
    #
    _, _, B = gHeuristic_run(problem)
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
    if log_fpath:
        with open(log_fpath, 'wt') as f:
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
        record_log(log_fpath, logContents)
    #
    # Define decision variables
    #
    cgMM = Model('materProblem')
    q_b = {}
    for b in range(len(B)):
        q_b[b] = cgMM.addVar(vtype=GRB.BINARY, name="q[%d]" % b)
    cgMM.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for b in range(len(B)):
        obj += p_b[b] * q_b[b]
    cgMM.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    #
    taskAC = {}
    for i in T:  # eq:taskA
        taskAC[i] = cgMM.addConstr(quicksum(e_bi[b][i] * q_b[b] for b in range(len(B))) == 1, name="taskAC[%d]" % i)
    numBC = cgMM.addConstr(quicksum(q_b[b] for b in range(len(B))) == bB, name="numBC")
    cgMM.update()
    #
    counter = 0
    while True:
        if len(B) == len(T) ** 2 - 1:
            break
        counter += 1
        relax = cgMM.relax()
        relax.Params.OutputFlag = X_GL
        relax.optimize()
        #
        pi_i = [relax.getConstrByName("taskAC[%d]" % i).Pi for i in T]
        mu = relax.getConstrByName("numBC").Pi
        startCpuTimeS, startWallTimeS = time.clock(), time.time()
        bestSols = pricing_run(pi_i, mu, B, input4subProblem, counter, log_fpath, numThreads, TimeLimit, pfCst)
        if bestSols is None:
            logContents = '\n\n'
            logContents += '%dth iteration (%s)\n' % (counter, str(datetime.now()))
            logContents += 'No solution!\n'
            break
        endCpuTimeS, endWallTimeS = time.clock(), time.time()
        eliCpuTimeS, eliWallTimeS = endCpuTimeS - startCpuTimeS, endWallTimeS - startWallTimeS
        logContents = '\n\n'
        logContents += '%dth iteration (%s)\n' % (counter, str(datetime.now()))
        logContents += '\t Cpu Time\n'
        logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeS)
        logContents += '\t\t End.Time: %s\n' % str(endCpuTimeS)
        logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeS
        logContents += '\t Wall Time\n'
        logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeS)
        logContents += '\t\t End.Time: %s\n' % str(endWallTimeS)
        logContents += '\t\t Eli.Time: %f\n' % eliWallTimeS
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
            q_b[len(B)] = cgMM.addVar(obj=p_b[len(B)], vtype=GRB.BINARY, name="q[%d]" % len(B), column=col)
            B.append(bundle)
            cgMM.update()
        if numPRC == 0:
            logContents += '\t No bundles\n'
            record_log(log_fpath, logContents)
            break
        else:
            record_log(log_fpath, logContents)
    #
    # Settings
    #
    if TimeLimit is not None:
        cgMM.setParam('TimeLimit', TimeLimit)
    if numThreads is not None:
        cgMM.setParam('Threads', numThreads)
    if log_fpath is not None:
        cgMM.setParam('LogFile', log_fpath)
    cgMM.setParam('OutputFlag', X_GL)
    #
    # Run Gurobi (Optimization)
    #
    cgMM.optimize()
    #
    endCpuTimeM, endWallTimeM = time.clock(), time.time()
    eliCpuTimeM, eliWallTimeM = endCpuTimeM - startCpuTimeM, endWallTimeM - startWallTimeM
    chosenB = [B[b] for b in range(len(B)) if q_b[b].x > 0.5]
    #
    logContents = '\n\n'
    logContents += 'Summary\n'
    logContents += '\t Cpu Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeM)
    logContents += '\t\t End.Time: %s\n' % str(endCpuTimeM)
    logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeM
    logContents += '\t Wall Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeM)
    logContents += '\t\t End.Time: %s\n' % str(endWallTimeM)
    logContents += '\t\t Eli.Time: %f\n' % eliWallTimeM
    logContents += '\t ObjV: %.3f\n' % cgMM.objVal
    logContents += '\t Gap: %.3f\n' % cgMM.MIPGap
    logContents += '\t chosen B.: %s\n' % str(chosenB)
    record_log(log_fpath, logContents)
    #
    return cgMM.objVal, cgMM.MIPGap, eliCpuTimeM, eliWallTimeM


def calTime_test():
    import time
    cSTime, cTTime = time.clock(), time.time()

    run(ex1(), pfCst=1.2, log_fpath='temp.log')
    print(time.clock() - cSTime, time.time() - cTTime)


def test():
    import pickle
    ifpath = 'nt10-np20-nb4-tv3-td7.pkl'
    inputs = None
    with open(ifpath, 'rb') as fp:
        inputs = pickle.load(fp)

    print(inputs)
    _pfCst = 1.5
    objV, gap, eliCpuTime, eliWallTime = run(inputs, log_fpath='temp(%.2f).log' % _pfCst, pfCst=_pfCst)

    print(objV, gap, eliCpuTime, eliWallTime)


if __name__ == '__main__':
    test()
    # calTime_test()

