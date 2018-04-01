import datetime, time
import numpy as np
from gurobipy import *
#
from _util import log2file, write_log, itr2file, res2file
from _util import set_grbSettings
from RMP import generate_RMP
from LS import run as LS_run
from PD import calc_expectedProfit
from problems import *

EPSILON = 0.000000001


def run(probSetting, etcSetting, grbSetting):
    startCpuTime, startWallTime = time.clock(), time.time()
    if 'TimeLimit' not in etcSetting:
        etcSetting['TimeLimit'] = 1e400
    etcSetting['startTS'] = startCpuTime
    etcSetting['startCpuTime'] = startCpuTime
    etcSetting['startWallTime'] = startWallTime
    itr2file(etcSetting['itrFile'])
    #
    probSetting['inputs'] = convert_p2i(*probSetting['problem'])
    #
    # Generate initial singleton bundles
    #
    inputs = probSetting['inputs']
    T, r_i, v_i, _lambda = [inputs.get(k) for k in ['T', 'r_i', 'v_i', '_lambda']]
    K, w_k = list(map(inputs.get, ['K', 'w_k']))
    t_ij, _delta = list(map(inputs.get, ['t_ij', '_delta']))
    C, sC, p_c, e_ci, TB = [], set(), [], [], set()
    for i in T:
        bc = [i]
        C.append(bc)
        sC.add(frozenset(tuple(bc)))
        #
        ep = calc_expectedProfit(probSetting, grbSetting, bc)
        p_c.append(ep)
        #
        vec = [0 for _ in range(len(T))]
        vec[i] = 1
        e_ci.append(vec)
    probSetting['C'] = C
    probSetting['sC'] = sC
    probSetting['p_c'] = p_c
    probSetting['e_ci'] = e_ci
    probSetting['TB'] = TB
    #
    RMP, q_c, taskAC, numBC = generate_RMP(probSetting)
    #
    write_log(etcSetting['LogFile'], 'Start column generation of CWL')
    #
    counter, is_terminated = 0, False
    while True:
        if len(C) == len(T) ** 2 - 1:
            break
        LRMP = RMP.relax()
        set_grbSettings(LRMP, grbSetting)
        LRMP.setParam('LogToConsole', False)
        LRMP.setParam('OutputFlag', False)
        LRMP.optimize()
        if LRMP.status == GRB.Status.INFEASIBLE:
            logContents = '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
            logContents += 'Relaxed model is infeasible!!\n'
            logContents += 'No solution!\n'
            logContents += '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
            log2file(etcSetting['LogFile'], logContents)
            LRMP.computeIIS()
            import os.path as opath
            LRMP.write('%s.ilp' % opath.basename(etcSetting['LogFile']).split('.')[0])
            LRMP.write('%s.lp' % opath.basename(etcSetting['LogFile']).split('.')[0])
            assert False
        #
        pi_i = [LRMP.getConstrByName("taskAC[%d]" % i).Pi for i in T]
        mu = LRMP.getConstrByName("numBC").Pi
        #
        write_log(etcSetting['LogFile'], 'Start column generation of CWL')
        logContents = 'Start %dth iteration\n' % counter
        logContents += '\t Columns\n'
        logContents += '\t\t # of columns %d\n' % len(probSetting['C'])
        logContents += '\t\t %s\n' % str(probSetting['C'])
        logContents += '\t\t %s\n' % str(['%.2f' % v for v in probSetting['p_c']])
        logContents += '\t Relaxed objVal\n'
        logContents += '\t\t z: %.2f\n' % LRMP.objVal
        logContents += '\t\t RC: %s\n' % str(['%.2f' % LRMP.getVarByName("q[%d]" % c).RC for c in range(len(C))])
        logContents += '\t Dual V\n'
        logContents += '\t\t Pi: %s\n' % str(['%.2f' % v for v in pi_i])
        logContents += '\t\t mu: %.2f\n' % mu
        write_log(etcSetting['LogFile'], logContents)
        #
        c0, minRC = -1, 1e400
        for rc, c in [(LRMP.getVarByName("q[%d]" % c).RC, c) for c in range(len(C))]:
        # for rc, c in sorted([(LRMP.getVarByName("q[%d]" % c).RC, c) for c in range(len(C))]):
            bc = C[c]
            if c in TB:
                continue
            if sum(v_i[i]for i in bc) == _lambda:
                continue
            if rc < minRC:
                minRC = rc
                c0 = c
        if c0 == -1:
            break
        #
        probSetting['pi_i'] = pi_i
        probSetting['mu'] = mu
        probSetting['c0'] = c0
        #
        startCpuTimeP, startWallTimeP = time.clock(), time.time()
        objV_bc = LS_run(probSetting, etcSetting, grbSetting)
        #
        if time.clock() - etcSetting['startTS'] > etcSetting['TimeLimit']:
            break
        #
        endCpuTimeP, endWallTimeP = time.clock(), time.time()
        eliCpuTimeP, eliWallTimeP = endCpuTimeP - startCpuTimeP, endWallTimeP - startWallTimeP
        #
        logContents = '%dth iteration (%s)\n' % (counter, str(datetime.datetime.now()))
        logContents += '\t Cpu Time\n'
        logContents += '\t\t Sta.Time: %s\n' % str(startCpuTimeP)
        logContents += '\t\t End.Time: %s\n' % str(endCpuTimeP)
        logContents += '\t\t Eli.Time: %f\n' % eliCpuTimeP
        logContents += '\t Wall Time\n'
        logContents += '\t\t Sta.Time: %s\n' % str(startWallTimeP)
        logContents += '\t\t End.Time: %s\n' % str(endWallTimeP)
        logContents += '\t\t Eli.Time: %f\n' % eliWallTimeP
        write_log(etcSetting['LogFile'], logContents)
        #
        objV, bc = objV_bc
        itr2file(etcSetting['itrFile'], [counter, '%.2f' % LRMP.objVal,
                                         C[c0], '%.2f' % minRC,
                                         str(bc), '%.2f' % objV,
                                         '%.2f' % eliCpuTimeP, '%.2f' % eliWallTimeP])
        if objV < 0:
            TB.add(c0)
        else:
            vec = [0 for _ in range(len(T))]
            for i in bc:
                vec[i] = 1
            p = objV + (np.array(vec) * np.array(pi_i)).sum() + mu
            C, p_c, e_ci = list(map(probSetting.get, ['C', 'p_c', 'e_ci']))
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
            sC.add(frozenset(tuple(bc)))
            RMP.update()
            #
            probSetting['C'] = C
            probSetting['sC'] = sC
            probSetting['p_c'] = p_c
            probSetting['e_ci'] = e_ci
        if len(C) == len(TB):
            break

        counter += 1
    #
    handle_termination(RMP, probSetting, etcSetting, grbSetting)




def handle_termination(RMP, probSetting, etcSetting, grbSetting):
    set_grbSettings(RMP, grbSetting)
    RMP.optimize()
    C, p_c, e_ci = list(map(probSetting.get, ['C', 'p_c', 'e_ci']))
    q_c = [RMP.getVarByName("q[%d]" % c).x for c in range(len(C))]
    chosenC = [(C[c], '%.2f' % q_c[c]) for c in range(len(C)) if q_c[c] > 0]
    #
    endCpuTime, endWallTime = time.clock(), time.time()
    eliCpuTime = endCpuTime - etcSetting['startCpuTime']
    eliWallTime = endWallTime - etcSetting['startWallTime']

    logContents = 'CWL model reach to the time limit or the end\n'
    logContents += '\n'
    logContents += 'Column generation summary\n'
    logContents += '\t Cpu Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(etcSetting['startCpuTime'])
    logContents += '\t\t End.Time: %s\n' % str(endCpuTime)
    logContents += '\t\t Eli.Time: %f\n' % eliCpuTime
    logContents += '\t Wall Time\n'
    logContents += '\t\t Sta.Time: %s\n' % str(etcSetting['startWallTime'])
    logContents += '\t\t End.Time: %s\n' % str(endWallTime)
    logContents += '\t\t Eli.Time: %f\n' % eliWallTime
    logContents += '\t ObjV: %.3f\n' % RMP.objVal
    logContents += '\t chosen B.: %s\n' % str(chosenC)
    write_log(etcSetting['LogFile'], logContents)
    #
    res2file(etcSetting['ResFile'], RMP.objVal, -1, eliCpuTime, eliWallTime)


if __name__ == '__main__':
    import os.path as opath
    from problems import paperExample, ex1
    #
    # problem = paperExample()
    # cwlLogF = opath.join('_temp', 'paperExample_CWL.log')
    # cwlResF = opath.join('_temp', 'paperExample_CWL.csv')
    # itrFile = opath.join('_temp', 'paperExample_itrCWL.csv')
    #
    problem = ex1()
    cwlLogF = opath.join('_temp', 'ex1_CWL.log')
    cwlResF = opath.join('_temp', 'ex1_CWL.csv')
    itrFile = opath.join('_temp', 'ex1_itrCWL.csv')
    probSetting = {'problem': problem}
    #
    etcSetting = {'LogFile': cwlLogF,
                  'ResFile': cwlResF,
                  'itrFile': itrFile,
                  'numPros': 8,
                  }
    grbSetting = {
        'LogFile': cwlLogF
                  }
    #
    run(probSetting, etcSetting, grbSetting)
