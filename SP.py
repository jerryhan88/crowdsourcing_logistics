import time
from gurobipy import *
#
from _util import log2file
from _util import set_grbSettings


def run(ori_inputs, bnp_inputs, etcSetting, grbSetting):
    def callbackF(m, where):
        if where == GRB.callback.MIPSOL:
            selectedTasks = [i for i in m._T if m.cbGetSolution(m._z_i[i]) > 0.5]
            for c in m._C:
                if len(selectedTasks) == len(c):
                    m.cbLazy(quicksum(m._z_i[i] for i in c) <= len(c) - 1)
        #
        if where == GRB.Callback.MIP:
            if time.clock() - etcSetting['startTS'] > etcSetting['TimeLimit']:
                logContents = 'Interrupted by time limit\n'
                log2file(etcSetting['LogFile'], logContents)
                m._is_terminated = True
                m.terminate()
    #
    T, r_i, v_i, _lambda = [ori_inputs.get(k) for k in ['T', 'r_i', 'v_i', '_lambda']]
    P, D, N = [ori_inputs.get(k) for k in ['P', 'D', 'N']]
    K, w_k, t_ij, _delta = [ori_inputs.get(k) for k in ['K', 'w_k', 't_ij', '_delta']]
    #
    C = bnp_inputs['C']
    pi_i, mu = [bnp_inputs.get(k) for k in ['pi_i', 'mu']]
    inclusiveC, exclusiveC = [bnp_inputs.get(k) for k in ['inclusiveC', 'exclusiveC']]
    #
    bigM1 = sum(r_i)
    bigM2 = len(N) + 2
    bigM3 = len(N) * max(t_ij.values())
    #
    # Define decision variables
    #
    SP = Model('SP')
    z_i, y_k, a_k, o_ki, x_kij, s_PN = {}, {}, {}, {}, {}, {}
    for i in T:
        z_i[i] = SP.addVar(vtype=GRB.BINARY, name='z[%d]' % i)
    for k in K:
        y_k[k] = SP.addVar(vtype=GRB.BINARY, name='y[%d]' % k)
        a_k[k] = SP.addVar(vtype=GRB.CONTINUOUS, name='a[%d]' % k)
    for k in K:
        kN = N.union({'ori%d' % k, 'dest%d' % k})
        for i in kN:
            o_ki[k, i] = SP.addVar(vtype=GRB.INTEGER, name='o[%d,%s]' % (k, i))
            for j in kN:
                x_kij[k, i, j] = SP.addVar(vtype=GRB.BINARY, name='x[%d,%s,%s]' % (k, i, j))
    for s in range(len(inclusiveC)):
        sP, sN = 'sP[%d]' % s, 'sN[%d]' % s
        s_PN[sP] = SP.addVar(vtype=GRB.BINARY, name=sP)
        s_PN[sN] = SP.addVar(vtype=GRB.BINARY, name=sN)
    R = SP.addVar(vtype=GRB.CONTINUOUS, name='R')
    SP.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for k in K:
        obj += w_k[k] * a_k[k]
    for i in T:
        obj -= pi_i[i] * z_i[i]
    obj -= mu
    SP.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    #  Branch and Bound
    #
    for s in range(len(inclusiveC)):
        i0, i1 = inclusiveC[s]
        sP, sN = 'sP[%d]' % s, 'sN[%d]' % s
        #  # eq:pInclusiveC
        SP.addConstr(s_PN[sP] + s_PN[sN] == 1, name='iC[%d]' % s)
        #  # eq:pInclusiveCP
        SP.addConstr(2 * s_PN[sP] <= z_i[i0] + z_i[i1], name='iCP[%d]' % s)
        #  # eq:pInclusiveCN
        SP.addConstr(z_i[i0] + z_i[i1] <= 2 * (1 - s_PN[sN]), name='iCN[%d]' % s)
    for i, (i0, i1) in enumerate(exclusiveC):
        #  eq:pExclusiveC
        SP.addConstr(z_i[i0] + z_i[i1] <= 1, name='eC[%d]' % i)
    #
    #  Linearization
    #
    for k in K:  # eq:linAlpha
        SP.addConstr(a_k[k] <= bigM1 * y_k[k],
                    name='la1[%d]' % k)
        SP.addConstr(a_k[k] <= R,
                    name='la2[%d]' % k)
    #
    #  Bundle
    #
    #  # eq:sumReward
    SP.addConstr(quicksum(r_i[i] * z_i[i] for i in T) == R, name='sr')
    #  # eq:volTh
    SP.addConstr(quicksum(v_i[i] * z_i[i] for i in T) <= _lambda, name='vt')
    #
    #  Routing_Flow
    #
    for k in K:
        kN = N.union({'ori%d' % k, 'dest%d' % k})
        #  # eq:initFlow
        SP.addConstr(quicksum(x_kij[k, 'ori%d' % k, j] for j in kN) == 1, name='pPDo[%d]' % k)
        SP.addConstr(quicksum(x_kij[k, j, 'dest%d' % k] for j in kN) == 1, name='pPDi[%d]' % k)
        #  # eq:noInFlowOutFlow
        SP.addConstr(quicksum(x_kij[k, j, 'ori%d' % k] for j in kN) == 0, name='XpPDo[%d]' % k)
        SP.addConstr(quicksum(x_kij[k, 'dest%d' % k, j] for j in kN) == 0, name='XpPDi[%d]' % k)
        for i in T:
            #  # eq:taskOutFlow
            SP.addConstr(quicksum(x_kij[k, 'p%d' % i, j] for j in kN) == z_i[i],
                         name='tOF[%d,%d]' % (k, i))
            #  # eq:taskInFlow
            SP.addConstr(quicksum(x_kij[k, j, 'd%d' % i] for j in kN) == z_i[i],
                        name='tIF[%d,%d]' % (k, i))
        for i in N:  # eq:flowCon
            SP.addConstr(quicksum(x_kij[k, i, j] for j in kN) == quicksum(x_kij[k, j, i] for j in kN),
                        name='fc[%d,%s]' % (k, i))
    #
    #  Routing_Ordering
    #
    for k in K:
        N_kM = N.union({'dest%d' % k})
        for i in N_kM:
            #  # eq:initOrder
            SP.addConstr(2 <= o_ki[k, i], name='initO1[%d,%s]' % (k, i))
            SP.addConstr(o_ki[k, i] <= bigM2, name='initO2[%d,%s]' % (k, i))
            for j in N_kM:
                #  # eq:subEli
                SP.addConstr(o_ki[k, i] + 1 <= o_ki[k, j] + bigM2 * (1 - x_kij[k, i, j]),
                     name='subEli[%d,%s,%s]' % (k, i, j))
        for i in T:
            #  # eq:pdSequence
            SP.addConstr(o_ki[k, 'p%d' % i] <= o_ki[k, 'd%d' % i] + bigM2 * (1 - z_i[i]),
                         name='pdS[%d,%d]' % (k, i))
    #
    # Detour feasibility
    #
    for k in K:
        #  # eq:detourFeasibility
        kN = N.union({'ori%d' % k, 'dest%d' % k})
        LRS = LinExpr()
        for i in kN:
            for j in kN:
                LRS += t_ij[i, j] * x_kij[k, i, j]
        LRS -= t_ij['ori%d' % k, 'dest%d' % k]
        SP.addConstr(LRS <= _delta + bigM3 * (1 - y_k[k]), name='df[%d]' % k)
    #
    # Number of tasks limit
    #
    #  # eq:moreThanTwo
    SP.addConstr(quicksum(z_i[i] for i in T) >= 2, name='mTT')
    #
    # For callback function
    #
    SP.setParam('LazyConstraints', True)
    SP._T, SP._C = T, C
    SP._z_i = z_i
    SP._is_terminated = False
    #
    # Run Gurobi (Optimization)
    #
    set_grbSettings(SP, grbSetting)
    SP.optimize(callbackF)
    #
    if SP._is_terminated:
        return 'terminated'
    elif SP.status == GRB.Status.INFEASIBLE:
        logContents = '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
        logContents += '!!!!!!!!Pricing infeasible!!!!!!!!'
        logContents += '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
        log2file(etcSetting['LogFile'], logContents)
        return None
    else:
        return SP.objVal, [i for i in T if z_i[i].x > 0.5]