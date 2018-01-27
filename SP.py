import time
from gurobipy import *
#
from _util import log2file
from _util import set_grbSettings, get_routeFromOri


def run(probSetting, etcSetting, grbSetting):
    inputs = probSetting['inputs']
    T, r_i, v_i, _lambda = [inputs.get(k) for k in ['T', 'r_i', 'v_i', '_lambda']]
    P, D, N = [inputs.get(k) for k in ['P', 'D', 'N']]
    K, w_k, t_ij, _delta = [inputs.get(k) for k in ['K', 'w_k', 't_ij', '_delta']]
    C = probSetting['C']
    #
    pi_i, mu = [probSetting.get(k) for k in ['pi_i', 'mu']]
    inclusiveC, exclusiveC = [probSetting.get(k) for k in ['inclusiveC', 'exclusiveC']]
    #
    def callbackF(m, where):
        if where == GRB.callback.MIPSOL:
            selectedTasks = [i for i in m._T if m.cbGetSolution(m._z_i[i]) > 0.5]
            #
            tNodes = []
            for i in selectedTasks:
                tNodes.append('p%d' % i)
                tNodes.append('d%d' % i)
            for k in m._K:
                ptNodes = tNodes[:] + ['ori%d' % k, 'dest%d' % k]
                selectedEdges = [(i, j) for j in ptNodes for i in ptNodes if m.cbGetSolution(m._x_kij[k, i, j]) > 0.5]
                route = get_routeFromOri(selectedEdges, ptNodes)
                if len(route) != len(ptNodes) - 1:
                    m.cbLazy(quicksum(m._x_kij[k, i, j] for i, j in route) <= len(route) - 1)  # eq:subTourElim
            #
            for bc in m._C:
                if len(selectedTasks) == len(bc):
                    m.cbLazy(quicksum(m._z_i[i] for i in bc) <= len(bc) - 1)
        #
        if where == GRB.Callback.MIP:
            if time.clock() - etcSetting['startTS'] > etcSetting['TimeLimit']:
                logContents = '\n\n'
                logContents += '======================================================================================\n'
                logContents += 'Interrupted by time limit\n'
                logContents += '======================================================================================\n'
                log2file(etcSetting['LogFile'], logContents)
                m._is_terminated = True
                m.terminate()
    #
    bigM1 = sum(r_i) * 2
    bigM2 = (len(T) + 1) * 2
    bigM3 = sum(t_ij.values())
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
    #


    SP.addConstr(quicksum(z_i[i] for i in T) >= 2,
                 name='mTT')  # eq:moreThanTwo



    # Handling inclusive constraints
    for s in range(len(inclusiveC)):
        i0, i1 = inclusiveC[s]
        sP, sN = 'sP[%d]' % s, 'sN[%d]' % s
        SP.addConstr(s_PN[sP] + s_PN[sN] == 1,
                           name='iC[%d]' % s)
        SP.addConstr(2 * s_PN[sP] <= z_i[i0] + z_i[i1],
                           name='iCP[%d]' % s)
        SP.addConstr(z_i[i0] + z_i[i1] <= 2 * (1 - s_PN[sN]),
                           name='iCN[%d]' % s)
    # Handling exclusive constraints
    for i, (i0, i1) in enumerate(exclusiveC):
        SP.addConstr(z_i[i0] + z_i[i1] <= 1,
                           name='eC[%d]' % i)
    # Linearization
    for k in K:  # eq:linAlpha
        SP.addConstr(a_k[k] <= bigM1 * y_k[k],
                    name='la2[%d]' % k)
        SP.addConstr(a_k[k] <= quicksum(r_i[i] * z_i[i] for i in T),
                    name='la3[%d]' % k)
    #
    # Volume
    SP.addConstr(quicksum(v_i[i] * z_i[i] for i in T) <= _lambda,
                name='vt')  # eq:volTh
    #
    # Flow based routing
    for k in K:
        #  # eq:circularF
        SP.addConstr(x_kij[k, 'dest%d' % k, 'ori%d' % k] == 1,
                    name='cf[%d]' % k)
        #  # eq:pathPD
        SP.addConstr(quicksum(x_kij[k, 'ori%d' % k, j] for j in P) == 1,
                    name='pPDo[%d]' % k)
        SP.addConstr(quicksum(x_kij[k, i, 'dest%d' % k] for i in D) == 1,
                    name='pPDi[%d]' % k)
        #  # eq:XpathPD
        SP.addConstr(quicksum(x_kij[k, 'ori%d' % k, j] for j in D) == 0,
                    name='XpPDo[%d]' % k)
        SP.addConstr(quicksum(x_kij[k, i, 'dest%d' % k] for i in P) == 0,
                    name='XpPDi[%d]' % k)
        #
        for i in T:  # eq:taskOutFlow
            SP.addConstr(quicksum(x_kij[k, 'p%d' % i, j] for j in N) == z_i[i],
                        name='tOF[%d,%d]' % (k, i))
        for j in T:  # eq:taskInFlow
            SP.addConstr(quicksum(x_kij[k, i, 'd%d' % j] for i in N) == z_i[j],
                        name='tIF[%d,%d]' % (k, j))
        #
        kP, kM = 'ori%d' % k, 'dest%d' % k
        kN = N.union({kP, kM})
        for i in kN:  # eq:XselfFlow
            SP.addConstr(x_kij[k, i, i] == 0,
                        name='Xsf[%d,%s]' % (k, i))
        for j in kN:  # eq:flowCon
            SP.addConstr(quicksum(x_kij[k, i, j] for i in kN) == quicksum(x_kij[k, j, i] for i in kN),
                        name='fc[%d,%s]' % (k, j))
        for i in kN:
            for j in kN:  # eq:direction
                if i == j: continue
                SP.addConstr(x_kij[k, i, j] + x_kij[k, j, i] <= 1,
                            name='dir[%d,%s,%s]' % (k, i, j))
        #
        SP.addConstr(o_ki[k, kP] == 1,
                    name='iOF[%d]' % k)
        SP.addConstr(o_ki[k, kM] <= bigM2,
                    name='iOE[%d]' % k)
        for i in T:  # eq:pdSequnce
            SP.addConstr(o_ki[k, 'p%d' % i] <= o_ki[k, 'd%d' % i] + bigM2 * (1 - z_i[i]),
                        name='pdS[%d]' % k)
        for i in kN:
            for j in kN:  # eq:ordering
                if i == j: continue
                if i == kM or j == kP: continue
                SP.addConstr(o_ki[k, i] + 1 <= o_ki[k, j] + bigM2 * (1 - x_kij[k, i, j]))
        #
        # Feasibility
        #  # eq:pathFeasibility
        SP.addConstr(quicksum(t_ij[i, j] * x_kij[k, i, j] for i in kN for j in kN) \
                    - t_ij[kM, kP] - t_ij[kP, kM] - _delta <= bigM3 * (1 - y_k[k]),
                    name='pf[%d]' % k)
    #
    # For callback function
    #
    SP.params.LazyConstraints = 1
    SP._T, SP._K = T, K
    SP._z_i, SP._x_kij = z_i, x_kij
    SP._C = C
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