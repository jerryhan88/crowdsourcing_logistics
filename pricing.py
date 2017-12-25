import os.path as opath
from gurobipy import *
#
prefix = 'gh_sBundling'
pyx_fn, c_fn = '%s.pyx' % prefix, '%s.c' % prefix
if opath.exists(c_fn):
    if opath.getctime(c_fn) < opath.getmtime(pyx_fn):
        from setup import cythonize; cythonize(prefix)
else:
    from setup import cythonize; cythonize(prefix)
from gh_sBundling import run as ghS_run
#
from _utils.recording import *
from _utils.mm_utils import *


# PRICING_TIME_LIMIT = 60 * 10
PRICING_TIME_LIMIT = 60 * 2


def run(counter, inputs, grbSetting, etcSetting):
    pi_i, mu, B = [inputs.get(k) for k in ['pi_i', 'mu', 'B']]
    inclusiveC, exclusiveC = [inputs.get(k) for k in ['inclusiveC', 'exclusiveC']]
    #
    T, r_i, v_i, _lambda = [inputs.get(k) for k in ['T', 'r_i', 'v_i', '_lambda']]
    P, D, N = [inputs.get(k) for k in ['P', 'D', 'N']]
    K, w_k, t_ij, _delta = [inputs.get(k) for k in ['K', 'w_k', 't_ij', '_delta']]
    #
    def process_callback(pricingM, where):
        if where == GRB.callback.MIPSOL:
            tNodes = []
            selectedTasks = set()
            for i in pricingM._T:
                if pricingM.cbGetSolution(pricingM._z_i[i]) > 0.5:
                    tNodes.append('p%d' % i); tNodes.append('d%d' % i)
                    selectedTasks.add(i)
            for k in pricingM._K:
                ptNodes = tNodes[:] + ['ori%d' % k, 'dest%d' % k]
                selectedEdges = [(i, j) for j in ptNodes for i in ptNodes if pricingM.cbGetSolution(pricingM._x_kij[k, i, j]) > 0.5]
                route = get_routeFromOri(selectedEdges, ptNodes)
                if len(route) != len(ptNodes) - 1:
                    expr = 0
                    for i, j in route:
                        expr += pricingM._x_kij[k, i, j]
                    pricingM.cbLazy(expr <= len(route) - 1)  # eq:subTourElim

        if where == GRB.callback.MIP and pricingM.cbGet(GRB.Callback.MIP_SOLCNT):
            runTime = pricingM.cbGet(GRB.callback.RUNTIME)
            objbst = pricingM.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = pricingM.cbGet(GRB.Callback.MIP_OBJBND)
            gap = abs(objbst - objbnd) / (1.0 + abs(objbst))
            timeIntv = runTime - pricingM._lastGapUpTime
            #
            if gap < pricingM._minGap:
                pricingM._minGap = gap
                pricingM._lastGapUpTime = runTime
            else:
                gapPct = gap * 100
                if PRICING_TIME_LIMIT < timeIntv:
                    logContents = '\n\n'
                    logContents += 'Termination\n'
                    logContents += '\t gapPct: %.2f \n' % gapPct
                    logContents += '\t timeIntv: %f \n' % timeIntv
                    record_log(grbSetting['LogFile'], logContents)
                    pricingM.terminate()
    #
    bigM1 = sum(r_i) * 2
    bigM2 = (len(T) + 1) * 2
    bigM3 = sum(t_ij.values())
    #
    # Define decision variables
    #
    pricingM = Model('PricingProblem %d' % counter)
    z_i, y_k, a_k, o_ki, x_kij, s_PN = {}, {}, {}, {}, {}, {}
    for i in T:
        z_i[i] = pricingM.addVar(vtype=GRB.BINARY, name='z[%d]' % i)
    for k in K:
        y_k[k] = pricingM.addVar(vtype=GRB.BINARY, name='y[%d]' % k)
        a_k[k] = pricingM.addVar(vtype=GRB.CONTINUOUS, name='a[%d]' % k)
    for k in K:
        kN = N.union({'ori%d' % k, 'dest%d' % k})
        for i in kN:
            o_ki[k, i] = pricingM.addVar(vtype=GRB.INTEGER, name='o[%d,%s]' % (k, i))
            for j in kN:
                x_kij[k, i, j] = pricingM.addVar(vtype=GRB.BINARY, name='x[%d,%s,%s]' % (k, i, j))
    for s in range(len(inclusiveC)):
        sP, sN = 'sP[%d]' % s, 'sN[%d]' % s
        s_PN[sP] = pricingM.addVar(vtype=GRB.BINARY, name=sP)
        s_PN[sN] = pricingM.addVar(vtype=GRB.BINARY, name=sN)
    pricingM.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for k in K:
        obj += w_k[k] * a_k[k]
    for i in T:
        obj -= pi_i[i] * z_i[i]
    obj -= mu
    pricingM.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    #
    # Handling inclusive constraints
    for s in range(len(inclusiveC)):
        i0, i1 = inclusiveC[s]
        sP, sN = 'sP[%d]' % s, 'sN[%d]' % s
        pricingM.addConstr(s_PN[sP] + s_PN[sN] == 1,
                           name='iC[%d]' % s)
        pricingM.addConstr(2 * s_PN[sP] <= z_i[i0] + z_i[i1],
                           name='iCP[%d]' % s)
        pricingM.addConstr(z_i[i0] + z_i[i1] <= 2 * (1 - s_PN[sN]),
                           name='iCN[%d]' % s)
    # Handling exclusive constraints
    for i, (i0, i1) in enumerate(exclusiveC):
        pricingM.addConstr(z_i[i0] + z_i[i1] <= 1,
                           name='eC[%d]' % i)
    # Linearization
    for k in K:  # eq:linAlpha
        pricingM.addConstr(a_k[k] >= quicksum(r_i[i] * z_i[i] for i in T) - bigM1 * (1 - y_k[k]),
                    name='la1[%d]' % k)
        pricingM.addConstr(a_k[k] <= bigM1 * y_k[k],
                    name='la2[%d]' % k)
        pricingM.addConstr(a_k[k] <= quicksum(r_i[i] * z_i[i] for i in T),
                    name='la3[%d]' % k)
    #
    # Volume
    pricingM.addConstr(quicksum(v_i[i] * z_i[i] for i in T) <= _lambda,
                name='vt')  # eq:volTh
    #
    # Flow based routing
    for k in K:
        #  # eq:circularF
        pricingM.addConstr(x_kij[k, 'dest%d' % k, 'ori%d' % k] == 1,
                    name='cf[%d]' % k)
        #  # eq:pathPD
        pricingM.addConstr(quicksum(x_kij[k, 'ori%d' % k, j] for j in P) == 1,
                    name='pPDo[%d]' % k)
        pricingM.addConstr(quicksum(x_kij[k, i, 'dest%d' % k] for i in D) == 1,
                    name='pPDi[%d]' % k)
        #  # eq:XpathPD
        pricingM.addConstr(quicksum(x_kij[k, 'ori%d' % k, j] for j in D) == 0,
                    name='XpPDo[%d]' % k)
        pricingM.addConstr(quicksum(x_kij[k, i, 'dest%d' % k] for i in P) == 0,
                    name='XpPDi[%d]' % k)
        #
        for i in T:  # eq:taskOutFlow
            pricingM.addConstr(quicksum(x_kij[k, 'p%d' % i, j] for j in N) == z_i[i],
                        name='tOF[%d,%d]' % (k, i))
        for j in T:  # eq:taskInFlow
            pricingM.addConstr(quicksum(x_kij[k, i, 'd%d' % j] for i in N) == z_i[j],
                        name='tIF[%d,%d]' % (k, j))
        #
        kP, kM = 'ori%d' % k, 'dest%d' % k
        kN = N.union({kP, kM})
        for i in kN:  # eq:XselfFlow
            pricingM.addConstr(x_kij[k, i, i] == 0,
                        name='Xsf[%d,%s]' % (k, i))
        for j in kN:  # eq:flowCon
            pricingM.addConstr(quicksum(x_kij[k, i, j] for i in kN) == quicksum(x_kij[k, j, i] for i in kN),
                        name='fc[%d,%s]' % (k, j))
        for i in kN:
            for j in kN:  # eq:direction
                if i == j: continue
                pricingM.addConstr(x_kij[k, i, j] + x_kij[k, j, i] <= 1,
                            name='dir[%d,%s,%s]' % (k, i, j))
        #
        pricingM.addConstr(o_ki[k, kP] == 1,
                    name='iOF[%d]' % k)
        pricingM.addConstr(o_ki[k, kM] <= bigM2,
                    name='iOE[%d]' % k)
        for i in T:  # eq:pdSequnce
            pricingM.addConstr(o_ki[k, 'p%d' % i] <= o_ki[k, 'd%d' % i] + bigM2 * (1 - z_i[i]),
                        name='pdS[%d]' % k)
        for i in kN:
            for j in kN:  # eq:ordering
                if i == j: continue
                if i == kM or j == kP: continue
                pricingM.addConstr(o_ki[k, i] + 1 <= o_ki[k, j] + bigM2 * (1 - x_kij[k, i, j]))
        #
        # Feasibility
        #  # eq:pathFeasibility
        pricingM.addConstr(quicksum(t_ij[i, j] * x_kij[k, i, j] for i in kN for j in kN) \
                    - t_ij[kM, kP] - t_ij[kP, kM] - _delta <= bigM3 * (1 - y_k[k]),
                    name='pf[%d]' % k)



    #
    # For callback function
    #
    pricingM.params.LazyConstraints = 1
    pricingM._B, pricingM._T, pricingM._K = B, T, K
    pricingM._z_i, pricingM._x_kij = z_i, x_kij
    pricingM._minGap = GRB.INFINITY
    pricingM._lastGapUpTime = -GRB.INFINITY
    #
    # Run Gurobi (Optimization)
    #
    if etcSetting['use_ghS']:
        rc, dvs = ghS_run(inputs)
        logContents = '\n\n'
        logContents += 'Initial solution\n'
        logContents += '\t rc: %.2f \n' % rc
        logContents += '\t b: %s \n' % str([i for i in T if dvs['_z_i'][i] > 0.5])
        record_log(grbSetting['LogFile'], logContents)
        #
        for i in T:
            z_i[i].start = dvs['_z_i'][i]
        for k in K:
            y_k[k].start = dvs['_y_k'][k]
            a_k[k].start = dvs['_a_k'][k]
        for k in K:
            kN = N.union({'ori%d' % k, 'dest%d' % k})
            for i in kN:
                o_ki[k, i].start = dvs['_o_ki'][k, i]
                for j in kN:
                    x_kij[k, i, j].start = dvs['_x_kij'][k, i, j]
        for s in range(len(inclusiveC)):
            sP, sN = 'sP[%d]' % s, 'sN[%d]' % s
            s_PN[sP].start = dvs['_s_PN'][sP]
            s_PN[sN].start = dvs['_s_PN'][sN]


    if 'mpsF' in etcSetting:
        pricingM.write(etcSetting['mpsF'])
        assert False, 'Done a mps file writing'
    #
    set_grbSettings(pricingM, grbSetting)
    #
    pricingM.optimize(process_callback)
    if LOGGING_FEASIBILITY:
        logContents = 'newBundle!!\n'
        logContents += '%s\n' % [i for i in T if z_i[i].x > 0.05]
        p = 0
        for k in K:
            kP, kM = 'ori%d' % k, 'dest%d' % k
            kN = N.union({kP, kM})
            detourTime = 0
            _route = {}
            for i in kN:
                for j in kN:
                    if x_kij[k, i, j].x > 0.5:
                        _route[i] = j
                        detourTime += t_ij[i, j]
            detourTime -= t_ij[kM, kP]
            detourTime -= t_ij[kP, kM]
            i = kP
            route = []
            while i != kM:
                route.append(i)
                i = _route[i]
            route.append(i)
            if int(y_k[k].x) > 0:
                logContents += '\t k%d, dt %.2f; %d;\t %s\n' % (k, detourTime, 1, str(route))
                p += w_k[k]
            else:
                logContents += '\t k%d, dt %.2f; %d;\t %s\n' % (k, detourTime, 0, str(route))
        logContents += '\t\t\t\t\t\t %.3f \t %.3f\n' % (pricingM.objVal, p)
        record_log(grbSetting['LogFile'], logContents)
    #
    if pricingM.status == GRB.Status.INFEASIBLE:
        logContents = '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
        logContents += '!!!!!!!!Pricing infeasible!!!!!!!!'
        logContents += '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
        record_log(grbSetting['LogFile'], logContents)
        return None
    else:
        nSolutions = pricingM.SolCount
        if nSolutions == 0:
            return None
        bestSols = []
        if nSolutions == 1:
            bestSols.append((pricingM.objVal, [i for i in T if z_i[i].x > 0.5]))
        else:
            bestSols = {}
            for e in range(nSolutions):
                pricingM.setParam(GRB.Param.SolutionNumber, e)
                bundle = tuple([i for i in T if z_i[i].Xn > 0.5])
                if bundle not in bestSols:
                    bestSols[bundle] = pricingM.PoolObjVal
            bestSols = [(objV, list(bundle)) for bundle, objV in bestSols.items()]
        return bestSols


if __name__ == '__main__':
    pass