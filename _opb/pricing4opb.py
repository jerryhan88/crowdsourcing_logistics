from gurobipy import *
#
from _utils.recording import *
from _utils.mm_utils import *
from optRouting import run as optR_run


PRICING_TIME_LIMIT = 60 * 10


def run(counter,
        k, pi_i, mu, B, input4subProblem,
        inclusiveC, exclusiveC,
        grbSetting):
    def process_callback(pricingM, where):
        if where == GRB.callback.MIPSOL:
            tNodes = []
            selectedTasks = set()
            for i in pricingM._T:
                if pricingM.cbGetSolution(pricingM._z_i[i]) > 0.5:
                    tNodes.append('p0%d' % i); tNodes.append('d%d' % i)
                    selectedTasks.add(i)
            ptNodes = tNodes[:] + ['ori%d' % pricingM._k, 'dest%d' % pricingM._k]
            selectedEdges = [(i, j) for j in ptNodes for i in ptNodes if pricingM.cbGetSolution(pricingM._x_ij[i, j]) > 0.5]
            route = get_routeFromOri(selectedEdges, ptNodes)
            if len(route) != len(ptNodes) - 1:
                expr = 0
                for i, j in route:
                    expr += pricingM._x_ij[i, j]
                pricingM.cbLazy(expr <= len(route) - 1)  # eq:subTourElim
    #
    T, r_i, v_i, _lambda, P, D, N, w_k, t_ij, _delta = input4subProblem
    bigM1 = sum(r_i) * 2
    bigM2 = (len(T) + 1) * 2
    bigM3 = sum(t_ij.values())
    #
    # Define decision variables
    #
    kP, kM = 'ori%d' % k, 'dest%d' % k
    kN = N.union({kP, kM})
    #
    pricingM = Model('PricingProblem %d, %d' % (counter, k))
    z_i, y, a, o_i, x_ij = {}, {}, {}, {}, {}
    s_PN = {}
    #
    for i in T:
        z_i[i] = pricingM.addVar(vtype=GRB.BINARY, name='z[%d]' % i)
    y[k] = pricingM.addVar(vtype=GRB.BINARY, name='y')
    a[k] = pricingM.addVar(vtype=GRB.CONTINUOUS, name='a')
    for i in kN:
        o_i[i] = pricingM.addVar(vtype=GRB.INTEGER, name='o[%s]' % i)
        for j in kN:
            x_ij[i, j] = pricingM.addVar(vtype=GRB.BINARY, name='x[%s,%s]' % (i, j))
    for s in range(len(inclusiveC)):
        sP, sN = 'sP[%d]' % s, 'sN[%d]' % s
        s_PN[sP] = pricingM.addVar(vtype=GRB.BINARY, name=sP)
        s_PN[sN] = pricingM.addVar(vtype=GRB.BINARY, name=sN)
    pricingM.update()
    #
    # Define objective
    #
    obj = LinExpr()
    obj += a[k]
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
    #  # eq:linAlpha
    pricingM.addConstr(a[k] >= quicksum(r_i[i] * z_i[i] for i in T) - bigM1 * (1 - y[k]),
                name='la1')
    pricingM.addConstr(a[k] <= bigM1 * y[k],
                name='la2')
    pricingM.addConstr(a[k] <= quicksum(r_i[i] * z_i[i] for i in T),
                name='la3')
    #
    # Volume
    pricingM.addConstr(quicksum(v_i[i] * z_i[i] for i in T) <= _lambda,
                name='vt')  # eq:volTh
    #
    # Flow based routing
    #
    #  # eq:circularF
    pricingM.addConstr(x_ij[kM, kP] == 1,
                name='cf')
    #  # eq:pathPD
    pricingM.addConstr(quicksum(x_ij[kP, j] for j in P) == 1,
                name='pPDo')
    pricingM.addConstr(quicksum(x_ij[i, kM] for i in D) == 1,
                name='pPDi')
    #  # eq:XpathPD
    pricingM.addConstr(quicksum(x_ij[kP, j] for j in D) == 0,
                name='XpPDo')
    pricingM.addConstr(quicksum(x_ij[i, kM] for i in P) == 0,
                name='XpPDi')
    #
    for i in T:  # eq:taskOutFlow
        pricingM.addConstr(quicksum(x_ij['p0%d' % i, j] for j in N) == z_i[i],
                    name='tOF[%d]' % i)
    for j in T:  # eq:taskInFlow
        pricingM.addConstr(quicksum(x_ij[i, 'd%d' % j] for i in N) == z_i[j],
                    name='tIF[%d]' % j)
    #
    for i in kN:  # eq:XselfFlow
        pricingM.addConstr(x_ij[i, i] == 0,
                    name='Xsf[%s]' % i)
    for j in kN:  # eq:flowCon
        pricingM.addConstr(quicksum(x_ij[i, j] for i in kN) == quicksum(x_ij[j, i] for i in kN),
                    name='fc[%s]' % j)
    for i in kN:
        for j in kN:  # eq:direction
            if i == j: continue
            pricingM.addConstr(x_ij[i, j] + x_ij[j, i] <= 1,
                        name='dir[%s,%s]' % (i, j))
    #
    pricingM.addConstr(o_i[kP] == 1,
                name='iOF')
    pricingM.addConstr(o_i[kM] <= bigM2,
                name='iOE')
    for i in T:  # eq:pdSequnce
        pricingM.addConstr(o_i['p0%d' % i] <= o_i['d%d' % i] + bigM2 * (1 - z_i[i]),
                    name='pdS')
    for i in kN:
        for j in kN:  # eq:ordering
            if i == j: continue
            if i == kM or j == kP: continue
            pricingM.addConstr(o_i[i] + 1 <= o_i[j] + bigM2 * (1 - x_ij[i, j]))
    #
    # Feasibility
    #  # eq:pathFeasibility
    pricingM.addConstr(quicksum(t_ij[i, j] * x_ij[i, j] for i in kN for j in kN) \
                - t_ij[kM, kP] - t_ij[kP, kM] - _delta <= bigM3 * (1 - y[k]),
                name='pf')
    #
    # For callback function
    #
    pricingM.params.LazyConstraints = 1
    pricingM._B, pricingM._T = B, T
    pricingM._k = k
    pricingM._z_i, pricingM._x_ij = z_i, x_ij
    #
    # Run Gurobi (Optimization)
    #
    set_grbSettings(pricingM, grbSetting)
    pricingM.optimize(process_callback)
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
            bestSols.append((pricingM.objVal, [i for i in T if z_i[i].x > 0.05]))
        else:
            bestSols = {}
            for e in range(nSolutions):
                pricingM.setParam(GRB.Param.SolutionNumber, e)
                if pricingM.PoolObjVal < EPSILON:
                    continue
                bundle = tuple([i for i in T if z_i[i].Xn > 0.05])
                if bundle not in bestSols:
                    bestSols[bundle] = pricingM.PoolObjVal
            bestSols = [(objV, list(bundle)) for bundle, objV in bestSols.items()]
        return bestSols





if __name__ == '__main__':
    pass