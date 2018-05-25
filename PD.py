import multiprocessing
from gurobipy import *
#
NUM_CORES = multiprocessing.cpu_count()


def run(prmt, pd_inputs):
    k, Ts = [pd_inputs.get(k) for k in ['k', 'Ts']]
    _kP, _kM = 'ori%d' % k, 'dest%d' % k
    P, D = set(), set()
    for i in Ts:
        P.add('p%d' % i)
        D.add('d%d' % i)
    N = P.union(D)
    kN = N.union({_kP, _kM})
    t_ij = prmt['t_ij']
    bigM = len(N) + 2
    #
    # Define decision variables
    #
    PD = Model('PD')
    x_ij, o_i = {}, {}
    for i in kN:
        o_i[i] = PD.addVar(vtype=GRB.INTEGER, name='o[%s]' % i)
        for j in kN:
            x_ij[i, j] = PD.addVar(vtype=GRB.BINARY, name='x[%s,%s]' % (i, j))
    PD.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for i in kN:
        for j in kN:
            obj += t_ij[i, j] * x_ij[i, j]
    obj -= t_ij[_kP, _kM]
    PD.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constrains
    #  Routing_Flow
    #
    #  # eq:initFlow
    PD.addConstr(quicksum(x_ij['ori%d' % k, j] for j in kN) == 1, name='pPDo')
    PD.addConstr(quicksum(x_ij[j, 'dest%d' % k] for j in kN) == 1, name='pPDi')
    #  # eq:noInFlowOutFlow
    PD.addConstr(quicksum(x_ij[j, 'ori%d' % k] for j in kN) == 0, name='XpPDo')
    PD.addConstr(quicksum(x_ij['dest%d' % k, j] for j in kN) == 0, name='XpPDi')
    for i in Ts:
        #  # eq:taskOutFlow
        PD.addConstr(quicksum(x_ij['p%d' % i, j] for j in kN) == 1, name='tOF[%d]' % i)
        #  # eq:taskInFlow
        PD.addConstr(quicksum(x_ij[j, 'd%d' % i] for j in kN) == 1, name='tIF[%d]' % i)
    for i in N:  # eq:flowCon
        PD.addConstr(quicksum(x_ij[i, j] for j in kN) == quicksum(x_ij[j, i] for j in kN),
                     name='fc[%s]' % i)
    #
    #  Routing_Ordering
    #
    N_kM = N.union({_kM})
    for i in N_kM:
        #  # eq:initOrder
        PD.addConstr(2 <= o_i[i], name='initO1[%s]' % i)
        PD.addConstr(o_i[i] <= bigM, name='initO2[%s]' % i)
        for j in N_kM:
                #  # eq:subEli
                PD.addConstr(o_i[i] + 1 <= o_i[j] + bigM * (1 - x_ij[i, j]),
                     name='subEli[%s,%s]' % (i, j))
    for i in Ts:
        #  # eq:pdSequence
        PD.addConstr(o_i['p%d' % i] <= o_i['d%d' % i], name='pdS[%d]' % i)
    #
    # Run Gurobi (Optimization)
    #
    PD.setParam('OutputFlag', False)
    PD.setParam('Threads', NUM_CORES)
    PD.optimize()
    #
    _route = {}
    for j in kN:
        for i in kN:
            if x_ij[i, j].x > 0.5:
                _route[i] = j
    i = _kP
    route = []
    while i != _kM:
        route.append(i)
        i = _route[i]
    route.append(i)
    #
    return PD.objVal, route


if __name__ == '__main__':
    pass
