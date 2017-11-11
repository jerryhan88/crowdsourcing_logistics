from gurobipy import *
#
from _utils.mm_utils import get_routeFromOri
from _utils.recording import *


def run(b, k, t_ij, log_fpath=None, numThreads=None, TimeLimit=None):
    def addSubTourElimC_minPD(m, where):
        if where == GRB.callback.MIPSOL:
            curRoute = []
            for i in m._N:
                for j in m._N:
                    if i == j:
                        continue
                    if m.cbGetSolution(m._x_ij[i, j]) > 0.5:
                        curRoute.append((i, j))
            route = get_routeFromOri(curRoute, m._N)
            expr = 0
            if len(route) != len(m._N) - 1:
                for i, j in route:
                    expr += m._x_ij[i, j]
                m.cbLazy(expr <= len(route) - 1)  # eq:subTourElim
    #
    _kP, _kM = 'ori%d' % k, 'dest%d' % k
    N = {_kP, _kM}
    P, D = set(), set()
    for i in b:
        P.add('p%d' % i); D.add('d%d' % i)
    N = N.union(P)
    N = N.union(D)
    #
    # Define decision variables
    #
    m = Model('minTimePD')
    x_ij, o_i = {}, {}
    for i in N:
        o_i[i] = m.addVar(vtype=GRB.INTEGER, name='o[%s]' % i)
        for j in N:
            x_ij[i, j] = m.addVar(vtype=GRB.BINARY, name='x[%s,%s]' % (i, j))
    m.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for i in N:
        for j in N:
            obj += t_ij[i, j] * x_ij[i, j]
    obj -= t_ij[_kM, _kP]
    obj -= t_ij[_kP, _kM]
    m.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constrains
    #
    # Flow based routing
    #  eq:pathPD
    m.addConstr(x_ij[_kM, _kP] == 1,
                name='cf')
    m.addConstr(quicksum(x_ij[_kP, j] for j in P) == 1,
                name='pPDo')
    m.addConstr(quicksum(x_ij[i, _kM] for i in D) == 1,
                name='pPDi')
    #  # eq:XpathPD
    m.addConstr(quicksum(x_ij[_kP, j] for j in D) == 0,
                name='XpPDo')
    m.addConstr(quicksum(x_ij[i, _kM] for i in P) == 0,
                name='XpPDi')
    #
    for i in N:  # eq:outFlow
        if i == _kM: continue
        m.addConstr(quicksum(x_ij[i, j] for j in N if j != _kP) == 1,
                    name='tOF[%s]' % i)
    for j in N:  # eq:inFlow
        if j == _kP: continue
        m.addConstr(quicksum(x_ij[i, j] for i in N if i != _kM) == 1,
                    name='tIF[%s]' % j)
    for i in N:  # eq:XselfFlow
        m.addConstr(x_ij[i, i] == 0,
                    name='XsF[%s]' % i)
    for i in N:
        for j in N:  # eq:direction
            m.addConstr(x_ij[i, j] + x_ij[j, i] <= 1,
                        name='dir[%s,%s]' % (i, j))
    #  # eq:initOrder
    m.addConstr(o_i[_kP] == 1,
                name='ordOri')
    m.addConstr(o_i[_kM] == len(N),
                name='ordDest')
    for i in b:  # eq:pdSequnce
        m.addConstr(o_i['p%d' % i] <= o_i['d%d' % i], name='ord[%s]' % i)
    for i in N:
        for j in N:  # eq:ordering
            if i == _kM or j == _kP:
                continue
            m.addConstr(o_i[i] + 1 <= o_i[j] + len(N) * (1 - x_ij[i, j]),
                        name='ord[%s,%s]' % (i, j))
    #
    # For callback function
    #
    m._N = N
    m._x_ij = x_ij
    m.params.LazyConstraints = 1
    #
    # Settings
    #
    if TimeLimit is not None:
        m.setParam('TimeLimit', TimeLimit)
    if numThreads is not None:
        m.setParam('Threads', numThreads)
    if log_fpath is not None:
        m.setParam('LogFile', log_fpath)
    else:
        m.Params.OutputFlag = X_GL
    #
    # Run Gurobi (Optimization)
    #
    m.optimize(addSubTourElimC_minPD)
    #
    _route = {}
    for j in N:
        for i in N:
            if x_ij[i, j].x > 0.5:
                _route[i] = j
    i = _kP
    route = []
    while i != _kM:
        route.append(i)
        i = _route[i]
    route.append(i)
    return m.objVal, route