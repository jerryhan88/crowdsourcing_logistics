from gurobipy import *
#
from _util import set_grbSettings, get_routeFromOri


def run(probSetting, grbSetting, dict_pid=None):
    def callbackF(m, where):
        if where == GRB.callback.MIPSOL:
            curRoute = []
            for i in m._kN:
                for j in m._kN:
                    if i == j:
                        continue
                    if m.cbGetSolution(m._x_ij[i, j]) > 0.5:
                        curRoute.append((i, j))
            route = get_routeFromOri(curRoute, m._kN)
            if len(route) != len(m._kN) - 1:
                m.cbLazy(quicksum(m._x_ij[i, j] for i, j in route) <= len(route) - 1)  # eq:subTourElim
    #
    bc, k, t_ij = [probSetting.get(k) for k in ['bc', 'k', 't_ij']]
    _kP, _kM = 'ori%d' % k, 'dest%d' % k
    kN = {_kP, _kM}
    P, D = set(), set()
    for i in bc:
        P.add('p%d' % i); D.add('d%d' % i)
    kN = kN.union(P)
    kN = kN.union(D)
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
    obj -= t_ij[_kM, _kP]
    obj -= t_ij[_kP, _kM]
    PD.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constrains
    #
    # Flow based routing
    #  eq:pathPD
    PD.addConstr(x_ij[_kM, _kP] == 1,
                name='cf')
    PD.addConstr(quicksum(x_ij[_kP, j] for j in P) == 1,
                name='pPDo')
    PD.addConstr(quicksum(x_ij[i, _kM] for i in D) == 1,
                name='pPDi')
    #  # eq:XpathPD
    PD.addConstr(quicksum(x_ij[_kP, j] for j in D) == 0,
                name='XpPDo')
    PD.addConstr(quicksum(x_ij[i, _kM] for i in P) == 0,
                name='XpPDi')
    #
    for i in kN:  # eq:outFlow
        if i == _kM: continue
        PD.addConstr(quicksum(x_ij[i, j] for j in kN if j != _kP) == 1,
                    name='tOF[%s]' % i)
    for j in kN:  # eq:inFlow
        if j == _kP: continue
        PD.addConstr(quicksum(x_ij[i, j] for i in kN if i != _kM) == 1,
                    name='tIF[%s]' % j)
    for i in kN:  # eq:XselfFlow
        PD.addConstr(x_ij[i, i] == 0,
                    name='XsF[%s]' % i)
    for i in kN:
        for j in kN:  # eq:direction
            PD.addConstr(x_ij[i, j] + x_ij[j, i] <= 1,
                        name='dir[%s,%s]' % (i, j))
    #  # eq:initOrder
    PD.addConstr(o_i[_kP] == 1,
                name='ordOri')
    PD.addConstr(o_i[_kM] == len(kN),
                name='ordDest')
    for i in bc:  # eq:pdSequnce
        PD.addConstr(o_i['p%d' % i] <= o_i['d%d' % i], name='ord[%s]' % i)
    for i in kN:
        for j in kN:  # eq:ordering
            if i == _kM or j == _kP:
                continue
            PD.addConstr(o_i[i] + 1 <= o_i[j] + len(kN) * (1 - x_ij[i, j]),
                        name='ord[%s,%s]' % (i, j))
    #
    # For callback function
    #
    PD._kN = kN
    PD._x_ij = x_ij
    #
    # Run Gurobi (Optimization)
    #
    PD.setParam('LogToConsole', False)
    PD.setParam('OutputFlag', False)
    PD.params.LazyConstraints = 1
    for k, v in grbSetting.items():
        if k.startswith('Log'):
            continue
        PD.setParam(k, v)
    PD.optimize(callbackF)
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
    if dict_pid is not None:
        dict_pid[0][dict_pid[1]] = PD.objVal
    else:
        return PD.objVal, route
