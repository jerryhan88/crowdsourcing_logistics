from gurobipy import *
#
from _utils.mm_utils import *


def run(probSetting, grbSetting):
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
    b, k, t_ij = [probSetting.get(k) for k in ['b', 'k', 't_ij']]
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
    orM = Model('minTimePD')
    x_ij, o_i = {}, {}
    for i in N:
        o_i[i] = orM.addVar(vtype=GRB.INTEGER, name='o[%s]' % i)
        for j in N:
            x_ij[i, j] = orM.addVar(vtype=GRB.BINARY, name='x[%s,%s]' % (i, j))
    orM.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for i in N:
        for j in N:
            obj += t_ij[i, j] * x_ij[i, j]
    obj -= t_ij[_kM, _kP]
    obj -= t_ij[_kP, _kM]
    orM.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constrains
    #
    # Flow based routing
    #  eq:pathPD
    orM.addConstr(x_ij[_kM, _kP] == 1,
                name='cf')
    orM.addConstr(quicksum(x_ij[_kP, j] for j in P) == 1,
                name='pPDo')
    orM.addConstr(quicksum(x_ij[i, _kM] for i in D) == 1,
                name='pPDi')
    #  # eq:XpathPD
    orM.addConstr(quicksum(x_ij[_kP, j] for j in D) == 0,
                name='XpPDo')
    orM.addConstr(quicksum(x_ij[i, _kM] for i in P) == 0,
                name='XpPDi')
    #
    for i in N:  # eq:outFlow
        if i == _kM: continue
        orM.addConstr(quicksum(x_ij[i, j] for j in N if j != _kP) == 1,
                    name='tOF[%s]' % i)
    for j in N:  # eq:inFlow
        if j == _kP: continue
        orM.addConstr(quicksum(x_ij[i, j] for i in N if i != _kM) == 1,
                    name='tIF[%s]' % j)
    for i in N:  # eq:XselfFlow
        orM.addConstr(x_ij[i, i] == 0,
                    name='XsF[%s]' % i)
    for i in N:
        for j in N:  # eq:direction
            orM.addConstr(x_ij[i, j] + x_ij[j, i] <= 1,
                        name='dir[%s,%s]' % (i, j))
    #  # eq:initOrder
    orM.addConstr(o_i[_kP] == 1,
                name='ordOri')
    orM.addConstr(o_i[_kM] == len(N),
                name='ordDest')
    for i in b:  # eq:pdSequnce
        orM.addConstr(o_i['p%d' % i] <= o_i['d%d' % i], name='ord[%s]' % i)
    for i in N:
        for j in N:  # eq:ordering
            if i == _kM or j == _kP:
                continue
            orM.addConstr(o_i[i] + 1 <= o_i[j] + len(N) * (1 - x_ij[i, j]),
                        name='ord[%s,%s]' % (i, j))
    #
    # For callback function
    #
    orM._N = N
    orM._x_ij = x_ij
    orM.params.LazyConstraints = 1
    #
    # Run Gurobi (Optimization)
    #
    set_grbSettings(orM, grbSetting)
    orM.optimize(addSubTourElimC_minPD)
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
    return orM.objVal, route


def test():
    from greedyHeuristic import run as gHeuristic_run
    from problems import convert_input4MathematicalModel
    import pickle
    ifpath = 'nt05-np12-nb2-tv3-td4.pkl'
    with open(ifpath, 'rb') as fp:
        inputs = pickle.load(fp)
    objV, B, _ = gHeuristic_run(inputs)

    print(objV, B)

    bB, \
    T, r_i, v_i, _lambda, P, D, N, \
    K, w_k, t_ij, _delta = convert_input4MathematicalModel(*inputs)

    logContents = ''
    grbSettingOP = {}
    for b in B:
        br = sum([r_i[i] for i in b])
        logContents += '%s (%d) \n' % (str(b), br)
        p = 0
        for k, w in enumerate(w_k):
            probSetting = {'b': b, 'k': k, 't_ij': t_ij}
            detourTime, route = run(probSetting, grbSettingOP)
            if detourTime <= _delta:
                p += w * br
                logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w, detourTime, 1, str(route))
            else:
                logContents += '\t k%d, w %.2f dt %.2f; %d;\t %s\n' % (k, w, detourTime, 0, str(route))
        logContents += '\t\t\t\t\t\t %.3f\n' % p

    print(logContents)


if __name__ == '__main__':
    test()
