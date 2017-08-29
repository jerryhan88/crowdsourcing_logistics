from init_project import *
#
from gurobipy import *
from math import ceil

SUB_SUB_LOGGING = False


def masterProblem(problem):
    bB, T, iP, iM, \
    r_i, v_i, K, kP, kM, w_k, _lambda, _delta, \
    t_ij = problem
    input4subProblem = [T, r_i, v_i, K, w_k, _lambda, _delta]
    input4subSubProblem = [iP, iM, kP, kM, t_ij]
    #
    # generate initial bundles
    #

    print(bB, _lambda)
    print(T)

    e_bi = [[0 for _ in range(len(T))] for _ in range(bB)]
    taskBN = ceil(len(T) / bB)
    for i in T:
        e_bi[i % taskBN][i] = 1
    B = range(len(e_bi))
    p_b = []
    for b in B:
        bundle = [i for i, v in enumerate(e_bi[b]) if v == 1]
        reward = sum([r_i[i] for i in bundle])
        # subProblem(pi_i, mu, input4subProblem, input4subSubProblem)
        profit = 0
        for k in K:
            detour = subSubProblem(bundle, k, input4subSubProblem)
            if detour < _delta:
                profit += w_k[k] * reward
        p_b.append(profit)
    #

    m = Model('materProblem')
    ofpath = opath.join(dpath['gurobiLog'], 'masterProblem.log')
    m.setParam('LogFile', ofpath)
    #
    # Define decision variables
    #
    q_b = {}
    for b in B:
        q_b[b] = m.addVar(vtype="I", name="q[%d]" % b)
    m.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for b in B:
        obj += p_b[b] * q_b[b]
    m.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    #
    taskAC = {}
    for i in T:  # eq:taskA
        taskAC[i] = m.addConstr(quicksum(e_bi[b][i] * q_b[b] for b in B) == 1, name="taskAC[%d]" % i)
    numBC = m.addConstr(quicksum(q_b[b] for b in B) == bB, name="numBC")
    #
    m.update()  # must update before calling relax()

    relax = m.relax()

    relax.write("temp.lp")

    relax.optimize()

    pi = [c.Pi for c in relax.getConstrs()]  # keep dual variables
    print(pi)

    # relax = m.relax()
    # relax.optimize()
    # # for c in relax.getConstrs():
    # #     print(c.ConstrName, dir(c))
    # #     c0 = model.getConstrByName("c0")
    # #     assert False
    pi_i = [relax.getConstrByName("taskAC[%d]" % i).Pi for i in T]
    mu = relax.getConstrByName("numBC").Pi

    print([relax.getVarByName("q[%d]" % b).x for b in B])



    # print('col', subProblem(pi_i, mu, input4subProblem, input4subSubProblem))





def subProblem(pi_i, mu, input4subProblem, input4subSubProblem):
    T, r_i, v_i, K, w_k, _lambda, _delta = input4subProblem
    big_M = sum(r_i)
    #
    m = Model('subProblem')
    ofpath = opath.join(dpath['gurobiLog'], 'subProblem.log')
    m.setParam('LogFile', ofpath)
    #
    # Define decision variables
    #
    y_k, z_i, a_k = {}, {}, {}
    for k in K:
        y_k[k] = m.addVar(vtype=GRB.BINARY, name='y[%d]' % k)
    for i in T:
        z_i[i] = m.addVar(vtype=GRB.BINARY, name='z[%d]' % i)
    for k in K:
        a_k[k] = m.addVar(vtype=GRB.CONTINUOUS, name='a[%d]' % k)
    m.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for k in K:
        obj += w_k[k] * a_k[k]
    for i in T:
        obj -= pi_i[i] * z_i[i]
    obj -= mu
    m.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constrains
    #
    for k in K:
        #
        # eq:linAlpha
        #
        m.addConstr(a_k[k] >= quicksum(r_i[i] * z_i[i] for i in T) - big_M * (1 - y_k[k]), name='la1[%d]' % k)
        m.addConstr(a_k[k] <= big_M * y_k[k], name='la2[%d]' % k)
        m.addConstr(a_k[k] <= quicksum(r_i[i] * z_i[i] for i in T), name='la3[%d]' % k)
    #  # eq:volTh
    m.addConstr(quicksum(v_i[i] * z_i[i] for i in T) <= _lambda)
    #
    # For callback function
    #
    m._K, m._T = K, T
    m._y_k, m._z_i = y_k, z_i
    m._delta, m._big_M = _delta, big_M
    m._input4subSubProblem = input4subSubProblem
    #
    m.params.LazyConstraints = 1
    m.optimize(addDetourC)
    return [i for i in T if z_i[i].x > 0.05]


def addDetourC(m, where):
    if where == GRB.callback.MIPSOL:
        b = [i for i in m._T if m.cbGetSolution(m._z_i[i]) > 0.5]
        if b:
            for k in m._K:
                d_bk = subSubProblem(b, k, m._input4subSubProblem)
                #  # eq:detourTh
                m.cbLazy(d_bk - m._delta <= m._big_M * (1 - m._y_k[k]))


def subSubProblem(b, k, input4subSubProblem):
    iP, iM, kP, kM, t_ij = input4subSubProblem
    _kP, _kM = 'ori%d' % k, 'dest%d' % k
    N = {_kP: kP[k],
         _kM: kM[k]}
    P, D = {}, {}
    for i in b:
        P['p%d' % i] = iP[i]
        D['d%d' % i] = iM[i]
        #
        N['p%d' % i] = iP[i]
        N['d%d' % i] = iM[i]
    _t_ij = {}
    for i in N:
        for j in N:
            _t_ij[i, j] = t_ij[N[i], N[j]]
    #
    m = Model('subSubProblem')
    m.Params.OutputFlag = SUB_SUB_LOGGING
    ofpath = opath.join(dpath['gurobiLog'], 'subSubProblem.log')
    m.setParam('LogFile', ofpath)
    #
    # Define decision variables
    #
    x_ij, o_i = {}, {}
    for i in N:
        for j in N:
            x_ij[i, j] = m.addVar(vtype=GRB.BINARY, name='x[%s,%s]' % (i, j))
    for i in N:
        o_i[i] = m.addVar(vtype=GRB.INTEGER, name='o[%s]' % (i))
    m.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for i in N:
        for j in N:
            obj += _t_ij[i, j] * x_ij[i, j]
    m.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constrains
    #
    # General TSP
    #
    for i in N:  # eq:outFlow
        if i == _kM: continue
        m.addConstr(quicksum(x_ij[i, j] for j in N if j != _kP) == 1, name='oF_%s' % i)
    for j in N:  # eq:inFlow
        if j == _kP: continue
        m.addConstr(quicksum(x_ij[i, j] for i in N if i != _kM) == 1, name='iF_%s' % i)
    for i in N:  # eq:XselfFlow
        m.addConstr(x_ij[i, i] == 0, name='XsF_%s' % i)
    #
    # Path based pickup and delivery
    #
    #
    #  # eq:pathBasedOutIn
    m.addConstr(quicksum(x_ij[_kP, j] for j in P) == 1, name='pbO')
    m.addConstr(quicksum(x_ij[i, _kM] for i in D) == 1, name='pbI')
    #  # eq:XoriDdestP
    m.addConstr(quicksum(x_ij[_kP, j] for j in D) == 0, name='XoriD')
    m.addConstr(quicksum(x_ij[i, _kM] for i in P) == 0, name='XdestP')
    #  # eq:reversedPD
    for i in b:
        _iP, _iM = 'p%d' % i, 'd%d' % i
        m.addConstr(x_ij[_iM, _iP] == 0, name='rPD_%d' % i)
    for i in N:
        for j in N:  # eq:direction
            m.addConstr(x_ij[i, j] + x_ij[j, i] <= 1, name='dir[%s,%s]' % (i, j))
    #  # eq:initOrder
    m.addConstr(o_i[_kP] == 1, name='ordOri')
    m.addConstr(o_i[_kM] == len(N), name='ordDest')
    for i in b:  # eq:pdSequnce
        _iP, _iM = 'p%d' % i, 'd%d' % i
        m.addConstr(o_i[_iP] <= o_i[_iM], name='ord_%s' % i)
    for i in N:
        for j in N:  # eq:ordering
            if i == _kM or j == _kP:
                continue
            m.addConstr(o_i[i] + 1 <= o_i[j] + len(N) * (1 - x_ij[i, j]), name='ord[%s,%s]' % (i, j))
    #
    # For callback function
    #
    m._N = N
    m._x_ij = x_ij
    #
    m.params.LazyConstraints = 1
    m.optimize(addSubTourElimC)
    #
    if m.status != GRB.Status.OPTIMAL:
        m.computeIIS()
        m.write('subSubProblem.ilp')
    return m.objVal


def addSubTourElimC(m, where):
    if where == GRB.callback.MIPSOL:
        curRoute = []
        for i in m._N:
            for j in m._N:
                if i == j:
                    continue
                if m.cbGetSolution(m._x_ij[i, j]) > 0.5:
                    curRoute.append((i, j))
        subtours = get_subtours(curRoute, m._N)
        for sb in subtours:
            if not sb:
                continue
            expr = 0
            visited_nodes = set()
            for i, j in sb:
                expr += m._x_ij[i, j]
                expr += m._x_ij[j, i]
                visited_nodes.add(i)
                visited_nodes.add(j)
            if len(visited_nodes) == len(m._N):
                continue
            # eq:subTourElim
            m.cbLazy(expr <= len(visited_nodes) - 1)


def get_subtours(edges, nodes):
    subtours = []
    visited, adj = {}, {}
    for i in nodes:
        visited[i] = False
        adj[i] = []
    for i, j in edges:
        adj[i].append(j)
    while True:
        for i in visited.keys():
            if not visited[i]:
                break
        thistour = []
        while True:
            visited[i] = True
            neighbors = [j for j in adj[i] if not visited[j]]
            if len(neighbors) == 0:
                break
            thistour.append((i, neighbors[0]))
            i = neighbors[0]
        subtours.append(thistour)
        if all(visited.values()):
            break
    return subtours


def convert_input4MathematicalModel(travel_time, \
                                    flows, paths, \
                                    tasks, rewards, volumes, \
                                    num_bundles, volume_th, detour_th):
    #
    # For master problem
    #
    bB = num_bundles
    T = list(range(len(tasks)))
    iP, iM = list(zip(*[tasks[i] for i in T]))
    #
    # For sub problem
    #
    r_i, v_i = rewards, volumes
    K = list(range(len(paths)))
    kP, kM = list(zip(*[paths[k] for k in K]))
    sum_f_k = sum(flows[i][j] for i in range(len(flows)) for j in range(len(flows)))
    w_k = [flows[i][j] / float(sum_f_k) for i, j in paths]
    _lambda = volume_th
    _delta = detour_th

    #
    # For subSub problem
    #
    t_ij = travel_time

    return bB, T, iP, iM, \
           r_i, v_i, K, kP, kM, w_k, _lambda, _delta, \
           t_ij


if __name__ == '__main__':
    from problems import *

    print(masterProblem(convert_input4MathematicalModel(*ex2())))
