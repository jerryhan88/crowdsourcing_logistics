from init_project import *
#
from gurobipy import *
import numpy as np


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
    _e_bi = [[0 for _ in range(len(T))] for _ in range(bB)]
    for i in T:
        _e_bi[i % bB][i] = 1

    # _e_bi = [[0 for _ in range(len(T))] for _ in range(len(T))]
    # for i in T:
    #     _e_bi[i][i] = 1
    colums = set()
    e_bi = []
    for l in _e_bi:
        if sum(l) == 0:
            continue
        e_bi.append(l)
        colums.add(tuple(l))
    B = range(len(e_bi))
    p_b = []
    for b in B:
        bundle = [i for i, v in enumerate(e_bi[b]) if v == 1]
        if bundle:
            p_b.append(calc_profit(K, w_k, _delta, r_i, input4subSubProblem, bundle))
        else:
            p_b.append(0)
    #
    m = Model('materProblem')
    ofpath = opath.join(dpath['gurobiLog'], 'masterProblem.log')
    m.setParam('LogFile', ofpath)
    #
    # Define decision variables
    #
    q_b = {}
    for b in B:
        q_b[b] = m.addVar(vtype=GRB.BINARY, name="q[%d]" % b)
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
    numBC = m.addConstr(quicksum(q_b[b] for b in B) <= bB, name="numBC")
    #
    m.update()  # must update before calling relax()
    m.write('master0.lp')

    while True:
        fn = "relaxted%d.lp" % len(B)

        relax = m.relax()
        # relax.write(fn)
        relax.Params.OutputFlag = SUB_SUB_LOGGING
        relax.optimize()
        #
        pi_i = [relax.getConstrByName("taskAC[%d]" % i).Pi for i in T]
        mu = relax.getConstrByName("numBC").Pi
        print(fn)
        print('\t sol', [relax.getVarByName("q[%d]" % i).x for i in B])
        print('\t p_b', [round(v, 4) for v in p_b])
        print('\t Pi', [round(v, 4) for v in pi_i])
        print('\t mu', mu)
        print('\t reduced costs')
        for i, l in enumerate(e_bi):
            print('\t\t', i, p_b[i] - (np.array(l) * np.array(pi_i)).sum() - mu)
        c_b, bundle = subProblem(pi_i, mu, input4subProblem, input4subSubProblem)
        vec = [0 for _ in range(len(T))]
        for i in bundle:
            vec[i] = 1
        p = calc_profit(K, w_k, _delta, r_i, input4subSubProblem, bundle)
        print('\t newBundle', bundle)
        print('\t\t reduced cost',c_b)
        print('\t\t coef', p)
        print('\t\t pimu', (np.array(vec) * np.array(pi_i)).sum() + mu)
        print()
        # if c_b > 0:
        #     break
        # if c_b == 0:
        #     break
        if c_b <= 0:
            break

        # if tuple(vec) in colums:
        #     break
        e_bi.append(vec)
        colums.add(tuple(vec))

        p_b.append(p)
        #
        col = Column()
        for i in range(len(T)):
            if e_bi[len(B)][i] > 0:
                col.addTerms(e_bi[len(B)][i], taskAC[i])
        col.addTerms(1, numBC)

        q_b[len(B)] = m.addVar(obj=p_b[len(B)], vtype=GRB.BINARY, name="q[%d]" % len(B), column=col)
        m.update()  # must update before calling relax()
        #
        B = range(len(e_bi))

    m.optimize()

    m.write('temp2.lp')





    print(B)
    print([q_b[b].x for b in B])
    print([p_b[b] * q_b[b].x for b in B])
    print(m.objVal)


def calc_profit(K, w_k, _delta, r_i, input4subSubProblem, bundle):
    profit = 0
    if not bundle:
        return profit
    else:
        reward = sum([r_i[i] for i in bundle])
        for k in K:
            detour = subSubProblem(bundle, k, input4subSubProblem)
            if detour < _delta:
                profit += w_k[k] * reward
    return profit


def subProblem(pi_i, mu, input4subProblem, input4subSubProblem):
    T, r_i, v_i, K, w_k, _lambda, _delta = input4subProblem
    big_M = 1000
    #
    m = Model('subProblem')
    m.Params.OutputFlag = SUB_SUB_LOGGING
    ofpath = opath.join(dpath['gurobiLog'], 'subProblem.log')
    m.setParam('LogFile', ofpath)
    #
    # Define decision variables
    #
    z_i, y_k, a_k = {}, {}, {}
    for i in T:
        z_i[i] = m.addVar(vtype=GRB.BINARY, name='z[%d]' % i)
    for k in K:
        y_k[k] = m.addVar(vtype=GRB.BINARY, name='y[%d]' % k)
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
    m.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    #
    #  # eq:volTh
    m.addConstr(quicksum(v_i[i] * z_i[i] for i in T) <= _lambda, name='volTH')
    #
    # For callback function
    #
    m._K, m._T = K, T
    m._z_i, m._y_k, m._a_k = z_i, y_k, a_k
    m._r_i = r_i
    m._delta, m._big_M = _delta, big_M
    m._input4subSubProblem = input4subSubProblem
    #
    m.params.LazyConstraints = 1
    m.optimize(addDetourC)
    #
    m.write('subProblem.lp')
    print(m.status)
    # if m.status != GRB.Status.OPTIMAL:
    #     m.computeIIS()
    #     m.write('subProblem.ilp')
    #     m.write('subProblem.lp')

    return m.objVal, [i for i in T if z_i[i].x > 0.05]


def addDetourC(m, where):
    if where == GRB.callback.MIPSOL:
        b = [i for i in m._T if m.cbGetSolution(m._z_i[i]) > 0.5]
        if b:

            new_m = m.copy()
            for i in b:
                new_m.addConstr(new_m.getVarByName("z[%d]" % i) == 1, name='cb[%d]' % i)

            for k in m._K:
                d_bk = subSubProblem(b, k, m._input4subSubProblem)
                #  # eq:detourTh
                if d_bk < m._delta:
                    m.cbLazy(m._y_k[k] == 1)
                    new_m.addConstr(new_m.getVarByName("y[%d]" % k) == 1, name='cf[%d]' % k)
                else:
                    m.cbLazy(m._y_k[k] == 0)
                    new_m.addConstr(new_m.getVarByName("y[%d]" % k) == 0, name='cf[%d]' % k)
                #
                # eq:linAlpha
                #
                m.cbLazy(m._a_k[k] >= quicksum(m._r_i[i] * m._z_i[i] for i in m._T) - m._big_M * (1 - m._y_k[k]))
                m.cbLazy(m._a_k[k] <= m._big_M * m._y_k[k])
                m.cbLazy(m._a_k[k] <= quicksum(m._r_i[i] * m._z_i[i] for i in m._T))

                a_k = new_m.getVarByName("a[%d]" % k)
                y_k = new_m.getVarByName("y[%d]" % k)
                new_m.addConstr(a_k >= quicksum(m._r_i[i] * new_m.getVarByName("z[%d]" % i) for i in m._T) - m._big_M * (1 - y_k), name='la1[%d]' % k)
                new_m.addConstr(a_k <= m._big_M * y_k, name='la2[%d]' % k)
                new_m.addConstr(a_k <= quicksum(m._r_i[i] * new_m.getVarByName("z[%d]" % i) for i in m._T), name='la3[%d]' % k)

            new_m.write('subProblem__.lp')
            new_m.Params.OutputFlag = True
            new_m.params.LazyConstraints = 0
            new_m.optimize()
            z = [new_m.getVarByName('z[%d]' % i).x for i in m._T]
            print('z', z)
            y = [new_m.getVarByName('y[%d]' % k).x for k in m._K]
            print('y', y)
            a = [new_m.getVarByName('a[%d]' % k).x for k in m._K]
            print('a', a)
        else:
            m.cbLazy(1 <= quicksum(m._z_i[i] for i in m._T))


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
    obj -= _t_ij[_kP, _kM]
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
