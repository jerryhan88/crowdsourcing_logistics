from init_project import *
#
from gurobipy import *
from time import time

big_M = 100
#
VALIDATION = False
NO_LOG = True

TimeLimit = 60 * 60
numThreads = 1

def convert_input4MathematicalModel(points, travel_time, \
                                    flows, paths, \
                                    tasks, rewards, volumes, \
                                    num_bundles, volume_th, detour_th):
    #
    # Convert inputs to mathematical model's notation
    #
    _o = num_bundles
    B = range(_o)
    _lambda = volume_th
    _delta = detour_th
    #
    _n = len(tasks)
    r_i, v_i = rewards, volumes
    T = list(range(_n))
    P = list(range(_n))
    D = list(range(_n, 2 * _n))
    N = P + D
    #
    _m = len(paths)
    K = range(_m)
    Omega = list(range(2 * _n, 2 * _n + _m))
    Pi = list(range(2 * _n + _m, 2 * _n + 2 * _m))
    omega_k = [Omega[k] for k in K]
    pi_k = [Pi[k] for k in K]
    sum_f_k = sum(flows[i][j] for i in xrange(len(flows)) for j in xrange(len(flows)))
    w_k = [flows[i][j] / float(sum_f_k) for i, j in paths]
    V = N + Omega + Pi
    t_ij = {}
    for i in V:
        p0 = None
        if i in set(P):
            p0 = tasks[i][0]
        elif i in set(D):
            p0 = tasks[i - _n][1]
        elif i in set(Omega):
            p0 = paths[i - 2 * _n][0]
        elif i in set(Pi):
            p0 = paths[i - (2 * _n + _m)][1]
        assert p0 != None
        for j in V:
            p1 = None
            if j in set(P):
                p1 = tasks[j][0]
            elif j in set(D):
                p1 = tasks[j - _n][1]
            elif j in set(Omega):
                p1 = paths[j - 2 * _n][0]
            elif j in set(Pi):
                p1 = paths[j - (2 * _n + _m)][1]
            assert p1 != None
            t_ij[i, j] = travel_time[p0, p1]
    return _o, B, _lambda, _delta, \
            _n, T, P, D, N, r_i, v_i, \
            _m, K, Omega, Pi, omega_k, pi_k, w_k, \
            V, t_ij



def run_mip_eliSubTour(problem):
    _o, B, _lambda, _delta, \
    _n, T, P, D, N, r_i, v_i, \
    _m, K, Omega, Pi, omega_k, pi_k, w_k, \
    V, t_ij = problem
    #
    startTime = time()
    m = Model('')
    ofpath = opath.join(dpath['gurobiLog'], 'nt%d-np%d-nb%d-tv%d-td%d.log' % (_n, _m, _o, _lambda, _delta))
    m.setParam('LogFile', ofpath)
    if NO_LOG:
        m.setParam('OutputFlag', False)
    #
    # Define decision variables
    #
    x_bkij = {}
    for b in B:
        for k in K:
            for i in [omega_k[k]] + N:
                for j in N + [pi_k[k]]:
                    x_bkij[b, k, i, j] = m.addVar(vtype=GRB.BINARY, name='x_(%d,%d,%d,%d)' % (b, k, i, j))
    #
    y_bk = {}
    for b in B:
        for k in K:
            y_bk[b, k] = m.addVar(vtype=GRB.BINARY, name='y_(%d,%d)' % (b, k))
    #
    z_bi = {}
    for b in B:
        for i in P:
            z_bi[b, i] = m.addVar(vtype=GRB.BINARY, name='z_(%d,%d)' % (b, i))
    #
    R_b = {}
    for b in B:
        R_b[b] = m.addVar(vtype=GRB.CONTINUOUS, name='R_(%d)' % b)
    #
    d_bkj = {}
    for b in B:
        for k in K:
            for j in [omega_k[k]] + N + [pi_k[k]]:
                d_bkj[b, k, j] = m.addVar(vtype=GRB.CONTINUOUS, name='d_(%d,%d,%d)' % (b, k, j))
    #
    #   Decision variables for linearization
    #
    alpha_bk = {}
    for b in B:
        for k in K:
            alpha_bk[b, k] = m.addVar(vtype=GRB.CONTINUOUS, name='alpha_(%d,%d)' % (b, k))
    #
    beta_bkij = {}
    for b in B:
        for k in K:
            for i in [omega_k[k]] + N:
                for j in N + [pi_k[k]]:
                    beta_bkij[b, k, i, j] = m.addVar(vtype=GRB.CONTINUOUS, name='beta_(%d,%d,%d,%d)' % (b, k, i, j))
    m.update()
    #
    # Define objective
    #
    #   eq:maxExpectedReward^prime
    obj = LinExpr()
    for b in B:
        for k in K:
            obj += w_k[k] * alpha_bk[b, k]
    m.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    #
    for i in T:
        #
        # eq:taskMembership
        #
        m.addConstr(quicksum(z_bi[b, i] for b in B) == 1, name='tm%d' % i)
    #
    for b in B:
        #
        # eq:bundleReward
        #
        m.addConstr(quicksum(z_bi[b, i] * r_i[i] for i in P) == R_b[b], name='br%d' % b)
    #
    for b in B:
        #
        # eq:volThreshold
        #
        m.addConstr(quicksum(z_bi[b, i] * v_i[i] for i in P) <= _lambda, name='vt%d' % b)
    #
    for b in B:
        for k in K:
            #
            # eq:flowBegin
            #
            m.addConstr(quicksum(x_bkij[b, k, omega_k[k], j] for j in P + [pi_k[k]]) == 1, name='fb(%d,%d)' % (b, k))
    #
    for b in B:
        for k in K:
            #
            # eq:flowEnd
            #
            m.addConstr(quicksum(x_bkij[b, k, i, pi_k[k]] for i in D + [omega_k[k]]) == 1, name='fe(%d,%d)' % (b, k))

    for b in B:
        for k in K:
            #
            # eq:flowBeginX
            #
            m.addConstr(quicksum(x_bkij[b, k, omega_k[k], j] for j in D) == 0, name='fbx(%d,%d)' % (b, k))
    for b in B:
        for k in K:
            #
            # eq:flowEndX
            #
            m.addConstr(quicksum(x_bkij[b, k, i, pi_k[k]] for i in P) == 0, name='fex(%d,%d)' % (b, k))
    #
    for b in B:
        for k in K:
            for i in P:
                #
                # eq:taskMembership_flowEnforcement
                #
                m.addConstr(quicksum(x_bkij[b, k, i, j] for j in N) == z_bi[b, i], name='tmfe(%d,%d,%d)' % (b, k, i))
                # m.addConstr(quicksum(x_bkij[b, k, j, i + _n] for j in N) == z_bi[b, i], name='tmfe(%d,%d,%d)' % (b, k, i))
    #
    for b in B:
        for k in K:
            for i in P:
                #
                # eq:pickupDeliveryConservation
                #
                m.addConstr(quicksum(x_bkij[b, k, i, j] for j in N) == quicksum(x_bkij[b, k, j, i + _n] for j in N),
                            name='pdc(%d,%d,%d)' % (b, k, i))
    #
    for b in B:
        for k in K:
            for j in N:
                #
                # eq:flowConservation
                #
                m.addConstr(
                    quicksum(x_bkij[b, k, i, j] for i in [omega_k[k]] + N) == quicksum(x_bkij[b, k, j, i] for i in N + [pi_k[k]]),
                    name='fc(%d,%d,%d)' % (b, k, j))
    #
    for b in B:
        for k in K:
            for i in N:
                #
                # eq:zeroFlow
                #
                m.addConstr(x_bkij[b, k, i, i] == 0, name='zf(%d,%d,%d)' % (b, k, i))
    #
    for b in B:
        for k in K:
            for i in P:
                #
                # eq:reverseFlow
                #
                m.addConstr(x_bkij[b, k, i + _n, i] == 0, name='rf(%d,%d,%d)' % (b, k, i))

    #
    for b in B:
        for k in K:
            #
            # eq:initBeginningDistance
            #
            m.addConstr(d_bkj[b, k, omega_k[k]] == 0, name='ibd(%d,%d)' % (b, k))
    #
    for b in B:
        for k in K:
            #
            # eq:calcAccumulatedDistance^prime
            #
            for j in N + [pi_k[k]]:
                m.addConstr(d_bkj[b, k, j] == quicksum(beta_bkij[b, k, i, j] for i in [omega_k[k]] + N),
                            name='cad(%d,%d,%d)' % (b, k, j))
    #
    for b in B:
        for k in K:
            #
            # eq:accumulatedDistanceFeasibility
            #
            m.addConstr((d_bkj[b, k, pi_k[k]] - t_ij[omega_k[k], pi_k[k]] - _delta) <= big_M * (1 - y_bk[b, k]),
                        name='adf(%d,%d)' % (b, k))
    #
    for b in B:
        for k in K:
            #
            # eq:linearization_alpha
            #
            m.addConstr(alpha_bk[b, k] >= R_b[b] - big_M * (1 - y_bk[b, k]), name='la1(%d,%d)' % (b, k))
            m.addConstr(alpha_bk[b, k] <= big_M * y_bk[b, k], name='la2(%d,%d)' % (b, k))
            m.addConstr(alpha_bk[b, k] <= R_b[b], name='la3(%d,%d)' % (b, k))
    #
    for b in B:
        for k in K:
            for i in [omega_k[k]] + N:
                for j in N + [pi_k[k]]:
                    #
                    # eq:linearization_beta
                    #
                    m.addConstr(beta_bkij[b, k, i, j] >= d_bkj[b, k, i] + t_ij[i, j] - big_M * (1 - x_bkij[b, k, i, j]),
                                name='lb1(%d,%d,%d,%d)' % (b, k, i, j))
                    m.addConstr(beta_bkij[b, k, i, j] <= big_M * x_bkij[b, k, i, j],
                                name='lb2(%d,%d,%d,%d)' % (b, k, i, j))
                    m.addConstr(beta_bkij[b, k, i, j] <= d_bkj[b, k, i] + t_ij[i, j],
                                name='lb3(%d,%d,%d,%d)' % (b, k, i, j))
    #
    m._B, m._K, m._P, m._omega_k, m._pi_k = B, K, P, omega_k, pi_k
    m._x_bkij = x_bkij
    m._z_bi = z_bi
    #
    m.setParam('TimeLimit', TimeLimit)
    m.setParam('Threads', numThreads)

    m.setParam(GRB.Param.DualReductions, 0)
    m.params.LazyConstraints = 1
    m.optimize(subtourelim)
    # m.optimize()
    if VALIDATION:
        m.write('model.lp')
        if m.status != GRB.Status.OPTIMAL:
            m.computeIIS()
            m.write('model.ilp')
        assert m.status == GRB.Status.OPTIMAL, 'Errors while optimization'
        print 'bundle-------------------'
        for b in B:
            print 'b%d' % b,
            for i in P:
                if z_bi[b, i].x > 0.5:
                    print i,
            print ''

        print 'route-------------------'

        for b in B:
            for k in K:
                print 'b%d k%d' % (b, k),
                route = []
                for i in [omega_k[k]] + N:
                    for j in N + [pi_k[k]]:
                        if x_bkij[b, k, i, j].x > 0.05:
                            route.append((i, j))
                print route
    assert m.status == GRB.Status.OPTIMAL, 'Errors while optimization'
    return m.objVal, time() - startTime


def subtourelim(m, where):
    if where == GRB.callback.MIPSOL:
        for b in m._B:
            pd_points = []
            x_pd_points = []
            for i in m._P:
                if m.cbGetSolution(m._z_bi[b, i]) > 0.5:
                    pd_points.append(i)
                    pd_points.append(i + len(m._P))
                else:
                    x_pd_points.append(i)
                    x_pd_points.append(i + len(m._P))
            if not pd_points:
                continue
            for k in m._K:
                m.cbLazy(m._x_bkij[b, k, m._omega_k[k], m._pi_k[k]] == 0)
                for i in x_pd_points:
                    m.cbLazy(m._x_bkij[b, k, m._omega_k[k], i] == 0)
                    m.cbLazy(m._x_bkij[b, k, i, m._pi_k[k]] == 0)
                    for j in pd_points:
                        m.cbLazy(m._x_bkij[b, k, i, j] == 0)
                        m.cbLazy(m._x_bkij[b, k, j, i] == 0)
                curSol = []
                for i in pd_points:
                    for j in pd_points:
                        if i == j:
                            continue
                        if m.cbGetSolution(m._x_bkij[b, k, i, j]) > 0.5:
                            curSol.append((i, j))
                subtours = get_subtours(curSol, pd_points)
                for sb in subtours:
                    if not sb:
                        continue
                    expr = 0
                    visited_nodes = set()
                    for i, j in sb:
                        expr += m._x_bkij[b, k, i, j]
                        expr += m._x_bkij[b, k, j, i]
                        visited_nodes.add(i)
                        visited_nodes.add(j)
                    if len(visited_nodes) == len(pd_points):
                        continue
                    m.cbLazy(expr <= len(visited_nodes) - 1)


def get_subtours(edges, oridest_pd_points):
    subtours = []
    visited, adj = {}, {}
    for i in oridest_pd_points:
        visited[i] = False
        adj[i] = []
    for i, j in edges:
        adj[i].append(j)
    while True:
        for i in visited.iterkeys():
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
    # subtours.sort(key=lambda l: len(l))
    return subtours


if __name__ == '__main__':
    from problems import *
    # run_mip(ex1)
    # run_mip(convert_input4MathematicalModel(*ex2()))
    print run_mip_eliSubTour(convert_input4MathematicalModel(*ex2()))
    # run_mip(convert_input4MathematicalModel(*ex1()))
    # run_mip(convert_input4MathematicalModel(*ex2()))
    # points, travel_time, \
    # flows, paths, \
    # tasks, rewards, volumes, \
    # numBundles, thVolume, thDetour = random_problem(2, 3, 3, 3, 1, 3, 1, 2, 4, 3.3, 1.5)
    # #
    # run_mip(convert_input4MathematicalModel(points, travel_time, \
    #             flows, paths, \
    #             tasks, rewards, volumes, \
    #             numBundles, thVolume, thDetour))
