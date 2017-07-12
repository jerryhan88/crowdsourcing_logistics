from init_project import *
#
from gurobipy import *

big_M = 100
#
VALIDATION = False
NO_LOG = False

def run_mip(problem):
    _o, B, _lambda, _delta, \
    _n, T, P, D, N, r_i, v_i, \
    _m, K, Omega, Pi, omega_k, pi_k, w_k, \
    V, t_ij = problem
    #
    m = Model('')
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
        m.addConstr(quicksum(z_bi[b, i] for b in B) == 1)
    #
    for b in B:
        #
        # eq:bundleReward
        #
        m.addConstr(quicksum(z_bi[b, i] * r_i[i] for i in P) == R_b[b])
    #
    for b in B:
        #
        # eq:volThreshold
        #
        m.addConstr(quicksum(z_bi[b, i] * v_i[i] for i in P) <= _lambda)
    #
    for b in B:
        for k in K:
            #
            # eq:flowBegin
            #
            m.addConstr(quicksum(x_bkij[b, k, omega_k[k], j] for j in P + [pi_k[k]]) == 1)
    #
    for b in B:
        for k in K:
            #
            # eq:flowEnd
            #
            m.addConstr(quicksum(x_bkij[b, k, i, pi_k[k]] for i in D + [omega_k[k]]) == 1)
    #
    for b in B:
        for k in K:
            for i in P:
                #
                # eq:taskMembership_flowEnforcement
                #
                m.addConstr(quicksum(x_bkij[b, k, i, j] for j in N) == z_bi[b, i])
    #
    for b in B:
        for k in K:
            for i in P:
                #
                # eq:pickupDeliveryConservation
                #
                m.addConstr(quicksum(x_bkij[b, k, i, j] for j in N) == quicksum(x_bkij[b, k, j, i + _n] for j in N))
    #
    for b in B:
        for k in K:
            for j in P:
                #
                # eq:flowConservation
                #
                m.addConstr(
                    quicksum(x_bkij[b, k, i, j] for i in [omega_k[k]] + N) == quicksum(x_bkij[b, k, j, i] for i in N + [pi_k[k]]))
    #
    for b in B:
        for k in K:
            for i in N:
                #
                # eq:zeroFlow
                #
                m.addConstr(x_bkij[b, k, i, i] == 0)
    #
    for b in B:
        for k in K:
            for i in P:
                #
                # eq:reverseFlow
                #
                m.addConstr(x_bkij[b, k, i + _n, i] == 0)

    #
    for b in B:
        for k in K:
            #
            # eq:initBeginningDistance
            #
            m.addConstr(d_bkj[b, k, omega_k[k]] == 0)
    #
    for b in B:
        for k in K:
            #
            # eq:calcAccumulatedDistance^prime
            #
            for j in N + [pi_k[k]]:
                m.addConstr(d_bkj[b, k, j] == quicksum(beta_bkij[b, k, i, j] for i in [omega_k[k]] + N))
    #
    for b in B:
        for k in K:
            #
            # eq:accumulatedDistanceFeasibility
            #
            m.addConstr((d_bkj[b, k, pi_k[k]] - t_ij[omega_k[k], pi_k[k]] - _delta) <= big_M * (1 - y_bk[b, k]))
    #
    for b in B:
        for k in K:
            #
            # eq:linearization_alpha
            #
            m.addConstr(alpha_bk[b, k] >= R_b[b] - big_M * (1 - y_bk[b, k]))
            m.addConstr(alpha_bk[b, k] <= big_M * y_bk[b, k])
            m.addConstr(alpha_bk[b, k] <= R_b[b])
    #
    for b in B:
        for k in K:
            for i in [omega_k[k]] + N:
                for j in N + [pi_k[k]]:
                    #
                    # eq:linearization_beta
                    #
                    m.addConstr(beta_bkij[b, k, i, j] >= d_bkj[b, k, i] + t_ij[i, j] - big_M * (1 - x_bkij[b, k, i, j]))
                    m.addConstr(beta_bkij[b, k, i, j] <= big_M * x_bkij[b, k, i, j])
                    m.addConstr(beta_bkij[b, k, i, j] <= d_bkj[b, k, i] + t_ij[i, j])



    #
    if VALIDATION:
        m.write('model.mps')
        m.write('model.rew')
        m.write('model.lp')
        m.computeIIS()
        m.write('model.ilp')

    m.setParam(GRB.Param.DualReductions, 0)
    m.optimize()

    assert m.status == GRB.Status.OPTIMAL, 'Errors while optimization'

    # print m.objVal
    # print '-------------------------------z_bi'
    # for b in B:
    #     for i in P:
    #         print b,i, z_bi[b, i].x
    #
    # print '-------------------------------y_bk'
    # for b in B:
    #     for k in K:
    #         print b, k, y_bk[b, k].x
    # print '-------------------------------R_b'
    # for b in B:
    #     print b, R_b[b].x
    # print '-------------------------------obj'
    # for b in B:
    #     sum_w_k = 0
    #     for k in K:
    #         print '\t', b, k, w_k[k], y_bk[b, k].x
    #         sum_w_k += w_k[k] * y_bk[b, k].x
    #     print b, sum_w_k
    # print '-------------------------------d_bkj'
    # for b in B:
    #     for k in K:
    #         # for j in [omega_k[k]] + N + [pi_k[k]]:
    #         print '(%d,%d,%d):' % (b, k, pi_k[k]),
    #         print '\t', (d_bkj[b, k, pi_k[k]].x, t_ij[omega_k[k], pi_k[k]]), (d_bkj[b, k, pi_k[k]].x - t_ij[omega_k[k], pi_k[k]] - _lambda)
    #
    # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # k = 1
    # print d_bkj[0, k, omega_k[k]].x, d_bkj[0, k, 0].x, d_bkj[0, k, 1].x, d_bkj[0, k, pi_k[k]].x
    # print x_bkij[0, k, omega_k[k], 0].x, x_bkij[0, k, 0, 1].x, x_bkij[0, k, 1, pi_k[k]].x
    #
    # print x_bkij[0, k, omega_k[k], 1].x, x_bkij[0, k, 0, 1].x, x_bkij[0, k, 1, 1].x
    #
    # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # for k in K:
    #     print k, [x_bkij[0, k, omega_k[k], j].x for j in N + [pi_k[k]]]






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




if __name__ == '__main__':
    from problems import *
    # run_mip(ex1)
    run_mip(convert_input4MathematicalModel(*ex1()))
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
