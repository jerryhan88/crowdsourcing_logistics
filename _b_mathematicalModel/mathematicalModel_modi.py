import __init__
from init_project import *
#
from gurobipy import *

big_M = 100000


def mip(problem):
    _o, B, _lambda, _delta, \
    _n, T, P, D, N, r_i, v_i, \
    _m, K, Omega, Pi, omega_k, pi_k, f_k, w_k, \
    V, t_ij = problem()
    #
    m = Model('')
    # m.setParam('OutputFlag', False)
    #
    # Define decision variables
    #
    x_bkij = {}  # pruning!!
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
        for i in T:
            z_bi[b, i] = m.addVar(vtype=GRB.BINARY, name='z_(%d,%d)' % (b, i))
    #
    R_b = {}
    for b in B:
        R_b[b] = m.addVar(name='R_(%d)' % b)
    #
    d_bkj = {}
    for b in B:
        for k in K:
            for j in [omega_k[k]] + N + [pi_k[k]]:
                d_bkj[b, k, j] = m.addVar(vtype=GRB.BINARY, name='d_(%d,%d,%d)' % (b, k, j))
    #
    #   Decision variables for linearization
    #
    alpha_bk = {}
    for b in B:
        for k in K:
            alpha_bk[b, k] = m.addVar(name='alpha_(%d,%d)' % (b, k))
    #
    beta_bkij = {}
    for b in B:
        for k in K:
            for i in [omega_k[k]] + N:
                for j in N + [pi_k[k]]:
                    beta_bkij[b, k, i, j] = m.addVar(name='beta_(%d,%d,%d,%d)' % (b, k, i, j))
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
        m.addConstr(quicksum(z_bi[b, i] * v_i[i] for i in T) <= _lambda)
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
            for j in N:
                #
                # eq:flowConservation
                #
                m.addConstr(quicksum(x_bkij[b, k, i, j] for i in N) == quicksum(x_bkij[b, k, j, i] for i in N))
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
    m.setParam(GRB.Param.DualReductions, 0)
    m.optimize()


if __name__ == '__main__':
    from problems import ex1, ex2
    # mip(ex1)
    mip(ex2)
