import __init__
from init_project import *
#
from gurobipy import *

big_M = 100000


def mip(problem):
    _o, B, _lambda, _delta, \
    _n, P, D, N, r_i, v_i, \
    _m, K, Omega, Phi, omega_k, phi_k, f_k, w_k, \
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
            for i in V:
                for j in V:
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
        R_b[b] = m.addVar(name='R_(%d)' % b)
    #
    d_bkj = {}
    for b in B:
        for k in K:
            for j in V:
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
            for i in N + [omega_k[k]]:
                for j in N + [phi_k[k]]:
                    beta_bkij[b, k, i, j] = m.addVar(name='beta_(%d,%d,%d,%d)' % (b, k, i, j))
    m.update()
    #
    # Define objective
    #
    #   eq:1^prime
    obj = LinExpr()
    for b in B:
        for k in K:
            obj += w_k[k] * alpha_bk[b, k]
    m.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    #


    #   eq:2
    for b in B:
        m.addConstr(quicksum(z_bi[b, i] * r_i[i] for i in P) == R_b[b])
    #   eq:3
    for i in P:
        m.addConstr(quicksum(z_bi[b, i] for b in B) == 1)
    #   eq:4
    for b in B:
        m.addConstr(quicksum(z_bi[b, i] * v_i[i] for i in P) <= _lambda)
    #   eq:5
    for b in B:
        for k in K:
            for j in P:
                m.addConstr(quicksum(x_bkij[b, k, i, j] for i in V) == z_bi[b, j])
    #   eq:6
    for b in B:
        for k in K:
            for i in P:
                m.addConstr(quicksum(x_bkij[b, k, i, j] for j in V) - quicksum(x_bkij[b, k, j, i + _n] for j in V) == 0)
    #   eq:7
    for b in B:
        for k in K:
            for j in N:
                m.addConstr(quicksum(x_bkij[b, k, i, j] for i in V) - quicksum(x_bkij[b, k, j, i] for i in V) == 0)
    #   eq:8
    for b in B:
        for k in K:
            m.addConstr(quicksum(x_bkij[b, k, omega_k[k], j] for j in P + [phi_k[k]]) == 1)
    #   eq:9
    for b in B:
        for k in K:
            m.addConstr(quicksum(x_bkij[b, k, i, phi_k[k]] for i in D + [omega_k[k]]) == 1)
    #   eq:10
    for b in B:
        for k in K:
            _Omega = set(Omega).difference(set([omega_k[k]]))
            for i in _Omega:
                m.addConstr(quicksum(x_bkij[b, k, i, j] for j in P + [phi_k[k]]) == 0)
    #   eq:11
    for b in B:
        for k in K:
            _Phi = set(Phi).difference(set([phi_k[k]]))
            for j in _Phi:
                m.addConstr(quicksum(x_bkij[b, k, i, j] for i in D + [omega_k[k]]) == 0)
    #   eq:12
    for b in B:
        for k in K:
            m.addConstr(d_bkj[b, k, omega_k[k]] == 0)
    #   eq:13^prime
    for b in B:
        for k in K:
            for j in N + [phi_k[k]]:
                m.addConstr(d_bkj[b, k, j] == quicksum(beta_bkij[b, k, i, j] for i in N + [omega_k[k]]))
    #   eq:14
    for b in B:
        for k in K:
            m.addConstr((d_bkj[b, k, phi_k[k]] - t_ij[omega_k[k], phi_k[k]] - _delta) - big_M * (1 - y_bk[b, k]) <= 0)
    #   eq:15
    for b in B:
        for k in K:
            m.addConstr(alpha_bk[b, k] >= R_b[b] - big_M * (1 - y_bk[b, k]))
            m.addConstr(alpha_bk[b, k] <= big_M * y_bk[b, k])
            m.addConstr(alpha_bk[b, k] <= R_b[b])
    #   eq:16
    for b in B:
        for k in K:
            for i in N + [omega_k[k]]:
                for j in N + [phi_k[k]]:
                    m.addConstr(beta_bkij[b, k, i, j] >= d_bkj[b, k, i] + t_ij[i, j] - big_M * (1 - x_bkij[b, k, i, j]))
                    m.addConstr(beta_bkij[b, k, i, j] <= big_M * (1 - x_bkij[b, k, i, j]))
                    m.addConstr(beta_bkij[b, k, i, j] <= d_bkj[b, k, i] + t_ij[i, j])


    m.optimize()


if __name__ == '__main__':
    from problems import ex1
    mip(ex1)
