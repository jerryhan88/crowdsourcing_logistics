import __init__
from init_project import *
#
from gurobipy import *

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
    x_bkij = {}
    for b in B:
        for k in K:
            for i in V:
                for j in V:
                    x_bkij[b, k, i, j] = m.addVar(vtype=GRB.BINARY, name='x_(%d,%d,%d,%d)' % (b, k, i, j))
    y_bk = {}
    for b in B:
        for k in K:
            y_bk[b, k] = m.addVar(vtype=GRB.BINARY, name='y_(%d,%d)' % (b, k))
    z_bi = {}
    for b in B:
        for i in xrange(_n):
            z_bi[b, i] = m.addVar(vtype=GRB.BINARY, name='z_(%d,%d)' % (b, i))
    R_b = {}
    for b in B:
        R_b[b] = m.addVar(name='R_(%d)' % b)
    d_bkj = {}
    for b in B:
        for k in K:
            for j in N + Phi:
                d_bkj[b, k, j] = m.addVar(vtype=GRB.BINARY, name='d_(%d,%d,%d)' % (b, k, j))
    m.update()
    #
    # Define constrains
    #








if __name__ == '__main__':
    from problems import ex1
    mip(ex1)
