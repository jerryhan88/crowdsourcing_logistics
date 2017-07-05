import __init__
from init_project import *
#


def ex1():
    #
    # Define a network
    #
    points, distance = {}, {}
    pid = 0
    for i in range(3):
        for j in range(3):
            points[pid] = point(pid, i, j)
            pid += 1
    for p0 in points.itervalues():
        for p1 in points.itervalues():
            distance[p0.pid, p1.pid] = abs(p0.i - p1.i) + abs(p0.j - p1.j)
    #
    # Inputs related to bundles
    #
    _o = 3
    B = range(_o)
    _lambda = 5
    _delta = 100
    #
    # Inputs related to tasks
    #   Define tasks and their pickup and delivery points
    #
    tasks = [(0, 2), (2, 1), (3, 5), (2, 7), (5, 1),
             (6, 8), (2, 6), (8, 4), (7, 3), (5, 1)]
    _n = len(tasks)
    r_i = [1, 2, 3, 2, 1,
           3, 1, 2, 1, 2]
    v_i = [1, 1, 1, 1, 1,
           1, 1, 1, 1, 1]
    T = list(range(_n))
    P = list(range(_n))
    D = list(range(_n, 2 * _n))
    N = P + D
    #
    # Inputs related to paths
    #   Define flows and calculate flows' weight
    #
    paths = [(i, j) for i in xrange(len(points)) for j in xrange(len(points)) if i != j]
    _m = len(paths)
    K = range(_m)
    Omega = list(range(2 * _n, 2 * _n + _m))
    Pi = list(range(2 * _n + _m, 2 * _n + 2 * _m))
    omega_k = [Omega[k] for k in K]
    pi_k = [Pi[k] for k in K]
    f_k = [
        [0, 1, 2, 1, 1, 2, 3, 2, 0],
        [0, 0, 2, 1, 2, 1, 1, 2, 1],
        [2, 1, 0, 2, 1, 1, 2, 1, 0],
        [1, 0, 3, 0, 1, 1, 2, 0, 2],
        [3, 2, 1, 2, 0, 1, 0, 2, 3],
        [2, 3, 1, 0, 2, 0, 3, 1, 1],
        [1, 1, 3, 2, 0, 2, 0, 3, 0],
        [0, 2, 2, 3, 0, 2, 2, 0, 2],
        [2, 2, 2, 3, 2, 0, 3, 0, 0]
    ]
    sum_f_k = sum(f_k[i][j] for i in xrange(len(f_k)) for j in xrange(len(f_k)))
    w_k = [f_k[i][j] / float(sum_f_k) for i, j in paths]
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
            t_ij[i, j] = distance[p0, p1]
    return _o, B, _lambda, _delta, \
            _n, T, P, D, N, r_i, v_i, \
            _m, K, Omega, Pi, omega_k, pi_k, f_k, w_k, \
            V, t_ij


def ex2():
    #
    # Define a network
    #
    points, distance = {}, {}
    pid = 0
    for i in range(2):
        for j in range(2):
            points[pid] = point(pid, i, j)
            pid += 1
    for p0 in points.itervalues():
        for p1 in points.itervalues():
            distance[p0.pid, p1.pid] = abs(p0.i - p1.i) + abs(p0.j - p1.j)
    #
    # Inputs related to bundles
    #
    _o = 1
    B = range(_o)
    _lambda = 10
    _delta = 1000
    #
    # Inputs related to tasks
    #   Define tasks and their pickup and delivery points
    #
    tasks = [(0, 2)]

    r_i = [1]
    v_i = [1]
    _n = len(tasks)
    T = list(range(_n))
    P = list(range(_n))
    D = list(range(_n, 2 * _n))
    N = P + D
    #
    # Inputs related to paths
    #   Define flows and calculate flows' weight
    #
    paths = [(i, j) for i in xrange(len(points)) for j in xrange(len(points)) if i != j]
    _m = len(paths)
    K = range(_m)
    Omega = list(range(2 * _n, 2 * _n + _m))
    Pi = list(range(2 * _n + _m, 2 * _n + 2 * _m))
    omega_k = [Omega[k] for k in K]
    pi_k = [Pi[k] for k in K]
    f_k = [
        [0, 1, 2, 1],
        [0, 0, 2, 1],
        [2, 1, 0, 2],
        [1, 0, 3, 0],
    ]
    sum_f_k = sum(f_k[i][j] for i in xrange(len(f_k)) for j in xrange(len(f_k)))
    w_k = [f_k[i][j] / float(sum_f_k) for i, j in paths]
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
            t_ij[i, j] = distance[p0, p1]
    return _o, B, _lambda, _delta, \
            _n, T, P, D, N, r_i, v_i, \
            _m, K, Omega, Pi, omega_k, pi_k, f_k, w_k, \
            V, t_ij


class point(object):
    def __init__(self, pid, i, j):
        self.pid, self.i, self.j = pid, i, j

    def __repr__(self):
        return 'pid%d' % self.pid


if __name__ == '__main__':
    ex1()