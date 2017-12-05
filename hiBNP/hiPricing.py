from gurobipy import *
#
from optRouting import run as optR_run
from _utils.recording import *
from _utils.mm_utils import *


def solve_knapsack(counter, problem):
    T, r_i, v_i, _lambda, \
    inclusiveC, exclusiveC, \
    pi_i, mu = [problem.get(k) for k in ['T', 'r_i', 'v_i', '_lambda',
                                         'inclusiveC', 'exclusiveC',
                                         'pi_i', 'mu']]
    _r_i = {}
    for i in T:
        _r_i[i] = r_i[i] - pi_i[i]
    #
    knapsackM = Model('knapsackProblem %d' % counter)
    #
    z_i = {}
    s_PN = {}
    for i in T:
        z_i[i] = knapsackM.addVar(vtype=GRB.BINARY, name='z[%d]' % i)
    for s in range(len(inclusiveC)):
        sP, sN = 'sP[%d]' % s, 'sN[%d]' % s
        s_PN[sP] = knapsackM.addVar(vtype=GRB.BINARY, name=sP)
        s_PN[sN] = knapsackM.addVar(vtype=GRB.BINARY, name=sN)
    #
    obj = LinExpr()
    for i in T:
        obj += _r_i[i] * z_i[i]
    knapsackM.setObjective(obj, GRB.MAXIMIZE)
    #
    # Volume
    knapsackM.addConstr(quicksum(v_i[i] * z_i[i] for i in T) <= _lambda,
                       name='vt')  # eq:volTh
    # Handling inclusive constraints
    for s in range(len(inclusiveC)):
        i0, i1 = inclusiveC[s]
        sP, sN = 'sP[%d]' % s, 'sN[%d]' % s
        knapsackM.addConstr(s_PN[sP] + s_PN[sN] == 1,
                           name='iC[%d]' % s)
        knapsackM.addConstr(2 * s_PN[sP] <= z_i[i0] + z_i[i1],
                           name='iCP[%d]' % s)
        knapsackM.addConstr(z_i[i0] + z_i[i1] <= 2 * (1 - s_PN[sN]),
                           name='iCN[%d]' % s)
    # Handling exclusive constraints
    for i, (i0, i1) in enumerate(exclusiveC):
        knapsackM.addConstr(z_i[i0] + z_i[i1] <= 1,
                           name='eC[%d]' % i)
    #
    knapsackM.optimize()
    #
    nSolutions = knapsackM.SolCount
    if nSolutions == 0:
        assert False
    bestSols = []
    if nSolutions == 1 and knapsackM.objVal > 0:
        bestSols.append((knapsackM.objVal, [i for i in T if z_i[i].x > 0.5]))
    else:
        for e in range(nSolutions):
            knapsackM.setParam(GRB.Param.SolutionNumber, e)
            if knapsackM.PoolObjVal > 0:
                bestSols.append((knapsackM.PoolObjVal, [i for i in T if z_i[i].Xn > 0.5]))
    return bestSols


def run(counter,
        pi_i, mu, B, input4subProblem,
        inclusiveC, exclusiveC,
        grbSetting):
    #
    T, r_i, v_i, _lambda, P, D, N, K, w_k, t_ij, _delta = input4subProblem
    problem = {'T': T,
               'r_i': r_i,
               'v_i': v_i,
               '_lambda': _lambda,
               'inclusiveC': inclusiveC,
               'exclusiveC': exclusiveC,
               'pi_i': pi_i,
               'mu': mu}

    nsBestSols = solve_knapsack(counter, problem)
    grbSettingOP = {}
    bestSols = []
    for objV, b in nsBestSols:
        if objV < 0:
            break
        br = sum([r_i[i] for i in b])
        p = 0
        for k, w in enumerate(w_k):
            probSetting = {'b': b, 'k': k, 't_ij': t_ij}
            detourTime, route = optR_run(probSetting, grbSettingOP)
            if detourTime <= _delta:
                p += w * br
        p -= sum([pi_i[i] for i in b])
        p -= mu
        bestSols.append((p, b))
    return bestSols


if __name__ == '__main__':
    pass