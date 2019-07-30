import multiprocessing
from gurobipy import *
#
NUM_CORES = multiprocessing.cpu_count()

#
# Mathematical Model for the pickup and delivery problem
#

def run(prmt, pd_inputs):
    k, Ts = [pd_inputs.get(k) for k in ['k', 'Ts']]
    _kP, _kM = 'ori%d' % k, 'dest%d' % k
    P, D = set(), set()
    for i in Ts:
        P.add('p%d' % i)
        D.add('d%d' % i)
    N = P.union(D)
    kN = N.union({_kP, _kM})
    t_ij = prmt['t_ij']
    bigM = len(N) + 2
    #
    # Define decision variables
    #
    PD = Model('PD')
    x_ij, o_i = {}, {}
    for i in kN:
        o_i[i] = PD.addVar(vtype=GRB.INTEGER, name='o[%s]' % i)
        for j in kN:
            x_ij[i, j] = PD.addVar(vtype=GRB.BINARY, name='x[%s,%s]' % (i, j))
    PD.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for i in kN:
        for j in kN:
            obj += t_ij[i, j] * x_ij[i, j]
    obj -= t_ij[_kP, _kM]
    PD.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constrains
    #  Routing_Flow
    #
    #  # eq:initFlow
    PD.addConstr(quicksum(x_ij['ori%d' % k, j] for j in kN) == 1, name='pPDo')
    PD.addConstr(quicksum(x_ij[j, 'dest%d' % k] for j in kN) == 1, name='pPDi')
    #  # eq:noInFlowOutFlow
    PD.addConstr(quicksum(x_ij[j, 'ori%d' % k] for j in kN) == 0, name='XpPDo')
    PD.addConstr(quicksum(x_ij['dest%d' % k, j] for j in kN) == 0, name='XpPDi')
    for i in Ts:
        #  # eq:taskOutFlow
        PD.addConstr(quicksum(x_ij['p%d' % i, j] for j in kN) == 1, name='tOF[%d]' % i)
        #  # eq:taskInFlow
        PD.addConstr(quicksum(x_ij[j, 'd%d' % i] for j in kN) == 1, name='tIF[%d]' % i)
    for i in N:  # eq:flowCon
        PD.addConstr(quicksum(x_ij[i, j] for j in kN) == quicksum(x_ij[j, i] for j in kN),
                     name='fc[%s]' % i)
    #
    #  Routing_Ordering
    #
    N_kM = N.union({_kM})
    for i in N_kM:
        #  # eq:initOrder
        PD.addConstr(2 <= o_i[i], name='initO1[%s]' % i)
        PD.addConstr(o_i[i] <= bigM, name='initO2[%s]' % i)
        for j in N_kM:
                #  # eq:subEli
                PD.addConstr(o_i[i] + 1 <= o_i[j] + bigM * (1 - x_ij[i, j]),
                     name='subEli[%s,%s]' % (i, j))
    for i in Ts:
        #  # eq:pdSequence
        PD.addConstr(o_i['p%d' % i] <= o_i['d%d' % i], name='pdS[%d]' % i)
    #
    # Run Gurobi (Optimization)
    #
    PD.setParam('OutputFlag', False)
    PD.setParam('Threads', NUM_CORES)
    PD.optimize()
    #
    _route = {}
    for j in kN:
        for i in kN:
            if x_ij[i, j].x > 0.5:
                _route[i] = j
    i = _kP
    route = []
    while i != _kM:
        route.append(i)
        i = _route[i]
    route.append(i)
    #
    return PD.objVal, route


def find_all_feasible_paths(prmt_fpath, sol_fpath, colFP_dpath):
    import os.path as opath
    import pickle
    #
    with open(prmt_fpath, 'rb') as fp:
        prmt = pickle.load(fp)
    with open(sol_fpath, 'rb') as fp:
        sol = pickle.load(fp)
    C, q_c = [sol.get(k) for k in ['C', 'q_c']]
    selCols = [(c, C[c]) for c in range(len(C)) if q_c[c] > 0.5]
    for bid, (c, Ts) in enumerate(selCols):
        feasiblePath = []
        for k in prmt['K']:
            detourTime, route = run(prmt, {'k': k, 'Ts': Ts})
            if detourTime <= prmt['_delta']:
                feasiblePath.append(k)
        colFP_fpath = opath.join(colFP_dpath, 'bid%d.pkl' % bid)
        #
        with open(colFP_fpath, 'wb') as fp:
            pickle.dump([c, Ts, feasiblePath], fp)


def handle_all_instances():
    import os.path as opath
    import os
    from functools import reduce
    #
    from __path_organizer import exp_dpath
    #
    prmt_dpath = reduce(opath.join, [exp_dpath, '_summary', 'prmt'])
    sol_dpath = reduce(opath.join, [exp_dpath, '_summary', 'sol'])
    selColFP_dpath = opath.join(sol_dpath, 'selColFP')
    if not opath.exists(selColFP_dpath):
        os.mkdir(selColFP_dpath)
    #
    for fn in sorted(os.listdir(sol_dpath)):
        if 'CWL' not in fn:
            continue
        if not fn.endswith('.pkl'):
            continue
        print(fn)
        _, prefix, aprc = fn[:-len('.csv')].split('_')
        scFP_dpath = opath.join(selColFP_dpath, 'scFP_%s_%s' % (prefix, aprc))
        if not opath.exists(scFP_dpath):
            os.mkdir(scFP_dpath)
            prmt_fpath = opath.join(prmt_dpath, 'prmt_%s.pkl' % prefix)
            sol_fpath = opath.join(sol_dpath, fn)
            find_all_feasible_paths(prmt_fpath, sol_fpath, scFP_dpath)


def handle_aInstances():
    import os.path as opath
    import os
    from functools import reduce
    #
    from __path_organizer import exp_dpath
    #
    prmt_dpath = reduce(opath.join, [exp_dpath, '_summaryScl', 'prmt'])
    sol_dpath = reduce(opath.join, [exp_dpath, '_summaryScl', 'sol'])
    selColFP_dpath = opath.join(sol_dpath, 'selColFP')
    if not opath.exists(selColFP_dpath):
        os.mkdir(selColFP_dpath)
    #
    fn = 'sol_11interOut-nt200-mDP20-mTB4-dp25-fp75-sn0_CWL4.pkl'
    _, prefix, aprc = fn[:-len('.csv')].split('_')
    scFP_dpath = opath.join(selColFP_dpath, 'scFP_%s_%s' % (prefix, aprc))
    if not opath.exists(scFP_dpath):
        os.mkdir(scFP_dpath)
        prmt_fpath = opath.join(prmt_dpath, 'prmt_%s.pkl' % prefix)
        sol_fpath = opath.join(sol_dpath, fn)
        find_all_feasible_paths(prmt_fpath, sol_fpath, scFP_dpath)



if __name__ == '__main__':
    # handle_all_instances()
    handle_aInstances()
