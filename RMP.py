from gurobipy import *


def generate_RMP(prmt, add_inputs):
    C, p_c, e_ci = list(map(add_inputs.get, ['C', 'p_c', 'e_ci']))
    includeBNB = True if 'inclusiveC' in add_inputs else False
    #
    bB, T = list(map(prmt.get, ['bB', 'T']))
    #
    # Define decision variables
    #
    RMP = Model('RMP')
    q_c = {}
    for c in range(len(C)):
        q_c[c] = RMP.addVar(vtype=GRB.BINARY, name="q[%d]" % c)
    RMP.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for c in range(len(C)):
        obj += p_c[c] * q_c[c]
    RMP.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    #
    taskAC = {}
    for i in T:  # eq:taskA
        taskAC[i] = RMP.addConstr(quicksum(e_ci[c][i] * q_c[c] for c in range(len(C))) <= 1,
                                  name="taskAC[%d]" % i)
    numBC = RMP.addConstr(quicksum(q_c[c] for c in range(len(C))) <= bB,
                              name="numBC")
    if includeBNB:
        inclusiveC, exclusiveC = list(map(add_inputs.get, ['inclusiveC', 'exclusiveC']))
        C_i0i1 = {}
        for i0, i1 in set(inclusiveC).union(set(exclusiveC)):
            for c in range(len(C)):
                if i0 in C[c] and i1 in C[c]:
                    if (i0, i1) not in C_i0i1:
                        C_i0i1[i0, i1] = []
                    C_i0i1[i0, i1].append(c)
        #
        for i, (i0, i1) in enumerate(inclusiveC):
            RMP.addConstr(quicksum(q_c[b] for b in C_i0i1[i0, i1]) >= 1,
                              name="mIC[%d]" % i)
        for i, (i0, i1) in enumerate(exclusiveC):
            RMP.addConstr(quicksum(q_c[b] for b in C_i0i1[i0, i1]) <= 0,
                              name="mEC[%d]" % i)
    RMP.update()
    #
    return RMP, q_c, taskAC, numBC