from math import sqrt
from random import seed, randint
from gurobipy import *


def subtourelim(model, where):
    if where == GRB.callback.MIPSOL:
        selected = []
        for i in xrange(n):
            sol = model.cbGetSolution([model._x_ij[i, j] for j in xrange(n)])
            selected += [(i, j) for j in xrange(n) if sol[j] > 0.5]
        print selected
        min_subtour = get_min_subtour(selected)
        print min_subtour
        if len(min_subtour) < n:
            expr = 0
            for i in xrange(len(min_subtour)):
                for j in xrange(i + 1, len(min_subtour)):
                    expr += model._x_ij[min_subtour[i], min_subtour[j]]
            model.cbLazy(expr <= len(min_subtour) - 1)

def get_min_subtour(edges):
    cycles = []
    visited, selected = [], []
    for _ in xrange(n):
        visited.append(False)
        selected.append([])
    for i, j in edges:
        selected[i].append(j)
    while True:
        i = visited.index(False)
        thiscycle = [i]
        while True:
            visited[i] = True
            neighbors = [j for j in selected[i] if not visited[j]]
            if len(neighbors) == 0:
                break
            i = neighbors[0]
            thiscycle.append(i)
        cycles.append(thiscycle)
        if all(visited):
            break
    cycles.sort(key=lambda l: len(l))
    return cycles[0]





seed(1)


n = 7
points = [(randint(0, 100), randint(0, 100)) for _ in xrange(n)]

distances = {}
for i, p0 in enumerate(points):
    for j, p1 in enumerate(points):
        dx = p0[0] - p1[0]
        dy = p0[1] - p1[1]
        distances[i, j] = sqrt(dx * dx + dy * dy)

m = Model()

x_ij = {}
for i in xrange(n):
    for j in xrange(i + 1):
        x_ij[i, j] = m.addVar(vtype=GRB.BINARY, name='x_(%d,%d)' % (i, j))
        x_ij[j, i] = x_ij[i, j]

m.update()

obj = LinExpr()
for i in xrange(n):
    for j in xrange(i + 1):
        obj += distances[i, j] * x_ij[i, j]
m.setObjective(obj, GRB.MINIMIZE)

for i in xrange(n):
    m.addConstr(quicksum(x_ij[i, j] for j in xrange(n)) == 2)
    x_ij[i, i].ub = 0

m._x_ij = x_ij
m.params.LazyConstraints = 1
m.optimize(subtourelim)

solution = m.getAttr('x', x_ij)
selected = [(i, j) for i in xrange(n) for j in xrange(n) if solution[i, j] > 0.5]
assert len(get_min_subtour(selected)) == n
print selected