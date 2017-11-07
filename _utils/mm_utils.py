EPSILON = 0.000000001


def set_grbSettings(mm, grb_settings):
    # if TimeLimit is not None:
    #     pricingM.setParam('TimeLimit', TimeLimit)
    # if numThreads is not None:
    #     pricingM.setParam('Threads', numThreads)
    # if log_fpath is not None:
    #     pricingM.setParam('LogFile', log_fpath)
    # if numPoolSols is not None:
    #     pricingM.setParam('PoolSolutions', numPoolSols)
    for k, v in grb_settings.items():
        mm.setParam(k, v)

def route_display(edges):
    route = [e for e in edges if e[0].startswith('ori')]
    while len(route) != len(edges):
        for e in edges:
            if e[0] == route[-1][1]:
                route.append(e)
                break
    return route

def get_routeFromOri(edges, nodes):
    visited, adj = {}, {}
    for i in nodes:
        visited[i] = False
        adj[i] = []
    for i, j in edges:
        adj[i].append(j)
    route = []
    cNode = None
    for n in nodes:
        if n.startswith('ori'):
            cNode = n
            break
    else:
        assert False
    while not cNode.startswith('dest'):
        visited[cNode] = True
        neighbors = [j for j in adj[cNode] if not visited[j]]
        route.append((cNode, neighbors[0]))
        cNode = neighbors[0]
        if visited[cNode]:
            break
    return route