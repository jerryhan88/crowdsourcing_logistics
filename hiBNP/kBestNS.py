from heapq import heappush, heappop
#
# volume is already sorted
#
v_i = [3, 4, 5, 6, 7]
p_i = [4, 3, 5, 7, 8]
L = 15
numItems = len(v_i)
minV = 1e400
for v in v_i:
    if v < minV:
        minV = v

nodes = {}
class nsNode(object):
    def __init__(self, nid):
        self.nid = nid
        self.objectives = []

    def __repr__(self):
        return 'nid%d' % self.nid

#
# Level 0 and 1
#
rootNode = nsNode(0)
nodes[0] = rootNode
rootNode.objectives = [0 for _ in range(numItems)]

def creat_aNode(new_nid):
    new_node = nsNode(new_nid)
    new_node.objectives = [-1 for _ in range(numItems)]
    return new_node



feasibleSols = []
leafNodes = []
for i, v in enumerate(v_i):
    new_nid = rootNode.nid + v
    new_objV = p_i[i]
    if new_nid not in nodes:
        new_node = creat_aNode(new_nid)
        new_node.objectives[i] = new_objV
        #
        nodes[new_nid] = new_node
        heappush(leafNodes, (new_nid, new_node))
    else:
        nodes[new_nid].objectives[i] = new_objV
    heappush(feasibleSols, (-new_objV, new_nid))


while leafNodes:
    _, n = heappop(leafNodes)
    minFI = numItems
    for i in range(numItems):
        objV = n.objectives[i]
        if objV != -1 and i < minFI:
            minFI = i

    bestObjV = -1e400
    for i in range(minFI, numItems):
        objV = n.objectives[i]
        if bestObjV < objV:
            bestObjV = objV
        #
        new_nid = n.nid + v_i[i]
        if L < new_nid:
            continue
        new_objV = bestObjV + p_i[i]
        if new_nid not in nodes:
            new_node = creat_aNode(new_nid)
            new_node.objectives[i] = new_objV
            #
            nodes[new_nid] = new_node
            heappush(leafNodes, (new_nid, new_node))
        else:
            nodes[new_nid].objectives[i] = new_objV
        heappush(feasibleSols, (-new_objV, new_nid))


def backtracking(objV, nid):
    n = nodes[nid]
    for chosen_item in range(numItems):
        if n.objectives[chosen_item] == objV:
            break
    else:
        assert False
    prev_objV = objV - p_i[chosen_item]
    prev_nid = nid - v_i[chosen_item]
    #
    return chosen_item, prev_objV, prev_nid

while True:
    _objV, nid = heappop(feasibleSols)
    objV = iterObjV = -_objV
    ks_comp = []
    while True:
        # print(iterObjV, nid)
        chosen_item, prev_objV, prev_nid = backtracking(iterObjV, nid)
        ks_comp.append(chosen_item)
        if prev_nid == 0:
            break
        else:
            iterObjV, nid = prev_objV, prev_nid
    print(objV, ks_comp)
    # assert False


