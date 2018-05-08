# Given a tree from file, and IDs in trace data, reorder the tree so that the child
# order matches the trace data order, and return [start, end) indices for each branch
"""
HACK: Need to figure out the good way to do this...
def reorderTreeForTraces(nodeID, nodes, dataIDs):
    if len(nodes[nodeID]['children']) > 1:
        nodes[nodeID]['children'].sort(key=lambda child: dataIDs.index(child['id']))
    for child in nodes[nodeID]['children']:
        reorderTreeForTraces(child['id'], nodes, dataIDs)

def getBranchIDs(nodeID, nodes):
    flatIDs = _flattenIDs(nodeID, nodes)
    nextChildren, branchIDs = [], []
    branchAt = -1
    for idx in flatIDs:
        if idx not in nextChildren:
            branchAt += 1
        branchIDs.append(branchAt)
        nextChildren = [n['id'] for n in nodes[idx]['children']]
    return branchIDs
"""

def buildBranchIDMap(nodeID, nodes, splitAtBranch=False):
    result = {}
    _fillBranchIDMap(nodeID, nodes, 0, result, splitAtBranch)
    return result

def _fillBranchIDMap(nodeID, nodes, branchAt, result, splitAtBranch):
    result[nodeID] = branchAt
    lastBranchUsed = branchAt
    if len(nodes[nodeID]['children']) > 0:
        if len(nodes[nodeID]['children']) == 1 or not splitAtBranch:
            lastBranchUsed -= 1
        for child in nodes[nodeID]['children']:
            lastBranchUsed = _fillBranchIDMap(child['id'], nodes, lastBranchUsed + 1, result, splitAtBranch)
    return lastBranchUsed

def _flattenIDs(nodeID, nodes):
    ids = [nodeID]
    for child in nodes[nodeID]['children']:
        ids.extend(_flattenIDs(child['id'], nodes))
    return ids

def _treeSize(nodeID, nodes):
    size = 1
    for child in nodes[nodeID]['children']:
        size += _treeSize(child['id'], nodes)
    return size
