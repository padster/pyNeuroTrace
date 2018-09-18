import numpy as np

def treePostProcessing(dataIDs, nodeXYZ, data, rootID, tree):
    # 1) Add location to all points that have it:
    if nodeXYZ is not None:
        for i in range(len(dataIDs)):
            tree[dataIDs[i]]['location'] = nodeXYZ[i]

    # 2) Calculate branches for each node
    branchIDMap = buildBranchIDMap(rootID, tree, splitAtBranch=True)
    branchIDs = [branchIDMap[nodeID] for nodeID in dataIDs]

    # 3) Reorder the nodes by branch
    idOrder = [i for i in range(len(dataIDs))]
    idOrder.sort(key=lambda a: (branchIDs[a], a))
    dataIDs = np.array(dataIDs)[idOrder].tolist()
    branchIDs = np.array(branchIDs)[idOrder].tolist()
    if nodeXYZ is not None:
        nodeXYZ = nodeXYZ[idOrder]
    data = data[idOrder]
    return dataIDs, nodeXYZ, data, branchIDs, branchIDMap

def buildBranchIDMap(nodeID, nodes, splitAtBranch=False):
    result = {}
    _fillBranchIDMap(nodeID, nodes, 0, result, splitAtBranch)
    return result

def _fillBranchIDMap(nodeID, nodes, branchAt, result, splitAtBranch):
    result[nodeID] = branchAt
    lastBranchUsed = branchAt
    if len(nodes[nodeID]['children']) > 0:
        if len(nodes[nodeID]['children']) == 1 or not splitAtBranch or nodes[nodeID]['type'] == 'Root':
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

# Return dictionary mapping: node ID -> (branchID if filo tip, -branchID if filo base, or 0 if neither)
def treeToFiloTipAndBase(nodeIDs, nodeXYZ, tree, rootID, filoDist=5.0):
    mapping = {}
    nextBranchID = [0]
    
    # Inner function to process a single node, given where it started and the distance to there.
    def _filoInner(nodeAtID, branchStartID, branchDist):
        # Default to nothing, change later if needed.
        mapping[nodeAtID] = 0
        
        nChildren = len(tree[nodeAtID]['children'])
        nodeAtIdx = nodeIDs.index(nodeAtID) 
        pAt = nodeXYZ[nodeAtIdx]

        # middle of branch, so just keep going:
        if nChildren == 1:
            nextNodeID = tree[nodeAtID]['children'][0]['id']
            pNext = nodeXYZ[nodeIDs.index(nextNodeID)]
            dist = np.linalg.norm(pNext - pAt)
            _filoInner(nextNodeID, branchStartID, branchDist + dist)
        # Child branches, so clear and start new below
        elif nChildren > 1:
            for child in tree[nodeAtID]['children']:
                pNext = nodeXYZ[nodeIDs.index(child['id'])]
                dist = np.linalg.norm(pNext - pAt)
                # Note: Base is the first in the filo, not the parent as it may have multiple.
                _filoInner(child['id'], child['id'], dist)
        # Filopodia tip (scale from m to um)
        elif (branchDist * 1e6) <= filoDist:
            mapping[branchStartID] = -nextBranchID[0]
            mapping[nodeAtID] = nextBranchID[0]
            nextBranchID[0] += 1
        # Branch tip (ignore)
        else:
            pass
        
    _filoInner(rootID, rootID, 0)
    return mapping
