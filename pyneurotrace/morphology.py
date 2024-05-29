import numpy as np

def treePostProcessing(nodeIDs, nodeXYZ, traceIDs, data, rootID, tree):
    """
    Processes the tree structure, adding locations, calculating branches, and reordering nodes by branch.

    Parameters
    ----------
    nodeIDs : array
        List of node IDs.
    nodeXYZ : array
        Array of node locations (XYZ coordinates).
    traceIDs : array
        List of trace IDs.
    data : array
        Data array containing raw traces.
    rootID : int
        ID of the root node.
    tree : dict
        Dictionary representing the tree structure.

    Returns
    -------
    nodeIDs : array
        Processed list of node IDs.
    nodeXYZ : array
        Processed array of node locations (XYZ coordinates).
    finalTraceIDs : array
        Processed list of trace IDs.
    finalTraceBranches : array
        List of trace branches.
    data : array
        Processed data array containing raw traces.
    branchIDs : array
        List of branch IDs.
    branchIDMap : dict
        Dictionary mapping node IDs to branch IDs.
    """

    # 1) Add location to all points that have it:
    if nodeXYZ is not None:
        for i in range(len(nodeIDs)):
            tree[nodeIDs[i]]['location'] = nodeXYZ[i]

    # 2) Calculate branches for each node
    branchIDMap = buildBranchIDMap(rootID, tree, splitAtBranch=True)
    branchIDs = [branchIDMap[nodeID] for nodeID in nodeIDs]
    traceBranches = []
    for traceID in traceIDs:
        if traceID in branchIDMap:
            traceBranches.append(branchIDMap[traceID])
        else:
            traceBranches.append(-1)

    # 3) Reorder the nodes by branch
    nodeIDOrder = [i for i in range(len(nodeIDs))]
    traceIDOrder = [i for i in range(len(traceIDs))]
    nodeIDOrder.sort(key=lambda a: (branchIDs[a], a))
    traceIDOrder.sort(key=lambda a: (traceBranches[a], a))
    
    nodeIDs = np.array(nodeIDs)[nodeIDOrder].tolist()
    branchIDs = np.array(branchIDs)[nodeIDOrder].tolist()
    if nodeXYZ is not None:
        nodeXYZ = nodeXYZ[nodeIDOrder]
    data = data[traceIDOrder]
    finalTraceIDs = np.array(traceIDs)[traceIDOrder].tolist()
    finalTraceBranches = np.array(traceBranches)[traceIDOrder].tolist()
    return nodeIDs, nodeXYZ, finalTraceIDs, finalTraceBranches, data, branchIDs, branchIDMap

def buildBranchIDMap(nodeID, nodes, splitAtBranch=False):
    """
    Builds a map of branch IDs for the given tree structure.

    Parameters
    ----------
    nodeID : int
        ID of the starting node.
    nodes : dict
        Dictionary representing the tree structure.
    splitAtBranch : bool, optional
        Flag indicating whether to split at branches. Default is `False`.

    Returns
    -------
    branchIDMap : dict
        Dictionary mapping node IDs to branch IDs.
    """
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
    """
    Maps nodes to filopodia tips and bases based on the specified distance.

    Parameters
    ----------
    nodeIDs : array
        List of node IDs.
    nodeXYZ : array
        Array of node locations (XYZ coordinates).
    tree : dict
        Dictionary representing the tree structure.
    rootID : int
        ID of the root node.
    filoDist : float, optional
        Distance threshold for identifying filopodia tips and bases. Default is `5.0`.

    Returns
    -------
    mapping : dict
        Dictionary mapping node IDs to branch IDs, indicating filopodia tips and bases.
    """
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
                # NO LONGER APPLIES: # Note: Base is the first in the filo, not the parent as it may have multiple.
                _filoInner(child['id'], child['id'], dist) #nodeAtID, dist)
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
