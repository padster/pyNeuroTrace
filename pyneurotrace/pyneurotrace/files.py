import numpy as np

# Denotes the start of stimulus index lines in metadata.
STIM_INDEX_KEY = '#STIM_START_STOP_INDICES'
# Sample rate inverse line in metadata
SAMPLERATE_KEY = '#MSMT_BLK_DUR'
# XY pixel size in world meters, in metadata
PIXEL_SIZE_KEY = '#PIXEL_SIZE_M'
# Z stack locations in metadata
Z_STACK_LOCATIONS_KEY = '#IMAGE_STACK_LOCATIONS_M'


"""
Load Node IDs, positions, and raw data from an EXPT.TXT file.
"""
def load2PData(path, hasLocation=True):
    print ("Loading 2P data from " + path)
    data = np.loadtxt(path)
    # With N nodes, this returns:
    # Node IDs (N), XYZ (N x 3), raw trace (N x S samples)
    nodeIDs = [int(idx) for idx in data[:, 0]]
    if hasLocation:
        return nodeIDs, data[:, 1:4], data[:, 4:]
    else:
        return nodeIDs, None, data[:, 1:]

"""
Load stimulus [start, end] sample indices, plus sample rate, from the rscan_metadata .txt file
"""
def loadMetadata(path):
    print ("Loading stim times from " + path)
    lines = []
    with open(path) as f:
        lines = [line.strip() for line in f.readlines()]
    if STIM_INDEX_KEY not in lines or SAMPLERATE_KEY not in lines:
        raise Exception("No stim and sample rate keys")

    stimIndices = []
    at = lines.index(STIM_INDEX_KEY) + 1
    while at < len(lines):
        if lines[at][0] == '#':
            break
        indices = lines[at].split('\t')
        assert len(indices) == 2
        stimIndices.append([int(index) for index in indices])
        at += 1

    inverse_hz = float(lines[lines.index(SAMPLERATE_KEY) + 1])
    hz = round(1.0 / inverse_hz)

    # Two new optional metadata fields added Sept 2018:
    xySizeM = 1.0
    if PIXEL_SIZE_KEY in lines:
        xySizeM = float(lines[lines.index(PIXEL_SIZE_KEY) + 1])

    zStackLocations = []
    if Z_STACK_LOCATIONS_KEY:
        at = lines.index(Z_STACK_LOCATIONS_KEY) + 1
        while at < len(lines):
            if lines[at][0] == '#':
                break
            zStackLocations.append(float(lines[at]))
            at += 1
    return np.array(stimIndices), hz, xySizeM, np.array(zStackLocations)

"""
Load Tree structure (branch & parent details) from an interp-neuron-.txt file
"""
def loadTreeStructure(path):
    print ("Loading tree structure from " + path)
    lines = []
    with open(path) as f:
        lines = [line.strip().split(':') for line in f.readlines()]

    rootId, at, nodes = None, 0, {}
    while at < len(lines):
        at, (nodeId, nodeType, parentId, parentType, nChildren) = _treeVerify(
            at, lines, ['NodeID', 'NodeType', 'NodeID', 'ParentType', 'NumChildren']
        )
        nodeId, parentId = int(nodeId), int(parentId)
        children = []
        for c in range(int(nChildren)):
            at, (childId, childType) = _treeVerify(at, lines, ['NodeID', 'ChildType'])
            childId = int(childId)
            children.append({
                'id':   childId,
                'type': childType,
            })
        nodes[nodeId] = {
            'id':       nodeId,
            'type':     nodeType,
            'parentId': parentId,
            'children': children,
        }
        if (nodeType == 'Root'):
            rootId = int(nodeId)

    return rootId, nodes

"""
Load (nodeID, x, y, z) from a file, separate from the traces file.
Return as the list of IDs, then the list of positions.
This supports having positions for nodes which have no trace recorded.
"""
def loadNodeXYZ(path):
    print ("Loading XYZ data from " + path)
    data = np.loadtxt(path)
    # With N nodes, this returns:
    # Node IDs (N), XYZ (N x 3)
    nodeIDs = [int(idx) for idx in data[:, 0]]
    return nodeIDs, data[:, 1:4]

"""
Load kymograph - a big (R x C) matrix with:
a) First row is node IDs, every pxlPerNode (the rest being -1)
b) The remaining are concatenated blocks of (R - 1) x (pxlPerNode) kymograph intensities
This parses them out, and returns as a mapping of node ID -> Kymograph data.
"""
def loadKymograph(path, pxlPerNode=11):
    print ("Loading Kymograph data from " + path)
    data = np.loadtxt(path)
    assert data.shape[1] % pxlPerNode == 0
    result = {}
    for i in range(0, data.shape[1], pxlPerNode):
        result[data[0, i]] = data[1:, i:(i+pxlPerNode)]
    return result
    
# Given lines and the current offset, verify the lines are the correct type, and return the values
def _treeVerify(at, lines, lineTypes):
    n = len(lineTypes)
    data = []
    for i in range(n):
        assert lines[at + i][0] == lineTypes[i]
        data.append(lines[at + i][1])
    return at + n, tuple(data)
