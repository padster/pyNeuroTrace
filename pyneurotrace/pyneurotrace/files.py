import numpy as np

# Denotes the start of stimulus index lines in metadata.
STIM_INDEX_KEY = '#STIM_START_STOP_INDICES'
# Sample rate inverse line in metadata
SAMPLERATE_KEY = '#MSMT_BLK_DUR'

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
    at = lines.index(SAMPLERATE_KEY) + 1
    return np.array(stimIndices), round(1.0 / float(lines[at]))

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

# Given lines and the current offset, verify the lines are the correct type, and return the values
def _treeVerify(at, lines, lineTypes):
    n = len(lineTypes)
    data = []
    for i in range(n):
        assert lines[at + i][0] == lineTypes[i]
        data.append(lines[at + i][1])
    return at + n, tuple(data)
