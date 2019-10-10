import numpy as np
import os

from .morphology import treePostProcessing

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
    if Z_STACK_LOCATIONS_KEY in lines:
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

"""
Loads as many of the above as possible for a single step of an experiment.

:param stepPath: Path to EXPT.TXT raw data traces
:param metaPath: Path to metadata file containing sample rate, stim times, and pixel sizes.
:param treePath: Path to tree structure file.
:param xyzsPath: (optional) Path to mapping from ID to x/y/z location of all points in the tree.
:param kymoPath: (optional) Path to kymograph intensity time series for all scanned points.
:param volumeXYZSource: (optional) Source for position data if not available for all points (e.g. planar scan)
"""
def loadSingleStep(stepPath, metaPath, treePath, xyzsPath, kymoPath, volumeXYZSource, convertXYZtoPx=False, normalizeXYZ=False):
    MAX_VOLUME_HZ = 20 # Hz above this are planar scans, below this are volume scans.
    
    stim, hz, xySizeM, zStackLocations = loadMetadata(metaPath)
    isPlanar = hz > MAX_VOLUME_HZ
    
    # Load XYZ separately if available
    if xyzsPath is not None:
        traceIDs, _, rawData = load2PData(stepPath)
        nodeIDs, xyz = loadNodeXYZ(xyzsPath)
    else: 
        nodeIDs, xyz, rawData = load2PData(stepPath)
        traceIDs = np.copy(nodeIDs)
        # Copy across from the last volume scan
        if isPlanar and volumeXYZSource is not None:
            print ('copying from old')
            nodeIDs = np.copy(volumeXYZSource['nodeIDs'])
            xyz = np.copy(volumeXYZSource['xyz'])

    if convertXYZtoPx:
        xyz = _worldToPixelXYZ(xyz, xySizeM, zStackLocations)
    elif normalizeXYZ:
        xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
        
    rootID, tree = loadTreeStructure(treePath)
    nodeIDs, xyz, traceIDs, traceBranches, rawData, branchIDs, branchIDMap = \
        treePostProcessing(nodeIDs, xyz, traceIDs, rawData, rootID, tree)
    
    kymoData = None
    if kymoPath is not None:
        kymoData = loadKymograph(kymoPath, pxlPerNode=11)
    
    # Tag whether this has a stim transition shift in the middle:
    hasTransition = False
    if stim.shape[0] == 9:
        middleStimSamples = stim[4][1] - stim[4][0]
        middleStimMs = middleStimSamples / hz * 1000.0
        hasTransition = middleStimMs > 1000
    
    return {
        'nodeIDs': nodeIDs,
        'xyz': xyz,
        'rawData': rawData,
        'traceIDs': traceIDs,
        'traceBranches': traceBranches,
        'stim': stim,
        'hz': hz,
        'rootID': rootID,
        'tree': tree,
        'branches': np.array(branchIDs),
        'planar': isPlanar,
        'hasTransition': hasTransition,
        'kymoData': kymoData
    }

"""
Loads an entire hybrid experiment from a folder, containing many scan results.

:param rootPath: Folder to load all the steps from
:param loadKymoData: Whether to load kymographs. Off by default as they're slow to load.
"""
def loadHybrid(rootPath, loadKymoData=False, convertXYZtoPx=False, getPlanarXYZFromVolume=False, normalizeXYZ=False):
    MAX_STEP_COUNT = 100 # Assume all experiments have fewer steps than this.
    stepData = {}
    
    files = os.listdir(rootPath)
    lastTreePath = None
    lastVolume = None
    for step in range(MAX_STEP_COUNT):
        stepPath = os.path.join(rootPath, "step_%d_EXPT.TXT" % step)
        metaPath = os.path.join(rootPath, "rscan_metadata_step_%d.txt" % step)
        treePath = os.path.join(rootPath, "interp-neuron-step-%d.txt" % (step - 1))
        xyzsPath = os.path.join(rootPath, "node_xyz_step_%d.txt" % step)
        kymoPath = os.path.join(rootPath, "kymograph_step_%d.txt" % step)
        volumeXYZSource = None
        if not os.path.isfile(stepPath) or not os.path.isfile(metaPath):
            continue
        if treePath is None or not os.path.isfile(treePath):
            treePath = lastTreePath
        if treePath is None or not os.path.isfile(treePath):
            treePath = os.path.join(rootPath, "interp-neuron-.txt")
        if treePath is None or not os.path.isfile(treePath):
            continue
        if not os.path.isfile(xyzsPath):
            xyzsPath = None
            # Load nodes and xyz from previous volume if no file with all xyz exists:
            volumeXYZSource = lastVolume if getPlanarXYZFromVolume else None
        if not loadKymoData or not os.path.isfile(kymoPath):
            kymoPath = None
        stepData[step] = loadSingleStep(stepPath, metaPath, treePath, xyzsPath, kymoPath, volumeXYZSource, convertXYZtoPx, normalizeXYZ)
        lastTreePath = treePath
        if not stepData[step]['planar']:
            lastVolume = stepData[step]
    return stepData
    
# Given a list of values, and a target find the index of the value closest to it.
def _closestIdx(target, values):
    bestIdx = 0
    for i in range(1, len(values)):
        if np.abs(values[i] - target) < np.abs(values[bestIdx] - target):
            bestIdx = i
    return bestIdx

# Given XYZ in world location, and XY->px plus Z px locations, convert into pixel XYZ
def _worldToPixelXYZ(xyz, xySizeM, zStackLocations):
    result = np.copy(xyz)
    for r in range(xyz.shape[0]):
        xM, yM, zM = xyz[r]
        xPx = xM / xySizeM
        yPx = yM / xySizeM
        zPx = _closestIdx(zM, zStackLocations)
        result[r] = (xPx, yPx, zPx)
    return result
    
# Given lines and the current offset, verify the lines are the correct type, and return the values
def _treeVerify(at, lines, lineTypes):
    n = len(lineTypes)
    data = []
    for i in range(n):
        assert lines[at + i][0] == lineTypes[i]
        data.append(lines[at + i][1])
    return at + n, tuple(data)
