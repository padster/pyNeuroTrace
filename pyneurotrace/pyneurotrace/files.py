import numpy as np

# Denotes the start of stimulus index lines in metadata.
STIM_INDEX_KEY = '#STIM_START_STOP_INDICES'

# Load Node IDs, positions, and raw data from an experiment txt file.
def load2PData(path):
    print ("Loading 2P data from " + path)
    data = np.loadtxt(path)
    # With N nodes, this returns:
    # Node IDs (N), XYZ (N x 3), raw trace (N x S samples)
    return data[:, 0], data[:, 1:3], data[:, 4:]

# Load stimulus [start, end] sample indices from the metadata file
def loadStimulusIndices(path):
    print ("Loading stim times from " + path)
    lines = []
    with open(path) as f:
        lines = [line.strip() for line in f.readlines()]
    if STIM_INDEX_KEY not in lines:
        return np.zeros((0, 2))

    stimIndices = []
    at = lines.index(STIM_INDEX_KEY) + 1
    while at < len(lines):
        if lines[at][0] == '#':
            break
        indices = lines[at].split('\t')
        assert len(indices) == 2
        stimIndices.append([int(index) for index in indices])
        at += 1
    return np.array(stimIndices)
