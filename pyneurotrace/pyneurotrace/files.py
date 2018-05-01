import numpy as np

def load2PData(path):
    print ("Loading 2P data from " + path)
    data = np.loadtxt(path)
    # With N nodes, this returns:
    # Node IDs (N), XYZ (N x 3), raw trace (N x S samples)
    return data[:, 0], data[:, 1:3], data[:, 4:]
