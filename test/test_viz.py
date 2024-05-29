import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from pyneurotrace.viz import plotIntensity, plotLine, plotAveragePostStimIntensity, plotAveragePostStimTransientParams, plotPlanarStructure, plotBaseTipScatter

def test_plotIntensity():
    data = np.random.random((10, 100))
    hz = 10
    fig, ax = plt.subplots()
    
    # Run the function to ensure it generates the plot without errors
    xAx, yAx = plotIntensity(data, hz)
    
    # Assert that axes were returned
    assert xAx is not None
    assert yAx is not None
    plt.close(fig)

def test_plotLine():
    data = np.random.random((5, 100))
    hz = 10
    fig, ax = plt.subplots()
    
    # Run the function to ensure it generates the plot without errors
    xAx, yAx = plotLine(data, hz)
    
    # Assert that axes were returned
    assert xAx is not None
    assert yAx is not None
    plt.close(fig)

def test_plotAveragePostStimIntensity():
    script_dir = os.path.dirname(__file__)
    file_name = "test_array.npy" 
    file_path = os.path.join(script_dir, file_name)
    data  = np.load(file_path)
    
    # Parameters
    num_signals = 10
    signal_length = 1000
    tau = 20  # Time constant for the decay
    hz = 10
    stimOffIdx = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])
    stimOnIdx = np.array([25, 35, 45, 55, 65, 75, 85, 95, 105])



    fig, ax = plt.subplots()
    
    # Run the function to ensure it generates the plot without errors
    plotAveragePostStimIntensity(data, hz, stimOffIdx, stimOnIdx)
    plt.close(fig)

def test_plotAveragePostStimTransientParams():
    script_dir = os.path.dirname(__file__)
    file_name = "test_array.npy" 
    file_path = os.path.join(script_dir, file_name)
    data  = np.load(file_path)
    hz = 10
    stimOffsets = np.array([2, 4, 6, 6, 8])
    secAfter = 3
    fig, ax = plt.subplots()
    
    # Run the function to ensure it generates the plot without errors
    plotAveragePostStimTransientParams(data, hz, stimOffsets, secAfter)
    plt.close(fig)

def test_plotPlanarStructure():
    tree = {
        0: {'location': np.array([0, 0, 0]), 'children': [{'id': 1}, {'id': 2}]},
        1: {'location': np.array([1, 1, 0]), 'children': []},
        2: {'location': np.array([-1, 1, 0]), 'children': []}
    }
    nodeXYZ = np.array([[0, 0, 0], [1, 1, 0], [-1, 1, 0]])
    rootID = 0
    branchIDs = [0, 1, 2]
    fig, ax = plt.subplots()
    
    # Run the function to ensure it generates the plot without errors
    plotPlanarStructure(tree, rootID, nodeXYZ, branchIDs)
    plt.close(fig)

def test_plotBaseTipScatter():
    baseTrace = np.random.random(100)
    tipTrace = np.random.random(100)
    fig, ax = plt.subplots()
    
    # Run the function to ensure it generates the plot without errors
    plotBaseTipScatter(baseTrace, tipTrace)
    plt.close(fig)

if __name__ == '__main__':
    pytest.main()
