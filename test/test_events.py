import pytest
import numpy as np
from pyneurotrace.events import ewma, cusum, matchedFilter, thresholdEvents

def test_ewma():
    data = np.random.random((5, 150))
    weight = 0.1

    result = ewma(data, weight)

    assert result.shape == data.shape
    assert isinstance(result, np.ndarray)

    assert not np.isnan(result).any()
    assert not np.allclose(data, result)

def test_cusum():
    data = np.random.random((5, 150))
    slack = 1.0

    result = cusum(data, slack)

    assert result.shape == data.shape
    assert isinstance(result, np.ndarray)
    
    assert not np.isnan(result).any()
    assert not np.allclose(data, result) 

def test_matchedFilter():
    data = np.random.random((5, 150))
    hz = 5
    windowSize = 50

    result = matchedFilter(data, hz, windowSize)

    assert result.shape == data.shape
    assert isinstance(result, np.ndarray)
    
    assert not np.isnan(result).any()
    assert not np.allclose(data, result)

def test_thresholdEvents():
    data = np.random.random((5, 150))
    threshold = 0.5
    minBelowBefore = 1

    result = thresholdEvents(data, threshold, minBelowBefore)

    assert result.shape == data.shape
    assert isinstance(result, np.ndarray)
    
    assert np.isin(result, [0, 1]).all()  
    assert np.sum(result) > 0  
    
if __name__ == '__main__':
    pytest.main()