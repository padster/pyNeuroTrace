import pytest
import numpy as np
from pyneurotrace.filters import nndSmooth, okada, deltaFOverF0

def test_nndSmooth():
    data = np.random.random(150)
    hz = 5
    tau = 0.5

    result = nndSmooth(data, hz, tau)

    
    assert isinstance(result, np.ndarray)
    assert result.shape == data.shape
    assert not np.isnan(result).any()
    assert not np.allclose(data, result)

def test_okada():
    data = np.random.random(150)
    
    
    result = okada(data)
    

    assert result.shape == data.shape
    assert not np.isnan(result).any()
    assert not np.allclose(data, result)


def test_deltaFOverF0():
    data = np.random.random(150)
    hz = 5

    result = deltaFOverF0(data, hz)

    # Assert the shape is the same
    assert result.shape == data.shape
    assert isinstance(result, np.ndarray)
    assert not np.isnan(result).any()
    assert not np.allclose(data, result)

if __name__ == '__main__':
    pytest.main()