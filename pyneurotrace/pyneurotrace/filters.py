import numpy as np
from oasis.functions import deconvolve as oasisDeconvolve

"""
Friedrich, J., Zhou, P., & Paninski, L. (2017).
Fast online deconvolution of calcium imaging data.
PLoS computational biology, 13(3), e1005423.
"""
def oasisSmooth(data):
    def _singleRowOasis(samples):
        clean, _, _, _, _ = oasisDeconvolve(samples)
        return clean

"""
Okada, M., Ishikawa, T., & Ikegaya, Y. (2016).
A computationally efficient filter for reducing shot noise in low S/N data.
PloS one, 11(6), e0157595.
"""
def okada(data):
    def _singleRowOkada(samples):
        x = np.copy(samples)
        for i in range(1, len(x) - 1):
            if (x[i] - x[i - 1]) * (x[i] - x[i + 1]) > 0:
                x[i] = (x[i - 1] + x[i + 1]) / 2.0
        return x
    return _forEachTimeseries(data, _singleRowOkada)

"""
Jia, H., Rochefort, N. L., Chen, X., & Konnerth, A. (2011).
In vivo two-photon imaging of sensory-evoked dendritic calcium signals in cortical neurons.
Nature protocols, 6(1), 28.
"""
def deltaFOverF0(data, hz, t0=0.2, t1=0.75, t2=3.0):
    def _singeRowDeltaFOverF(samples):
        # TODO - proper code
        f0 = np.mean(samples)
        return (samples - f0) / f0
    return _forEachTimeseries(data, _singeRowDeltaFOverF)


# Input is either 1d (timeseries), 2d (each row is a timeseries) or 3d (x, y, timeseries)
def _forEachTimeseries(data, func):
    dim = len(data.shape)
    result = np.zeros(data.shape)
    if dim == 1: # single timeseries
        result = func(data)
    elif dim == 2: # (node, timeseries)
        for i in range(data.shape[0]):
            result[i] = func(data[i])
    elif dim == 3: # (x, y, timeseries)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                result[i, j] = func(data[i, j])
    return result
