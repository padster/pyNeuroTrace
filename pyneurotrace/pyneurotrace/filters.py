from oasis.functions import deconvolve as oasisDeconvolve
import numpy as np

"""
Friedrich, J., Zhou, P., & Paninski, L. (2017).
Fast online deconvolution of calcium imaging data.
PLoS computational biology, 13(3), e1005423.
"""
def oasisSmooth(data):
    def _singleRowOasis(samples):
        clean, _, _, _, _ = oasisDeconvolve(samples)
        return clean
    return _forEachTimeseries(data, _singleRowOasis)

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
    t0ratio = None if t0 is None else np.exp(-1 / (t0 * hz))
    t1samples, t2samples = round(t1 * hz), round(t2*hz)

    def _singeRowDeltaFOverF(samples):
        fBar = _windowFunc(np.mean, samples, t1samples, mid=True)
        f0 = _windowFunc(np.min, fBar, t2samples)
        result = (samples - f0) / f0
        if t0ratio is not None:
            result = _ewma(result, t0ratio)
        return result
    return _forEachTimeseries(data, _singeRowDeltaFOverF)


def _windowFunc(f, x, window, mid=False):
    n = len(x)
    startOffset = (window - 1) // 2 if mid else window - 1

    result = np.zeros(x.shape)
    for i in range(n):
        startIdx = i - startOffset
        endIdx = startIdx + window
        startIdx, endIdx = max(0, startIdx), min(endIdx, n)
        result[i] = f(x[startIdx:endIdx])
    return result

def _ewma(x, ratio):
    result = np.zeros(x.shape)
    weightedSum, sumOfWeights = 0.0, 0.0
    for i in range(len(x)):
        weightedSum = ratio * weightedSum + x[i]
        sumOfWeights = ratio * sumOfWeights + 1.0
        result[i] = weightedSum / sumOfWeights
    return result

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
