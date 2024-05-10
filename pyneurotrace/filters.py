import math
import numpy as np

from .nndFilter import nonNegativeDeconvolution as NND

"""
Podgorski, K., & Haas, K. (2013).
Fast non‐negative temporal deconvolution for laser scanning microscopy.
Journal of biophotonics, 6(2), 153-162.
"""
def nndSmooth(data, hz, tau, iterFunc=None):
    tauSamples = tau * hz

    # This is the transient shape we're deconvolving against:
    # e^(x/tauSamples), for 8 times the length of tau.
    cutoff = round(8 * tauSamples)
    fitted = np.exp(-np.arange(cutoff + 1) / tauSamples)

    def _singleRowNND(samples):
        result = np.copy(samples)
        nanSamples = np.isnan(samples)
        if np.all(nanSamples):
            pass # No data
        elif not np.any(nanSamples):
            # All samples exist, so fit in one go
            result = np.convolve(NND(samples, tauSamples), fitted)[:len(samples)]
        else:
            # Lots of different runs of samples, fit each separately
            starts = np.where((not nanSamples) & np.isnan(np.concatenate(([1], samples[:-1]))))[0]
            ends = np.where((not nanSamples) & np.isnan(np.concatenate((samples[1:], [1]))))[0]
            for start, end in zip(starts, ends):
                tmp = np.convolve(NND(samples[start:end], tauSamples), fitted)
                result[start:end] = np.max(0, tmp[:end - start + 1])
        return result

    return _forEachTimeseries(data, _singleRowNND, iterFunc)


"""
Okada, M., Ishikawa, T., & Ikegaya, Y. (2016).
A computationally efficient filter for reducing shot noise in low S/N data.
PloS one, 11(6), e0157595.
"""
def okada(data, iterFunc=None):
    def _singleRowOkada(samples):
        x = np.copy(samples)
        for i in range(1, len(x) - 1):
            if (x[i] - x[i - 1]) * (x[i] - x[i + 1]) > 0:
                x[i] = (x[i - 1] + x[i + 1]) / 2.0
        return x
    return _forEachTimeseries(data, _singleRowOkada, iterFunc)

"""
Jia, H., Rochefort, N. L., Chen, X., & Konnerth, A. (2011).
In vivo two-photon imaging of sensory-evoked dendritic calcium signals in cortical neurons.
Nature protocols, 6(1), 28.
"""
def deltaFOverF0(data, hz, t0=0.2, t1=0.75, t2=3.0, iterFunc=None):
    t0ratio = None if t0 is None else np.exp(-1 / (t0 * hz))
    t1samples, t2samples = round(t1 * hz), round(t2*hz)

    def _singeRowDeltaFOverF(samples):
        fBar = _windowFunc(np.mean, samples, t1samples, mid=True)
        f0 = _windowFunc(np.min, fBar, t2samples)
        result = (samples - f0) / f0
        if t0ratio is not None:
            result = _ewma(result, t0ratio)
        return result
    return _forEachTimeseries(data, _singeRowDeltaFOverF, iterFunc)


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
def _forEachTimeseries(data, func, iterFunc=None):
    if iterFunc is None:
        iterFunc = lambda x: x
    dim = len(data.shape)
    result = np.zeros(data.shape)
    if dim == 1: # single timeseries
        result = func(data)
    elif dim == 2: # (node, timeseries)
        for i in iterFunc(range(data.shape[0])):
            result[i] = func(data[i])
    elif dim == 3: # (x, y, timeseries)
        for i in iterFunc(range(data.shape[0])):
            for j in iterFunc(range(data.shape[1])):
                result[i, j] = func(data[i, j])
    return result
