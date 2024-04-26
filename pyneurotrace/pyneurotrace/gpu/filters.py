# from oasis.functions import deconvolve as oasisDeconvolve
import math
import cupy as cu

from cupyx.scipy.ndimage import minimum_filter1d, uniform_filter1d
from .nndFilter import nonNegativeDeconvolution as NND

"""
TODO Turn these into vectorized GPU versions 

Friedrich, J., Zhou, P., & Paninski, L. (2017).
Fast online deconvolution of calcium imaging data.
PLoS computational biology, 13(3), e1005423.

def oasisSmooth(data, iterFunc=None):
    def _singleRowOasis(samples):
        #clean, _, _, _, _ = oasisDeconvolve(samples)
        clean, events, _, gParams, _ = oasisDeconvolve(samples, optimize_g=5)
        #print (gParams)
        #clean, _, _, _, _ = oasisDeconvolve(samples, g=(None, None), optimize_g=5)
        return clean
    return _forEachTimeseries(data, _singleRowOasis, iterFunc)



Podgorski, K., & Haas, K. (2013).
Fast non‐negative temporal deconvolution for laser scanning microscopy.
Journal of biophotonics, 6(2), 153-162.

def nndSmooth(data, hz, tau, iterFunc=None):
    tauSamples = tau * hz

    # This is the transient shape we're deconvolving against:
    # e^(x/tauSamples), for 8 times the length of tau.
    cutoff = round(8 * tauSamples)
    fitted = cu.exp(-cu.arange(cutoff + 1) / tauSamples)

    def _singleRowNND(samples):
        result = cu.copy(samples)
        nanSamples = cu.isnan(samples)
        if cu.all(nanSamples):
            pass # No data
        elif not cu.any(nanSamples):
            # All samples exist, so fit in one go
            result = cu.convolve(NND(samples, tauSamples), fitted)[:len(samples)]
        else:
            # Lots of different runs of samples, fit each separately
            starts = cu.where((not nanSamples) & cu.isnan(cu.concatenate(([1], samples[:-1]))))[0]
            ends = cu.where((not nanSamples) & cu.isnan(cu.concatenate((samples[1:], [1]))))[0]
            for start, end in zip(starts, ends):
                tmp = cu.convolve(NND(samples[start:end], tauSamples), fitted)
                result[start:end] = cu.max(0, tmp[:end - start + 1])
        return result

    return _forEachTimeseries(data, _singleRowNND, iterFunc)




"""


"""
Jia, H., Rochefort, N. L., Chen, X., & Konnerth, A. (2011).
In vivo two-photon imaging of sensory-evoked dendritic calcium signals in cortical neurons.
Nature protocols, 6(1), 28.
"""

"""
Okada, M., Ishikawa, T., & Ikegaya, Y. (2016).
A computationally efficient filter for reducing shot noise in low S/N data.
PloS one, 11(6), e0157595.
"""
def okada(data, iterFunc=None):
    def _singleRowOkada(samples):
        x = cu.copy(samples)

        shiftLeft = cu.roll(x[1:], -1)
        shiftRight = cu.roll(x[1:], 1)
               
        # Find where filter should be applied
        filterCondition = (x[1:] - shiftLeft) * (x[1:] - shiftRight) > 0

        # Replace values with average
        x[1:] = cu.where(filterCondition, (shiftLeft + shiftRight)/2, x[1:])
        
        return x
    return _forEachTimeseries(data, _singleRowOkada, iterFunc)


def deltaFOverF0(data, hz, t0=0.2, t1=0.75, t2=3.0, iterFunc=None): 
    t1samples, t2samples = round(t1 * hz), round(t2*hz)
    alpha = None if t0 is None else 1 - cu.exp(-1 / (t0 * hz))

    def _singeRowDeltaFOverF(samples):
        fBar = uniform_filter1d(samples, t1samples, mode='nearest')
        startOffset = t2samples//2-1
        f0   = minimum_filter1d(fBar, t2samples, mode='nearest', origin=startOffset)

        result = (samples - f0) / f0
        if alpha is not None:
            result = _ewma(result, alpha)
        return result
    return _forEachTimeseries(data, _singeRowDeltaFOverF, iterFunc)


def _ewma(data, alpha):
    # Vectorized approximation of EWMA function
    # Generate weights    
    window_size = int(-cu.log(1e-10) / alpha)
    kernel = alpha * (1 - alpha) ** cu.arange(window_size)
    kernel /= cu.sum(kernel)
    
    #Peform convolution
    convoluted = cu.convolve(data, kernel, mode='full')
    cumulative_kernel_sum = cu.cumsum(kernel)
    normalizer = cu.zeros_like(convoluted)
    end_idx = len(cumulative_kernel_sum)

    # Place the cumulative sum correctly in the normalizer
    normalizer[:end_idx] = cumulative_kernel_sum
    normalizer[end_idx:] = cumulative_kernel_sum[-1]

    normalized_result = convoluted[:len(data)] / normalizer[:len(data)]
    
    return normalized_result

# Input is either 1d (timeseries), 2d (each row is a timeseries) or 3d (x, y, timeseries)
def _forEachTimeseries(data, func, iterFunc=None):
    data = cu.array(data)

    if iterFunc is None:
        iterFunc = lambda x: x
    dim = len(data.shape)
    result = cu.zeros(data.shape)
    if dim == 1: # single timeseries
        result = func(data)
    elif dim == 2: # (node, timeseries)
        for i in iterFunc(range(data.shape[0])):
            result[i] = func(data[i])
    elif dim == 3: # (x, y, timeseries)
        for i in iterFunc(range(data.shape[0])):
            for j in iterFunc(range(data.shape[1])):
                result[i, j] = func(data[i, j])
    return result.get()
