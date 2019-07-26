# from oasis.functions import deconvolve as oasisDeconvolve
import math
import numpy as np

"""
Friedrich, J., Zhou, P., & Paninski, L. (2017).
Fast online deconvolution of calcium imaging data.
PLoS computational biology, 13(3), e1005423.
"""
def oasisSmooth(data, iterFunc=None):
    def _singleRowOasis(samples):
        #clean, _, _, _, _ = oasisDeconvolve(samples)
        clean, events, _, gParams, _ = oasisDeconvolve(samples, optimize_g=5)
        #print (gParams)
        #clean, _, _, _, _ = oasisDeconvolve(samples, g=(None, None), optimize_g=5)
        return clean
    return _forEachTimeseries(data, _singleRowOasis, iterFunc)

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
    
    
"""
TODO : Clean up
"""
# performs fast nonnegative deconvolution on pmt signal to solve for minimum MSE photon rate
#   trace  :   The data to be deconvolved
#   tau    :   The time constant of the PMT, in data samples
#   return :   estimated photon rate
def NN_KP(trace, tau):
    T = len(trace)
    counts = np.zeros(T)
    counts[-1] = trace[-1]
    cutoff = math.ceil(8 * tau)
    kernel = np.exp(-np.arange(cutoff + 1)/tau) # convolution kernel
    recent = np.full(1 + round(T / 2), np.nan).astype(int)
    recent[0] = T #stored locations where we assigned counts
    recentIdx = 0
   
    # the points that could potentially be assigned counts:
    _delayed = np.concatenate(([0], trace[:-2]))
    points = (trace[:-1] > kernel[1] * _delayed) & (trace[:-1] > 0)
    
    # dividing these points up into runs, for speed
    runStarts = np.where(points & ~(np.concatenate(([False], points[:-1]))))[0].astype(int)
    runEnds = np.where(points & ~(np.concatenate((points[1:], [False]))))[0].astype(int)
    runIdx = len(runEnds) - 1
    
    while runIdx >= 0:
        oldTop, oldBottom = 0, 0
        t = runEnds[runIdx]
        t1 = t
        accum = 0
        
        converged = False
        while not converged:
            if recentIdx >= 0 and recent[recentIdx] < (t+cutoff):
                t2 = recent[recentIdx] - 1
                C_max = counts[t2] / kernel[t2-t]
            else:
                t2 = min(t + cutoff, T+1) - 1
                C_max = np.inf
            

            b = kernel[t1-t:t2-t]
            top = np.dot(b, trace[t1:t2]) + oldTop #this is the numerator of the least squares fit for an exponential
            bottom = np.dot(b, b) + oldBottom #this is the denominator of the fit
            
            done = False
            while not done:
                #the error function is (data-kernel.*C)^2
                bestC = max(top/bottom, 0);  #C=top/bottom sets the derivative of the error to 0

                # does not meet nonnegative constraint. Continue to adjust previous solutions.
                if bestC > (C_max+accum): 
                    accum = accum + counts[t2] / kernel[t2-t]
                    counts[t2] = 0
                    t1 = t2
                    oldTop = top
                    oldBottom = bottom
                    recentIdx -= 1
                    done = True
                    
                else: # converged!
                    #now that we have found the MSE counts for times t<end, check if
                    #this will be swamped by the next timepoint in the run
                    if  (t == runStarts[runIdx]) or (trace[t-1] < bestC/kernel[1]): #%C_max won't necessarily get swamped
                        if recentIdx >= 0 and t2 <= t + cutoff:
                            counts[t2] = counts[t2] - (bestC - accum) * kernel[t2-t]
                        runStart = runStarts[runIdx]
                        initIdx = recentIdx + 1
                        recentIdx = recentIdx + 1 + t - runStart;
                        
                        _skipped = 0
                        if recentIdx + 1 > len(recent):
                            _skipped = recentIdx - (len(recent) - 1)
                            recentIdx = len(recent) - 1
                            
                            
                        recent[initIdx:recentIdx + 1] = np.arange(t+1, runStart + _skipped, -1)
                        counts[runStart:(t+1)] = \
                               np.concatenate((trace[runStart:t], [bestC])) - \
                               np.concatenate(([0], kernel[1]*trace[runStart:t]))
                        done = True
                        converged = True
                    else: #%C_max will get swamped
                        #%in this situation, we know that this point will be removed
                        #%as we continue to process the run. To save time:
                        t -= 1
                        runEnds[runIdx] = t
                        accum = accum / kernel[1]
                        top = top * kernel[1] + trace[t] #% %this is the correct adjustment to the derivative term above
                        bottom = bottom * (kernel[1] ** 2) + 1 #% %this is the correct adjustment to the derivative term above

        runIdx -= 1
    return counts

def filterNND(traces, tau):
    result = np.copy(traces)
    
    cutoff = round(8 * tau)
    fitted = np.exp(-np.arange(cutoff + 1)/tau)
    
    nRows, nSamples = traces.shape
    for i in range(nRows):
        trace = traces[i]
        traceIsNan = np.isnan(trace)
        
        if np.all(traceIsNan):
            continue
            
        if not np.any(traceIsNan):
            result[i] = np.convolve(NN_KP(trace, tau), fitted)[:nSamples]
        else:
            starts = np.where((not np.isnan(trace)) & np.isnan(np.concatenate(([1], trace[:-1]))))[0]
            ends = np.where((not np.isnan(trace)) & np.isnan(np.concatenate((trace[1:], [1]))))[0]
            for start, end in zip(starts, ends):
                tmp = np.convolve(NN_KP(trace[start:end], tau), fitted)
                result[i, start:end] = np.max(0, tmp[:end - start + 1])
    return result


def kpNNDSmooth(data, hz, tau, iterFunc=None):
    tauSamples = tau * hz
    return filterNND(data, tauSamples)
    
