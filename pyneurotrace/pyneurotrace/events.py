import numpy as np

from filters import _forEachTimeseries

def ewma(data, weight=0.1):
    def _singleRowEWMA(trace):
        n = len(trace)
        result = np.zeros(n)
        y = 0
        for i in range(n):
            y = weight * trace[i] + (1 - weight) * y
            result[i] = y
        return result
    return _forEachTimeseries(data, _singleRowEWMA)

def runningMean(trace):
    n = len(trace)
    result = np.zeros(n)
    cumsum = 0
    for i in range(n):
        cumsum += trace[i]
        result[i] = cumsum / (i + 1)
    return result

def cusum(data, slack=1.0):
    def _singleRowCusum(trace):
        n = len(trace)
        result = np.zeros(n)
        means = runningMean(trace)
        y = 0
        for i in range(n):
            y = max(0, trace[i] - means[i] - slack)
            result[i] = y
        return result
    return _forEachTimeseries(data, _singleRowCusum)

def _matchedFilterShape(hz, windowSize, A, tA, tB):
    shape = np.zeros(windowSize)
    scale = A * np.power(tA, tA / (tA - tB)) * np.power(tB, tB / (tB - tA)) / (tB - tA)
    for i in range(windowSize):
        shape[i] = scale * (np.exp(-i / tB) - np.exp(-i / tA))
    return shape

def matchedFilter(data, hz, windowSize, A=2.0, riseRate=0.028, decayRate=0.39):
    m = _matchedFilterShape(hz, windowSize, A, riseRate * hz, decayRate * hz)

    def _noisePDF(value, mean, std):
        z = (value - mean) / std
        return np.exp(-z * z / 2) / (std * np.sqrt(2 * np.pi))

    def _singleRowMF(trace):
        mean, std = np.mean(trace), np.std(trace)
        logPDF = lambda v: np.log(_noisePDF(v, mean, std))

        n = len(trace)
        result = np.zeros(n)
        probs = np.zeros(windowSize)
        for i in range(n):
            val = trace[i]
            lpdft = logPDF(trace[i])
            for j in reversed(range(1, min(i, windowSize))):
                probs[j] = probs[j - 1] + logPDF(trace[i] - m[j]) - lpdft
            probs[0] = logPDF(trace[i] - m[0]) - lpdft
            result[i] = probs[windowSize - 1]
        return result
    return _forEachTimeseries(data, _singleRowMF)

def thresholdEvents(data, threshold, minBelowBefore=2):
    def _singleRowThreshold(trace):
        n, numBelowBefore = len(trace), 0
        isEvent = np.zeros(n)
        for i in range(n):
            below = trace[i] < threshold
            isEvent[i] = (not below and numBelowBefore >= minBelowBefore)
            numBelowBefore = numBelowBefore + 1 if below else 0
        return isEvent
    return _forEachTimeseries(data, _singleRowThreshold)
