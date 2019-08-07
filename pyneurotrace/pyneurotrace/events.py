"""
Tools for detecting and analysing transient events within traces.
"""

import numpy as np

from .filters import _forEachTimeseries

"""
EWMA:  Exponentially Weighted Moving Average
https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
Performs smoothing on each row, given how strongly to weight new values (vs existing average)
"""
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

# Replaces trace[i] with the mean of trace[0:i]
def _runningMean(trace):
    n = len(trace)
    result = np.zeros(n)
    cumsum = 0
    for i in range(n):
        cumsum += trace[i]
        result[i] = cumsum / (i + 1)
    return result

"""
CUSUM:  Cumulative sum of movement above the mean.
https://en.wikipedia.org/wiki/CUSUM
Subtracts the rolling average from each point, then accumulates how far it is above a slack noise value.
"""
def cusum(data, slack=1.0):
    def _singleRowCusum(trace):
        n = len(trace)
        result = np.zeros(n)
        means = _runningMean(trace)
        y = 0
        for i in range(n):
            y = max(0, trace[i] - means[i] - slack)
            result[i] = y
        return result
    return _forEachTimeseries(data, _singleRowCusum)

# Defintes the shape MF is trying to match - double exponential given rise (tA) and decay (tB) rates
def _matchedFilterShape(hz, windowSize, A, tA, tB):
    scale = A * np.power(tA, tA / (tA - tB)) * np.power(tB, tB / (tB - tA)) / (tB - tA)
    shape = np.zeros(windowSize)
    for i in range(windowSize):
        shape[i] = scale * (np.exp(-i / tB) - np.exp(-i / tA))
    return shape

"""
MF:  Matched Filter
https://en.wikipedia.org/wiki/Matched_filter
Performs a (log) likelihood ratio test, to see how well a previous window matches a desired signal over no signal.
"""
def matchedFilter(data, hz, windowSize, A=2.0, riseRate=0.028, decayRate=0.39):
    m = _matchedFilterShape(hz, windowSize, A, riseRate * hz, decayRate * hz)

    # PDF that a given value is noise, using gaussian distribution
    def _noisePDF(value, mean, std):
        z = (value - mean) / std
        return np.exp(-z * z / 2) / (std * np.sqrt(2 * np.pi))

    def _singleRowMF(trace):
        mean, std = np.mean(trace), np.std(trace)
        logPDF = lambda v: np.log(_noisePDF(v, mean, std))

        n = len(trace)
        result = np.zeros(n)
        # probs[i] = log(probability that the previous i samples match first i of the shape/noise)
        probs = np.zeros(windowSize)
        for i in range(n):
            val = trace[i]
            lpdft = logPDF(trace[i])
            for j in reversed(range(1, min(i, windowSize))): # Reversed, as new[i+1] uses old[i]
                probs[j] = probs[j - 1] + (logPDF(trace[i] - m[j]) - lpdft)
            probs[0] = logPDF(trace[i] - m[0]) - lpdft
            result[i] = probs[windowSize - 1]
        return result
    return _forEachTimeseries(data, _singleRowMF)

"""
Given an event detector, turn into event yes/no indicators by cutting at a confidence threshold
and only keeping events that happened after a minimum number of non-event samples.
"""
def thresholdEvents(data, threshold, minBelowBefore=1):
    def _singleRowThreshold(trace):
        n, numBelowBefore = len(trace), 0
        isEvent = np.zeros(n)
        for i in range(n):
            below = trace[i] < threshold
            isEvent[i] = (not below and numBelowBefore >= minBelowBefore)
            numBelowBefore = numBelowBefore + 1 if below else 0
        return isEvent
    return _forEachTimeseries(data, _singleRowThreshold)
