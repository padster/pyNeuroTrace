import numpy as np

from .filters import _forEachTimeseries


def ewma(data, weight=0.1):
    """
    Performs smoothing on each row using the Exponentially Weighted Moving Average (EWMA) method.

    EWMA:  Exponentially Weighted Moving Average
    https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    Performs smoothing on each row, given how strongly to weight new values (vs existing average)

    Parameters
    ----------
    data : array
        Data array to be smoothed.
    weight : float, optional
        Weight for new values versus existing average. Default is 0.1.

    Returns
    -------
    smoothed_data : array
        Smoothed data array.
    """
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


def cusum(data, slack=1.0):
    """
    Calculates the Cumulative Sum (CUSUM) of movement above the mean for each row in the data.

    CUSUM:  Cumulative sum of movement above the mean.
    https://en.wikipedia.org/wiki/CUSUM
    Subtracts the rolling average from each point, then accumulates how far it is above a slack noise value.

    Parameters
    ----------
    data : array
        Data array to be analyzed.
    slack : float, optional
        Slack noise value. Default is 1.0.

    Returns
    ----------
    cusum_data : array
        CUSUM data array.
        """
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



def matchedFilter(data, hz, windowSize, A=2.0, riseRate=0.028, decayRate=0.39):
    """
    Performs a likelihood ratio test using a Matched Filter (MF) to see how well a previous window matches a desired signal over no signal.

    MF:  Matched Filter
    https://en.wikipedia.org/wiki/Matched_filter
    Performs a (log) likelihood ratio test, to see how well a previous window matches a desired signal over no signal.

    Parameters
    ----------

    data : array
        Data array to be filtered.
    hz : int
        Sampling rate in Hz.
    windowSize : int
        Size of the window to match.
    A : float, optional
        Amplitude of the desired signal. Default is 2.0.
    riseRate : float, optional
        Rise rate of the desired signal. Default is 0.028.
    decayRate : float, optional
        Decay rate of the desired signal. Default is 0.39.

    Returns
    ----------
    mf_data : array
        Matched filter data array.
    """
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
    """
    Turns an event detector into event yes/no indicators by applying a threshold and only keeping events that occur after a minimum number of non-event samples.

    Parameters
    ----------
    data : array
        Data array containing event detector outputs.
    threshold : float
        Confidence threshold for events.
    minBelowBefore : int, optional
        Minimum number of non-event samples before an event is considered. Default is 1.

    Returns
    ----------
    events : array
       Array indicating detected events.
    """
    def _singleRowThreshold(trace):
        n, numBelowBefore = len(trace), 0
        isEvent = np.zeros(n)
        for i in range(n):
            below = trace[i] < threshold
            isEvent[i] = (not below and numBelowBefore >= minBelowBefore)
            numBelowBefore = numBelowBefore + 1 if below else 0
        return isEvent
    return _forEachTimeseries(data, _singleRowThreshold)


__all__ = ["ewma", "cusum", "thresholdEvents", "matchedFilter"]