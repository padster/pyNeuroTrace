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
    def _singleRowEWMA(data):
        # Vectorized approximation of EWMA function
        # Generate weights    
        window_size = int(-cu.log(1e-10) / weight)
        kernel = weight * (1 - weight) ** cu.arange(window_size)
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
    return _forEachTimeseries(data, _singleRowEWMA)

# Replaces trace[i] with the mean of trace[0:i]
def _runningMean(trace):
    index  = cu.arange(1, trace.shape[0]+1)
    cumsum = cu.cumsum(trace)
    return cumsum/index

"""
CUSUM:  Cumulative sum of movement above the mean.
https://en.wikipedia.org/wiki/CUSUM
Subtracts the rolling average from each point, then accumulates how far it is above a slack noise value.
"""
def cusum(data, slack=1.0):
    def _singleRowCusum(trace):
        mean = _runningMean(trace)
        
        adjustment = trace - mean - slack
               
        return cu.maximum(0, adjustment)
    return _forEachTimeseries(data, _singleRowCusum)
    

"""
Given an event detector, turn into event yes/no indicators by cutting at a confidence threshold
and only keeping events that happened after a minimum number of non-event samples.
"""
def thresholdEvents(data, threshold, minBelowBefore=1):
        def _singleRowThreshold(trace):
            n = trace.size
            below_threshold = trace < threshold

            cumulative_below = cu.cumsum(below_threshold) - cu.cumsum(below_threshold) * ~below_threshold
            cumulative_below = cu.pad(cumulative_below, (1, 0), mode='constant', constant_values=0)[:-1]

            # Detect crossings from below to above threshold
            valid_crossings = cumulative_below >= minBelowBefore
            isEvent = cu.logical_and(crosses_threshold, valid_crossings)

            return isEvent.astype(cu.int32)
    return _forEachTimeseries(data, _singleRowThreshold)
