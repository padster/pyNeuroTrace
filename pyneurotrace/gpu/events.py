import cupy as cu


def ewma(data, weight=0.1):
    """
    Performs smoothing on each row using a vectorized approximation of the Exponentially Weighted Moving Average (EWMA) method.
    This implementation allows for the EWMA to be calculated on GPUs. 

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
        mean = _runningMean(trace)
        
        adjustment = trace - mean - slack
               
        return cu.maximum(0, adjustment)
    return _forEachTimeseries(data, _singleRowCusum)
    


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
        n = trace.size
        below_threshold = trace < threshold

        cumulative_below = cu.cumsum(below_threshold) - cu.cumsum(below_threshold) * ~below_threshold
        cumulative_below = cu.pad(cumulative_below, (1, 0), mode='constant', constant_values=0)[:-1]

        # Detect crossings from below to above threshold
        valid_crossings = cumulative_below >= minBelowBefore
        isEvent = cu.logical_and(crosses_threshold, valid_crossings)

        return isEvent.astype(cu.int32)
    return _forEachTimeseries(data, _singleRowThreshold)

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

__all__ = ['ewma', 'cusum', 'thresholdEvents'] 