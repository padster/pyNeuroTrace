import math
import cupy as cu
from cupyx.scipy.ndimage import minimum_filter1d, uniform_filter1d



def okada(data, iterFunc=None):
    """
    A computationally efficient filter for reducing shot noise in low S/N data.

    Okada, M., Ishikawa, T., & Ikegaya, Y. (2016).
    A computationally efficient filter for reducing shot noise in low S/N data.
    PloS one, 11(6), e0157595.

    Parameters
    ----------
    data : cupy array
        Data array to be filtered.
    iterFunc : function, optional
        Optional iteration function. Default is `None`.

    Returns
    -------
    filtered_data : cupy array
        Filtered data array.
    """

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
    """
    Calculates the change in fluorescence over baseline fluorescence. Optionally smoothed with an EWMA.
    
    Jia, H., Rochefort, N. L., Chen, X., & Konnerth, A. (2011).
    In vivo two-photon imaging of sensory-evoked dendritic calcium signals in cortical neurons.
    Nature protocols, 6(1), 28.
    
    Parameters
    ----------
    data : cupy array
        Data array to be analyzed.
    hz : int
        Sampling rate in Hz.
    t0 : float, optional
        Time constant for exponential moving average. Default is `0.2`.
    t1 : float, optional
        Time window for calculating the mean baseline. Default is `0.75`.
    t2 : float, optional
        Time window for calculating the minimum baseline. Default is `3.0`.
    iterFunc : function, optional
        Optional iteration function. Default is `None`.

    Returns
    -------
    deltaF_over_F0 : cupy array
        Calculated change in fluorescence over baseline fluorescence.
    """

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


__all__ = ['okada', 'deltaFOverF0', '_forEachTimeseries'] 
