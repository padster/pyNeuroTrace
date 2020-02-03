import math
import numpy as np
from scipy.optimize import curve_fit

def epochs(traces, hz, startSamples, secBefore, secAfter):
    samplesBefore, samplesAfter = int(math.ceil(hz * secBefore)), int(math.ceil(hz * secAfter))
    epochs = []
    for sample in startSamples:
        epochs.append(traces[:, sample - samplesBefore : sample + samplesAfter])
    return np.array(epochs)

def epochAverage(traces, hz, startSamples, secBefore, secAfter):
    return np.mean(epochs(traces, hz, startSamples, secBefore, secAfter), axis=0)

def fitDoubleExp(y, hz, tAGuess=0.1, tBGuess=0.4, method='trf'):
    n = len(y)
    aGuess = np.max(y)
    x = np.arange(n)

    def f(x, A, t0, tA, tB):
        y = np.zeros(n)
        if not (0 < tA and tA < tB - 1e-8 and tB < 10000):
            return np.ones(n) * 1000
        scale = A * np.power(tA, tA / (tA - tB)) * np.power(tB, tB / (tB - tA)) / (tB - tA)
        for i in range(n):
            if i > t0:
                xi = i - t0
                y[i] = scale * (np.exp(-xi / tB) - np.exp(-xi / tA))
        return y

    popt, _ = curve_fit(f, x, y, p0=(aGuess, 0, tAGuess * hz, tBGuess * hz), method=method)
    return popt, f(x, *popt)

# HACK - remove once fixed
def HACKcorrectLowColumnsInPlace(traces, justFind=False):
    CUTOFF = 0.42
    avTrace = np.mean(traces, axis=0)
    badIdx = np.where(avTrace[1:-1] < CUTOFF * (avTrace[0:-2] + avTrace[2:]))[0] + 1
    if (len(badIdx) > 0):
        print ("BAD NODES")
        print (badIdx)
        print ("DELTA:")
        print (badIdx[1:] - badIdx[:-1])
    if not justFind:
        traces[:, badIdx] = 0.5 * (traces[:, badIdx - 1] + traces[:, badIdx + 1])
