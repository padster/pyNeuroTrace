import math
import numpy as np

def epochs(traces, hz, startSamples, secBefore, secAfter):
    samplesBefore, samplesAfter = int(math.ceil(hz * secBefore)), int(math.floor(hz * secAfter))
    epochs = []
    for sample in startSamples:
        epochs.append(traces[:, sample - samplesBefore : sample + samplesAfter+1])
    return np.array(epochs)

def epochAverage(traces, hz, startSamples, secBefore, secAfter):
    return np.mean(epochs(traces, hz, startSamples, secBefore, secAfter), axis=0)


# HACK - remove once fixed
def HACKcorrectLowColumnsInPlace(traces):
    CUTOFF = 0.42
    avTrace = np.mean(traces, axis=0)
    badIdx = np.where(avTrace[0, 1:-1] < CUTOFF * (avTrace[0, 0:-2] + avTrace[0, 2:]))[0] + 1
    traces[:, badIdx] = 0.5 * (traces[:, badIdx - 1] + traces[:, badIdx + 1])
