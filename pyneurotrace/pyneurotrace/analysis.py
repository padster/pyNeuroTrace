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
