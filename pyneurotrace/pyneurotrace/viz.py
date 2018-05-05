import matplotlib.pyplot as plt

PAD = 0.05

def _plotIntensityOnto(ax, data):
    # TODO - set limits for min/max values
    if len(data.shape) != 2:
        print ("Intensity plot must be 2D, nodes x samples")
        return
    ax.imshow(data, cmap='hot', interpolation='nearest', aspect='auto')

def _plotLineOnto(ax, data):
    ax.plot(data.T)

def _plotStimOnto(ax, stim, xLim):
    ax.set_xlim(xLim)
    for stimEnd in stim[:, 1]:
        ax.axvline(x=stimEnd, c='r')
    for stimStart in stim[:, 0]:
        ax.axvline(x=stimStart, c='y')

def plotIntensity(data, stim = None):
    fig, aData, aStim = None, None, None
    if stim is None:
        fig, (aData) = plt.subplots(1, 1) # gridspec_kw = {'width_ratios':[3, 1]})
    else:
        fig, (aData, aStim) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[8, 1]})
    fig.subplots_adjust(left=PAD, right=(1 - PAD), top=(1 - PAD), bottom=PAD)

    _plotIntensityOnto(aData, data)
    if aStim is not None:
        aData.get_xaxis().set_visible(False)
        aStim.get_yaxis().set_visible(False)
        fig.subplots_adjust(hspace=0.0)
        _plotStimOnto(aStim, stim, xLim=aData.get_xlim())



def plotLine(data, stim = None):
    fig, aData, aStim = None, None, None
    if stim is None:
        fig, (aData) = plt.subplots(1, 1) # gridspec_kw = {'width_ratios':[3, 1]})
    else:
        fig, (aData, aStim) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[8, 1]})
    fig.subplots_adjust(left=PAD, right=(1 - PAD), top=(1 - PAD), bottom=PAD)

    _plotLineOnto(aData, data)
    if aStim is not None:
        aData.get_xaxis().set_visible(False)
        aStim.get_yaxis().set_visible(False)
        fig.subplots_adjust(hspace=0.0)
        _plotStimOnto(aStim, stim, xLim=aData.get_xlim())
