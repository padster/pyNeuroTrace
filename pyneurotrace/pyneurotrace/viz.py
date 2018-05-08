import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

PAD = 0.05

# TODO: Move into a reusable place:
LINE_COLOR_COUNT = 7
LINE_COLORS = plt.get_cmap('hsv')(np.arange(0.0, 1.0, 1.0/LINE_COLOR_COUNT))[:, :3]

def _plotIntensityOnto(ax, data):
    # TODO - set limits for min/max values
    if len(data.shape) != 2:
        print ("Intensity plot must be 2D, nodes x samples")
        return
    ax.imshow(data, cmap='hot', interpolation='nearest', aspect='auto', origin='lower')

def _plotLineOnto(ax, data, labels, colors):
    ax.patch.set_facecolor('black')
    if isinstance(data, list):
        assert isinstance(labels, list) and isinstance(colors, list)
        assert len(data) == len(labels) and len(data) == len(colors)
        for d, l, c in zip(data, labels, colors):
            ax.plot(d.T, label=l, c=c, linewidth=0.5)
            nSamples = d.shape[0]
        ax.legend()
    else:
        ax.plot(data.T)
        nSamples = data.shape[1]
    ax.set_xlim((0, nSamples))

def _plotBranchesOnto(ax, branches, yLim):
    branchRGB = LINE_COLORS[(np.array(branches)) % LINE_COLOR_COUNT]
    branchRGB = np.expand_dims(branchRGB, axis=1)
    ax.imshow(branchRGB, interpolation='nearest', aspect='auto', origin='lower')

def _plotStimOnto(ax, stim, xLim):
    ax.set_xlim(xLim)
    ax.patch.set_facecolor('black')
    for stimEnd in stim[:, 1]:
        ax.axvline(x=stimEnd, c='r')
    for stimStart in stim[:, 0]:
        ax.axvline(x=stimStart, c='y')

def plotIntensity(data, hz, branches=None, stim=None, title=None):
    fig, aBranches, aData, aStim, aBlank = None, None, None, None, None
    if branches is None and stim is None:
        fig, (aData) = plt.subplots(1, 1)
    elif branches is not None and stim is None:
        fig, (aBranches, aData) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1, 20]})
    elif branches is None and stim is not None:
        fig, (aData, aStim) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[8, 1]})
    else:
        fig, ((aBranches, aData), (aBlank, aStim)) = plt.subplots(2, 2, gridspec_kw = {'height_ratios':[8, 1], 'width_ratios':[1, 20]})
    fig.suptitle(title)
    fig.subplots_adjust(left=PAD, right=(1 - PAD), top=(1 - PAD), bottom=PAD)

    _plotIntensityOnto(aData, data)
    if aBranches is not None:
        aData.get_yaxis().set_visible(False)
        aBranches.get_xaxis().set_visible(False)
        fig.subplots_adjust(wspace=0.0)
        _plotBranchesOnto(aBranches, branches, yLim=aData.get_ylim())
    if aStim is not None:
        aData.get_xaxis().set_visible(False)
        aStim.get_yaxis().set_visible(False)
        aStim.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
        fig.subplots_adjust(hspace=0.0)
        _plotStimOnto(aStim, stim, xLim=aData.get_xlim())
    else:
        aData.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))

    if aBlank is not None:
        aBlank.get_xaxis().set_visible(False)
        aBlank.get_yaxis().set_visible(False)

def plotLine(data, hz, stim=None, labels=None, colors=None, title=None):
    fig, aData, aStim = None, None, None
    if stim is None:
        fig, (aData) = plt.subplots(1, 1) # gridspec_kw = {'width_ratios':[3, 1]})
    else:
        fig, (aData, aStim) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[8, 1]})
    fig.suptitle(title)
    fig.subplots_adjust(left=PAD, right=(1 - PAD), top=(1 - PAD), bottom=PAD)

    _plotLineOnto(aData, data, labels, colors)
    if aStim is not None:
        aData.get_xaxis().set_visible(False)
        aStim.get_yaxis().set_visible(False)
        aStim.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
        fig.subplots_adjust(hspace=0.0)
        _plotStimOnto(aStim, stim, xLim=aData.get_xlim())
    else:
        aData.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))

def debugPlotPlanar(nodeXYZ, branchIDs):
    fig, ax = plt.subplots(1, 1)
    ax.patch.set_facecolor('black')
    for branch in range(np.max(branchIDs) + 1):
        x = nodeXYZ[branchIDs == branch, 0] * 10000
        y = nodeXYZ[branchIDs == branch, 1] * 10000
        ax.scatter(x, y, color=LINE_COLORS[branch % LINE_COLOR_COUNT])

# Debug helper to print tree strucuture to commandline:
def printTree(nodeAt, nodes, indent=''):
    print (indent + str(nodeAt))
    for child in nodes[nodeAt]['children']:
        printTree(child['id'], nodes, indent + ' ')
