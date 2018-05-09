import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import matplotlib.patches as patches
from matplotlib import collections as mc
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


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

def _plotLineOnto(ax, data, labels, colors, split):
    ax.patch.set_facecolor('black')
    if isinstance(data, list):
        assert isinstance(labels, list) and isinstance(colors, list)
        assert len(data) == len(labels) and len(data) == len(colors)
        perLineOffset = np.max(data) - np.min(data) if split else 0.0
        for i, (d, l, c) in enumerate(zip(data, labels, colors)):
            ax.plot(d.T + i * perLineOffset, label=l, c=c, linewidth=0.5)
            nSamples = d.shape[0]
        ax.legend()
    else:
        perLineOffset = np.max(data) - np.min(data) if split else 0.0
        for i in range(data.shape[0]):
            data[i] += i * perLineOffset
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
        aStim.set_xticks(stim[:, 0])
        fig.subplots_adjust(hspace=0.0)
        _plotStimOnto(aStim, stim, xLim=aData.get_xlim())
    else:
        aData.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))

    if aBlank is not None:
        aBlank.get_xaxis().set_visible(False)
        aBlank.get_yaxis().set_visible(False)

def plotLine(data, hz, stim=None, labels=None, colors=None, title=None, split=True):
    fig, aData, aStim = None, None, None
    if stim is None:
        fig, (aData) = plt.subplots(1, 1) # gridspec_kw = {'width_ratios':[3, 1]})
    else:
        fig, (aData, aStim) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[8, 1]})
    fig.suptitle(title)
    fig.subplots_adjust(left=PAD, right=(1 - PAD), top=(1 - PAD), bottom=PAD)

    _plotLineOnto(aData, data, labels, colors, split)
    if aStim is not None:
        aData.get_xaxis().set_visible(False)
        aStim.get_yaxis().set_visible(False)
        aStim.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
        aStim.set_xticks(stim[:, 0])
        fig.subplots_adjust(hspace=0.0)
        _plotStimOnto(aStim, stim, xLim=aData.get_xlim())
    else:
        aData.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
    if split:
        aData.get_yaxis().set_visible(False)

def debugPlotPlanar(nodeXYZ, branchIDs):
    fig, ax = plt.subplots(1, 1)
    ax.patch.set_facecolor('black')
    for branch in range(np.max(branchIDs) + 1):
        x = nodeXYZ[branchIDs == branch, 0] * 10000
        y = nodeXYZ[branchIDs == branch, 1] * 10000
        ax.scatter(x, y, color=LINE_COLORS[branch % LINE_COLOR_COUNT])


CIRCLES_HACK = []
def planarAnimation(nodeXYZ, traceData, hz):
    DOWNSAMPLE = 5
    hz = hz // DOWNSAMPLE
    traceData = traceData[:, ::DOWNSAMPLE]
    # lineCollection = mc.LineCollection(lines, colors=[(0,0,0,1)], linewidths=1)
    fig, ax = plt.subplots(figsize=(13, 6))
    # ax.add_collection(lineCollection)
    # ax.autoscale()
    # ax.margins(0.1)


    nNodes = nodeXYZ.shape[0]
    assert nNodes == traceData.shape[0]
    xys = [(nodeXYZ[i, 0] * 10000, nodeXYZ[i, 1] * 10000) for i in range(nNodes)]
    print (xys)
    traceData = traceData.clip(min=0)
    traceData = traceData / np.max(traceData)

    PAD = 0.02
    xs, ys = zip(*xys)
    ax.set_xlim(np.min(xs) - PAD, np.max(xs) + PAD)
    ax.set_ylim(np.min(ys) - PAD, np.max(ys) + PAD)

    hotMap = plt.get_cmap('hot')

    def _frameCircles(frameValues):
        circles = []
        # print (frameValues)
        for i in range(nNodes):
            v = frameValues[i]
            patch = patches.Circle(xys[i], radius=0.005, color=hotMap(v))
            circles.append(patch)
        return circles

    def _animFrame(i):
        global CIRCLES_HACK
        if (i % 200 == 0):
            print ("%d/%d" % (i, traceData.shape[1]))
        for circle in CIRCLES_HACK:
            circle.remove()
        CIRCLES_HACK = _frameCircles(traceData[:, i])
        for circle in CIRCLES_HACK:
            ax.add_patch(circle)
        ax.set_xlabel("%.2fs / %.2fs" % (i / hz, traceData.shape[1] / hz))
        return tuple()

    anim = FuncAnimation(
        fig,
        _animFrame,
        frames=traceData.shape[1],
        interval=1,
        blit=True,
        repeat=False
    )
    print ("Saving...")
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=hz, metadata=dict(artist='Me'), bitrate=1800)
    anim.save("fire.mp4", writer=writer)
    # anim.save("fire.gif", dpi=80, writer='imagemagick')
    print ("Saved!")
    # plt.show()
    return None

# Debug helper to print tree strucuture to commandline:
def printTree(nodeAt, nodes, indent=''):
    print (indent + str(nodeAt))
    for child in nodes[nodeAt]['children']:
        printTree(child['id'], nodes, indent + ' ')
