import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm_notebook

import matplotlib.patches as patches
from matplotlib import collections as mc
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from .analysis import epochAverage, fitDoubleExp

PAD = 0.08

# TODO: Move into a reusable place:
LINE_COLOR_COUNT = 7
LINE_COLORS = plt.get_cmap('hsv')(np.arange(0.0, 1.0, 1.0/LINE_COLOR_COUNT))[:, :3]

def _plotIntensityOnto(ax, data, **kwargs):
    # TODO - set limits for min/max values
    if len(data.shape) != 2:
        print ("Intensity plot must be 2D, nodes x samples")
        return
    ax.imshow(data, cmap='hot', interpolation='nearest', aspect='auto', origin='lower', **kwargs)

def _plotLineOnto(ax, data, labels, colors, split):
    ax.patch.set_facecolor('black')
    if isinstance(data, list):
        assert isinstance(labels, list) and isinstance(colors, list)
        assert len(data) == len(labels) and len(data) == len(colors)
        perLineOffset = np.max(data) - np.min(data) if split else 0.0
        for i, (d, l, c) in enumerate(zip(data, labels, colors)):
            ax.plot(d.T + i * perLineOffset, label=l, c=c, linewidth=1)
            nSamples = d.shape[0]
        ax.legend()
    else:
        if colors is not None:
            ax.set_prop_cycle('color', colors)
        perLineOffset = np.max(data) - np.min(data) if split else 0.0
        dataCopy = np.copy(data)
        for i in range(data.shape[0]):
            dataCopy[i] += i * perLineOffset
        ax.plot(dataCopy.T)
        nSamples = data.shape[1]
    ax.set_xlim((0, nSamples))

def _plotBranchesOnto(ax, branches, yLim):
    branchRGB = LINE_COLORS[(np.array(branches)) % LINE_COLOR_COUNT]
    branchRGB = np.expand_dims(branchRGB, axis=1)
    ax.imshow(branchRGB, interpolation='nearest', aspect='auto', origin='lower')

def _plotStimOnto(ax, stim, xLim, isDataPlot=False):
    alpha = 0.2 if isDataPlot else 1.0
    ls = '--' if isDataPlot else '-'
    ax.set_xlim(xLim)
    ax.patch.set_facecolor('black')
    for stimEnd in stim[:, 1]:
        ax.axvline(x=stimEnd, color=(1.0, 0.0, 0.0, alpha), linestyle=ls)
    for stimStart in stim[:, 0]:
        ax.axvline(x=stimStart, color=(1.0, 1.0, 0.0, alpha), linestyle=ls)

def plotIntensity(data, hz, branches=None, stim=None, title=None, overlayStim=False, savePath=None, **kwargs):
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
    fig.subplots_adjust(left=PAD/2, right=(1 - PAD/2), top=(1 - PAD), bottom=PAD)

    _plotIntensityOnto(aData, data, **kwargs)
    if aBranches is not None:
        aData.get_yaxis().set_visible(False)
        aBranches.get_xaxis().set_visible(False)
        aBranches.set_ylabel("Node ID and Branch")

        fig.subplots_adjust(wspace=0.0)
        _plotBranchesOnto(aBranches, branches, yLim=aData.get_ylim())
    else:
        aData.set_ylabel("Node ID")

    if aStim is not None:
        aData.get_xaxis().set_visible(False)
        aStim.get_yaxis().set_visible(False)
        aStim.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
        aStim.set_xticks(stim[:, 0])
        aStim.set_xlabel("Time and Stimuli")

        fig.subplots_adjust(hspace=0.0)
        _plotStimOnto(aStim, stim, xLim=aData.get_xlim())
        if overlayStim:
            _plotStimOnto(aData, stim, xLim=aData.get_xlim(), isDataPlot=True)
    else:
        aData.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
        aData.set_xlabel("Time")

    if aBlank is not None:
        aBlank.get_xaxis().set_visible(False)
        aBlank.get_yaxis().set_visible(False)

    if savePath is not None:
        fig.savefig(savePath)

def plotLine(data, hz, branches=None, stim=None, labels=None, colors=None, title=None, split=True, limitSec=None, overlayStim=True, savePath=None):
    fig, aData, aStim = None, None, None
    if stim is None:
        fig, (aData) = plt.subplots(1, 1) # gridspec_kw = {'width_ratios':[3, 1]})
    else:
        fig, (aData, aStim) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[8, 1]})
    if title is not None:
        fig.suptitle(title)
    fig.subplots_adjust(left=PAD, right=(1 - PAD), top=(1 - PAD), bottom=PAD)

    # Color by branches if required.
    if colors is None and branches is not None:
        colors = [(0.8,0.8,0.8) if b == -1 else LINE_COLORS[b % LINE_COLOR_COUNT] for b in branches]

    _plotLineOnto(aData, data, labels, colors, split)

    if aStim is not None:
        aData.get_xaxis().set_visible(False)
        aStim.get_yaxis().set_visible(False)
        aStim.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
        aStim.set_xticks(stim[:, 0])

        fig.subplots_adjust(hspace=0.0)
        _plotStimOnto(aStim, stim, xLim=aData.get_xlim())
        if overlayStim:
            _plotStimOnto(aData, stim, xLim=aData.get_xlim(), isDataPlot=True)
    else:
        aData.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
    #Disabled so we have a hacky way to see y scale
    #if split:
    #    aData.get_yaxis().set_visible(False)

    # Cull the start/end times:
    if limitSec is not None:
        startTime, endTime = limitSec
        startSample = None if startTime is None else startTime * hz
        endSample = None if endTime is None else endTime * hz
        aData.set_xlim(startSample, endSample)
        if aStim is not None:
            aStim.set_xlim(startSample, endSample)

    if savePath is not None:
        fig.savefig(savePath)


def plotAveragePostStimIntensity(data, hz, stimOffIdx, stimOnIdx, branches=None, title=None, secAfter=3, savePath=None, **kwargs):
    fig, aBranches, aData = None, None, None
    if branches is None:
        fig, (aDataOff, aDataOn) = plt.subplots(2, 1)
    else:
        fig, ((aBranchesOff, aDataOff), (aBranchesOn, aDataOn)) = \
            plt.subplots(2, 2, gridspec_kw = {'width_ratios':[1, 20]})

    if title is not None:
        fig.suptitle(title)

    offAverage = epochAverage(data, hz, stimOffIdx, 0, secAfter)
    onAverage  = epochAverage(data, hz,  stimOnIdx, 0, secAfter)
    maxOn, maxOff = np.max(onAverage), np.max(offAverage)
    if 'vmax' not in kwargs:
        kwargs['vmax'] = max(maxOn, maxOff)
    if 'vmin' not in kwargs:
        kwargs['vmin'] = 0
    aDataOff.set_title("Av. OFF stim response (%.2fs, max = %.2f/%.2f)" % (secAfter, maxOff, kwargs['vmax']))
    aDataOn.set_title("Av. ON stim response (%.2fs, max = %.2f/%.2f)" % (secAfter, maxOn, kwargs['vmax']))
    fig.subplots_adjust(left=PAD, right=(1 - PAD), top=(1 - PAD), bottom=PAD, hspace=0.2)

    _plotIntensityOnto(aDataOff, offAverage.clip(min=0), **kwargs)
    _plotIntensityOnto(aDataOn, onAverage.clip(min=0), **kwargs)
    if aBranchesOff is not None:
        aDataOff.get_yaxis().set_visible(False)
        aDataOn.get_yaxis().set_visible(False)
        aBranchesOff.get_xaxis().set_visible(False)
        aBranchesOn.get_xaxis().set_visible(False)
        aBranchesOff.set_ylabel("Node ID and Branch")
        aBranchesOn.set_ylabel("Node ID and Branch")

        fig.subplots_adjust(wspace=0.0)
        _plotBranchesOnto(aBranchesOff, branches, yLim=aDataOff.get_ylim())
        _plotBranchesOnto(aBranchesOn, branches, yLim=aDataOn.get_ylim())
    else:
        aDataOff.set_ylabel("Node ID")
        aDataOn.set_ylabel("Node ID")

    aDataOff.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
    aDataOn.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))

    if savePath is not None:
        fig.savefig(savePath)

def plotAveragePostStimTransientParams(dfof, hz, stimOffsets, secAfter, vizTrace=None, savePath=None):
    windowSz = secAfter * hz
    if vizTrace is not None:
        t = "df/f0 for trace %d" % vizTrace
        stim = np.array([stimOffsets, stimOffsets + windowSz]).T
        plotLine(np.array([dfof[vizTrace]]), hz=hz, stim=stim, title=t, split=False)

    allParams = []
    for trace in tqdm_notebook(range(dfof.shape[0])):
        y = np.mean(np.array([dfof[trace, i:i + windowSz] for i in stimOffsets]), axis=0)
        y = y - np.min(y)
        params, bestFit = fitDoubleExp(y, hz=hz)
        if not np.isnan(np.array(params)).any() and params[0] > 0.1:
            allParams.append(params)
        if trace == vizTrace:
            col = [LINE_COLORS[b % LINE_COLOR_COUNT] for b in range(2)]
            lab = ["dfof", "best fit"]
            plotLine([y, bestFit], hz=hz, split=False, colors=col, labels=lab)

    allP = np.array(allParams)
    allA, allT0, allTA, allTB = allP[:, 0], allP[:, 1] / hz, allP[:, 2] / hz, allP[:, 3] / hz

    fig, ax = plt.subplots(2, 2, tight_layout=True)
    ax[0][0].hist(allA)
    ax[0][1].hist(allT0)
    ax[1][0].hist(allTA)
    ax[1][1].hist(allTB)
    ax[0][0].set_title("A")
    ax[0][1].set_title("t0")
    ax[1][0].set_title("tA")
    ax[1][1].set_title("tB")
    # plt.show()

    if savePath is not None:
        fig.savefig(savePath)

def plotPlanarStructure(tree, rootID, nodeXYZ, branchIDs=None, colors=None, title=None, flipY=False, scale=10000, savePath=None, lineAlpha=0.8, flatten='Z'):
    # Default to flatten Z
    idxA, idxB = 0, 1 # X, Y
    scaleA, scaleB = scale, scale * (-1 if flipY else 1)
    if flatten == 'X':
        idxA, idxB = 1, 2 # Y, Z
        scaleA, scaleB = scale * (-1 if flipY else 1), scale
    elif flatten == 'Y':
        idxA, idxB = 0, 2 # X, Z
        scaleA, scaleB = scale, scale
   
    fig, ax = plt.subplots(1, 1)
    ax.patch.set_facecolor('black')
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if title is not None:
        ax.set_title(title)
        
    if colors is None and branchIDs is not None:
        colors = []
        for branchID in branchIDs:
            color = (1,1,1,0.6) if branchID == -1 else LINE_COLORS[branchID % LINE_COLOR_COUNT]
            colors.append(color)
        
    # First find the closest branch to the soma, and plot it bigger in that colour
    loc = tree[rootID]['location']
    nearestBranch, nearestDelta = 0, 1e9
    for i in range(len(nodeXYZ)):
        delta = np.linalg.norm(list(a - b for a, b in zip(nodeXYZ[i], loc)))
        if delta < nearestDelta:
            nearestDelta = delta
            nearestColor = colors[i]
    x = scaleA * tree[rootID]['location'][idxA]
    y = scaleB * tree[rootID]['location'][idxB]
    #c = (1,1,1) if nearestBranch == -1 else LINE_COLORS[nearestBranch % LINE_COLOR_COUNT]
    ax.scatter(x, y, s=150, c=[(nearestColor[0], nearestColor[1], nearestColor[2], 0.6)], marker='o')

    if branchIDs is not None:
        for branch in range(np.min(branchIDs), np.max(branchIDs) + 1):
            x = scaleA * nodeXYZ[branchIDs == branch, idxA]
            y = scaleB * nodeXYZ[branchIDs == branch, idxB]
            c = (1,1,1,0.6) if branch == -1 else LINE_COLORS[branch % LINE_COLOR_COUNT]
            s = 16 if branch == -1 else 36
            ax.scatter(x, y, c=[c], s=s)
    else:
        x = scaleA * nodeXYZ[:, idxA]
        y = scaleB * nodeXYZ[:, idxB]
        ax.scatter(x, y, c=colors, s=36)

    lines = _genLines(tree, rootID, scale=scale, flipY=flipY, flatten=flatten)
    lineCollection = mc.LineCollection(lines, colors=[(1,1,1,lineAlpha)], linewidths=1)
    ax.add_collection(lineCollection)

    if savePath is not None:
        fig.savefig(savePath)

# Scatter plot of all samples from traces from a filopodia tip vs. filopodia base.
def plotBaseTipScatter(baseTrace, tipTrace, title=None, **kwargs):
    fig, (ax) = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)

    ax.set_aspect('equal')
    ax.scatter(baseTrace, tipTrace, **kwargs)
    ax.set_xlabel('Base DF/F0')
    ax.set_ylabel('Tip DF/F0')

    (xLo, xHi) = ax.get_xlim()
    (yLo, yHi) = ax.get_ylim()
    bounds = [max(xLo, yLo), min(xHi, yHi)]
    ax.plot(bounds, bounds, c='k', label='Base = Tip')
    ax.legend()

def _buildStimAlpha(n, stim):
    if stim is None:
        return None

    # TODO - change to actual on vs off?
    stimAlpha = np.zeros(n)
    for i in range(stim.shape[0]):
        stimAlpha[stim[i][0]:(stim[i][1] + 1)] = 1.0
    return stimAlpha

def planarAnimation(tree, rootID, nodeXYZ, traceData, hz, flipY=False, stim=None, stimXY=(0,0), radius=0.005, savePath=None, scale=1, flatten='Z'):
    scales = np.array([scale, scale * (-1 if flipY else 1)])
    stimAlpha = _buildStimAlpha(traceData.shape[1], stim)

    if (hz < 10):
        print ("%d hz too small for output video, increasing..." % hz)
        hz = hz * 3

    DOWNSAMPLE = 1
    hz = hz // DOWNSAMPLE
    traceData = traceData[:, ::DOWNSAMPLE]
    traceData = traceData / np.max(traceData)
    nFrames = traceData.shape[1]

    xys = [(nodeXYZ[i, 0] * scales[0], nodeXYZ[i, 1] * scales[1]) for i in range(nodeXYZ.shape[0])]
    xs, ys = zip(*xys)

    PAD = 0.02
    xlim = (np.min(xs) - PAD, np.max(xs) + PAD)
    ylim = (np.min(ys) - PAD, np.max(ys) + PAD)

    # fig, ax = plt.subplots(figsize=(13, 6))
    width = 10
    height = width *  (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig, ax = plt.subplots(figsize=(width, height))

    PAD = 0
    fig.subplots_adjust(left=PAD/2, right=(1 - PAD/2), top=(1 - PAD), bottom=PAD)
    ax.patch.set_facecolor('black')
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Connections in the tree:
    lines = _genLines(tree, rootID, scale=scale, flipY=flipY)
    ax.add_collection(mc.LineCollection(lines, colors=[(0.5,0.5,0.5,0.25)], linewidths=1))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    progressBar = tqdm_notebook(total=nFrames)

    frameCircles = [patches.Circle(xy, radius=radius) for xy in xys]
    patchCollection = mc.PatchCollection(frameCircles, cmap=plt.get_cmap('hot'))
    patchCollection.set_clim([0, 1])
    ax.add_collection(patchCollection)

    stimPatch = None
    if stimAlpha is not None:
        xPos = (xlim[0] + xlim[1]) / 2.0 + stimXY[0] * (xlim[1] - xlim[0]) / 2.0
        yPos = (ylim[0] + ylim[1]) / 2.0 + stimXY[1] * (ylim[1] - ylim[0]) / 2.0
        stimPatch = patches.Rectangle((xPos, yPos), radius * 3, radius * 3, color=(1,1,1,1))
        ax.add_patch(stimPatch)

    def _animFrame(i):
        patchCollection.set(array=traceData[:, i], cmap='hot')
        progressBar.update(1)
        if savePath is None:
            ax.set_xlabel("%.2fs / %.2fs" % (i / hz, traceData.shape[1] / hz))
        if stimPatch is None:
            return patchCollection,
        else:
            stimPatch.set_facecolor((1, 1, 1, stimAlpha[i]))
            return patchCollection, stimPatch

    intMS = 1000.0 / hz
    anim = FuncAnimation(fig, _animFrame, frames=nFrames, blit=True, interval=intMS)

    # Set up formatting for the movie files
    if savePath is not None:
        print ("Saving...")
        Writer = animation.writers['ffmpeg']
        anim.save(savePath, writer=Writer(fps=hz)) #, metadata=dict(artist='Me'), bitrate=1800))
        # anim.save("fire.gif", dpi=80, writer='imagemagick')

    # progressBar.close()
    print ("Saved!")

def _genLines(nodes, nodeAt, scale, flipY, flatten='Z'):
    # Default to flatten Z
    scales = np.array([scale, scale * (-1 if flipY else 1)])
    selector = [True, True, False]
    if flatten == 'X':
        scales = np.array([scale * (-1 if flipY else 1), scale])
        selector = [False, True, True]
    elif flatten == 'Y':
        scales = np.array([scale, scale])
        selector = [True, False, True]
        
    fullNode = nodes[nodeAt]
    lineList = []
    if 'children' in fullNode:
        for child in fullNode['children']:
            childId = child['id']
            lineList.append([
                fullNode['location'][selector] * scales,
                nodes[childId]['location'][selector] * scales
            ])
            lineList.extend(_genLines(nodes, childId, scale, flipY, flatten))
    return lineList

# Shows raw intensity across the 11 (or whatever) pixel line scanned around the POI,
# used to check whether the POI drifted away from the sensed location.
def kymograph(kymoData, hz, smooth=False, title=None, widthInches=10, heightInches=60):
    # Optionally smooth with neighbours, to give a less noisy sense of drift
    if smooth:
        kymoData = np.copy(kymoData)
        kymoData = np.pad(kymoData, 1, 'edge')
        kymoData = (kymoData[:, 2:] + kymoData[:, 1:-1] + kymoData[:, :-2]) / 3
        kymoData = (kymoData[2:, :] + kymoData[1:-1, :] + kymoData[:-2, :]) / 3

    fig, (ax) = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel("Time")
    ax.set_xlabel("Pixel offset")
    _plotIntensityOnto(ax, kymoData[::-1])
    ax.figure.set_size_inches(widthInches, heightInches)

    halfSize = kymoData.shape[1]//2
    def _yLabelFormatter(y, pos):
        y = kymoData.shape[0] - y # top left = first sample, so invert
        return "%.2fs" % (y / hz)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (x - halfSize)))
    ax.get_yaxis().set_major_formatter(FuncFormatter(_yLabelFormatter))

# Debug helper to print tree strucuture to commandline:
def printTree(nodeAt, nodes, indent=''):
    print (indent + str(nodeAt))
    for child in nodes[nodeAt]['children']:
        printTree(child['id'], nodes, indent + ' ')
