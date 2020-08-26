import numpy as np

from tqdm import tqdm, tqdm_notebook

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from matplotlib import collections as mc
from matplotlib.animation import FuncAnimation
from matplotlib_scalebar.scalebar import ScaleBar

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
        
    colorBarShrink = kwargs.pop('colorBarShrink', 0.0)
    im = ax.imshow(
        data, cmap='hot', interpolation='nearest', aspect='auto', origin='lower', **kwargs)
    if colorBarShrink > 0:
        cbar = ax.figure.colorbar(im, ax=ax, shrink=colorBarShrink)
        cbar.ax.set_xlabel(kwargs.pop('colorBarTitle', 'DF/F0'))
        cbar.ax.xaxis.set_label_position('top') 

def _plotLineOnto(ax, data, labels, colors, split, yPad=.05):
    ax.patch.set_facecolor('black')
    
    ymin, ymax = None, None
    
    if isinstance(data, list):
        assert isinstance(labels, list) and isinstance(colors, list)
        assert len(data) == len(labels) and len(data) == len(colors)
        perLineOffset = np.max(data) - np.min(data) if split else 0.0
        for i, (d, l, c) in enumerate(zip(data, labels, colors)):
            ax.plot(d.T + i * perLineOffset, label=l, c=c, linewidth=1)
            if ymin is None:
                ymin, ymax = np.min(d.T + i * perLineOffset), np.max(d.T + i * perLineOffset)
            else:
                ymin, ymax = min(ymin, np.min(d.T + i * perLineOffset)), max(ymax, np.max(d.T + i * perLineOffset))
            nSamples = d.shape[0]
        ax.legend()
        yZeros = [i * perLineOffset for i in range(len(data))]
    else:
        if colors is not None:
            ax.set_prop_cycle('color', colors)
        perLineOffset = np.max(data) - np.min(data) if split else 0.0
        dataCopy = np.copy(data)
        for i in range(data.shape[0]):
            dataCopy[i] += i * perLineOffset
        ax.plot(dataCopy.T)
        ymin, ymax = np.min(dataCopy), np.max(dataCopy)
        nSamples = data.shape[1]
        yZeros = [i * perLineOffset for i in range(data.shape[0])]
    ax.set_xlim((0, nSamples))
    ax.set_ylim(ymin + yPad * (ymin - ymax), ymax + yPad * (ymax - ymin))
    return np.array(yZeros if split else [0])

def _plotBranchesOnto(ax, branches, yLim, colors=None):
    # TODO: Remove branches, only accept colors.
    branchRGB = colors
    if branchRGB is None:
        branchRGB = LINE_COLORS[(np.array(branches)) % LINE_COLOR_COUNT]
    branchRGB = np.expand_dims(branchRGB, axis=1)
    ax.imshow(branchRGB, interpolation='nearest', aspect='auto', origin='lower')

def hasTransitionShift(stim, hz):
    middleStimSamples = stim[4][1] - stim[4][0]
    middleStimMs = middleStimSamples / hz * 1000.0
    return middleStimMs > 1000

def _stimToHybridColoursImg(stim, hz, xLim, offV=0.0, onV=0.3, borderV=1.0, widen=1):
    assert stim.shape[0] == 9, "Hybrid Stim needs 9 stim values, do you need hybridStimColours=False?"
    hasTransition = hasTransitionShift(stim, hz)
    
    nSamples = int(round(xLim[1] + 0.5)) # Limit will be #samples - 0.5
    asImg = np.zeros((20, nSamples))
    
    if hasTransition:
        # First stage: background = red, four stim of black
        asImg[:, :stim[4, 0]] = onV
        for i in range(4):
            asImg[:, stim[i,0]:stim[i,1] + widen] = offV
        # Middle: transition shift:
        for sample in range(stim[4,0], stim[4,1] + widen):
            factor = (sample - stim[4,0]) / (stim[4,1] - stim[4,0])
            asImg[:, sample] = factor * offV + (1 - factor) * onV
        # Last stage: background = black, four stim of red
        asImg[:, stim[4, 1]:] = offV
        for i in range(5,9):
            asImg[:, stim[i,0]:stim[i,1] + widen] = onV
    else:
        # All off stim:
        asImg[:, :] = onV
        for i in range(stim.shape[0]):
            asImg[:, stim[i,0]:stim[i,1] + widen] = offV
    asImg[-1, :] = borderV
    return asImg
    
def _plotStimOnto(ax, stim, hz, xLim, hybridStimColours=False, isDataPlot=False, **kwargs):
    mappable = None
    if hybridStimColours and not isDataPlot:
        stimAsImg = _stimToHybridColoursImg(stim, hz, xLim)
        mappable = ax.imshow(stimAsImg, cmap='hot', interpolation=None, aspect='auto', origin='lower', vmax=1)
    else:
        alpha = 0.2 if isDataPlot else 1.0
        ls = '--' if isDataPlot else '-'
        ax.set_xlim(xLim)
        # Black background.
        nSamples = int(round(xLim[1] + 0.5))
        mappable = None
        if not isDataPlot:
            mappable = ax.imshow(np.zeros((20, nSamples)), cmap='hot', interpolation=None, aspect='auto', origin='lower', vmax=1)
        for stimEnd in stim[:, 1]:
            ax.axvline(x=stimEnd, color=(1.0, 0.0, 0.0, alpha), linestyle=ls)
        for stimStart in stim[:, 0]:
            ax.axvline(x=stimStart, color=(1.0, 1.0, 0.0, alpha), linestyle=ls)
            
    colorBarShrink = kwargs.pop('colorBarShrink', 0.0)
    if colorBarShrink > 0 and not isDataPlot and mappable is not None:
        cbar = ax.figure.colorbar(mappable, ax=ax, shrink=colorBarShrink)

def plotIntensity(data, hz, branches=None, colors=None, stim=None, title=None, 
    overlayStim=False, savePath=None, hybridStimColours=False, forceStimWidth=None, 
    **kwargs):
    with plt.style.context(('seaborn-dark-palette')):
        fig, aBranches, aData, aStim, aBlank = None, None, None, None, None
        xAx, yAx = None, None
        
        wRatio = kwargs.pop('width_ratio', 20)
        hRatio = kwargs.pop('height_ratio', 8)

        if branches is not None:
            assert colors is None, "Cannot provide both colors and branches to plotIntensity"
            colors = LINE_COLORS[(np.array(branches)) % LINE_COLOR_COUNT]

        
        if colors is None and stim is None:
            fig, (aData) = plt.subplots(1, 1)
            xAx, yAx = aData, aData
        elif colors is not None and stim is None:
            fig, (aBranches, aData) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1, wRatio]})
            xAx, yAx = aData, aBranches
        elif colors is None and stim is not None:
            fig, (aData, aStim) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[hRatio, 1]})
            xAx, yAx = aStim, aData
        else:
            fig, ((aBranches, aData), (aBlank, aStim)) = plt.subplots(2, 2, gridspec_kw = {'height_ratios':[hRatio, 1], 'width_ratios':[1, wRatio]})
            xAx, yAx = aStim, aBranches
        fig.suptitle(title)
        fig.subplots_adjust(left=PAD/2, right=(1 - PAD/2), top=(1 - PAD), bottom=PAD)

        kwargsCpy = kwargs.copy()
        _plotIntensityOnto(aData, data, **kwargs)
        if aBranches is not None:
            aData.get_yaxis().set_visible(False)
            aBranches.get_xaxis().set_visible(False)
            aBranches.set_ylabel("Node ID and Branch")

            fig.subplots_adjust(wspace=0.0)
            _plotBranchesOnto(aBranches, None, yLim=aData.get_ylim(), colors=colors)
        else:
            aData.set_ylabel("Node ID")

        if aStim is not None:
            if forceStimWidth is not None:
                if hybridStimColours:
                    oldTransitionEnd = stim[4, 1]
                    stim[:, 1] = stim[:, 0] + forceStimWidth
                    stim[4, 1] = oldTransitionEnd
                else:
                    stim[:, 1] = stim[:, 0] + forceStimWidth
        
            aData.get_xaxis().set_visible(False)
            aStim.get_yaxis().set_visible(False)
            aStim.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
            aStim.set_xticks(stim[:, 0])
            aStim.set_xlabel("Time and Stimuli")

            fig.subplots_adjust(hspace=0.0)
            _plotStimOnto(aStim, stim, hz, xLim=aData.get_xlim(), hybridStimColours=hybridStimColours, isDataPlot=False, **kwargsCpy)
            if overlayStim:
                _plotStimOnto(aData, stim, hz, xLim=aData.get_xlim(), hybridStimColours=False, isDataPlot=True)
        else:
            aData.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
            aData.set_xlabel("Time")

        if aBlank is not None:
            aBlank.get_xaxis().set_visible(False)
            aBlank.get_yaxis().set_visible(False)

        if savePath is not None:
            fig.savefig(savePath)
            
    return xAx, yAx

def plotLine(data, hz, branches=None, stim=None, labels=None, colors=None, title=None, yTitle='DF/F0',
        split=True, limitSec=None, overlayStim=True, savePath=None, hybridStimColours=True, 
        yTickScale=1, yTickPct=True, yPad=.05):
    with plt.style.context(('seaborn-dark-palette')):        
        fig, aData, aStim = None, None, None
        xAx, yAx = None, None
        
        if stim is None:
            fig, (aData) = plt.subplots(1, 1) # gridspec_kw = {'width_ratios':[3, 1]})
            xAx, yAx = aData, aData
        else:
            fig, (aData, aStim) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[8, 1]})
            xAx, yAx = aStim, aData
        if title is not None:
            fig.suptitle(title)
        fig.subplots_adjust(left=PAD, right=(1 - PAD), top=(1 - PAD), bottom=PAD)

        # Color by branches if required.
        if colors is None and branches is not None:
            colors = [(0.8,0.8,0.8) if b == -1 else LINE_COLORS[b % LINE_COLOR_COUNT] for b in branches]

        yZeros = _plotLineOnto(aData, data, labels, colors, split, yPad=yPad)

        if aStim is not None:
            aData.get_xaxis().set_visible(False)
            aStim.get_yaxis().set_visible(False)
            aStim.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
            aStim.set_xticks(stim[:, 0])

            fig.subplots_adjust(hspace=0.0)
            _plotStimOnto(aStim, stim, hz, xLim=aData.get_xlim(), hybridStimColours=hybridStimColours)
            if overlayStim:
                _plotStimOnto(aData, stim, hz, xLim=aData.get_xlim(), hybridStimColours=False, isDataPlot=True)
        else:
            aData.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))

        # Cull the start/end times:
        if limitSec is not None:
            startTime, endTime = limitSec
            startSample = None if startTime is None else startTime * hz
            endSample = None if endTime is None else endTime * hz
            aData.set_xlim(startSample, endSample)
            if aStim is not None:
                aStim.set_xlim(startSample, endSample)

        # Show Y Axis title as well as major ticks for zeros, minor for DF/F0=x
        if yTitle is not None:
            yAx.set_ylabel(yTitle)
        
        if yTickScale is not None:
            tickFmt = "%d%%" if yTickPct else "%.2f"            
            zeroText, deltaText = tickFmt % (0), tickFmt % (yTickScale * (100 if yTickPct else 1))
            yAx.set_yticks(yZeros, minor=False)
            yAx.set_yticks(yZeros + yTickScale, minor=True)
            yAx.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: zeroText))
            yAx.get_yaxis().set_minor_formatter(FuncFormatter(lambda x, pos: deltaText))
            
        xAx.set_xlabel("Time (sec)")
    
        if savePath is not None:
            fig.savefig(savePath)
            
    return xAx, yAx


def plotAveragePostStimIntensity(data, hz, stimOffIdx, stimOnIdx, branches=None, title=None, secBefore=0, secAfter=3, savePath=None, **kwargs):
    with plt.style.context(('seaborn-dark-palette')):   
        fig, aBranchesOff, aBranchesOn, aDataOff, aDataOn = None, None, None, None, None
        
        drawOn = (stimOnIdx is not None)
        nRows = 2 if drawOn else 1
        
        if branches is None:
            if drawOn:
                fig, (aDataOff, aDataOn) = plt.subplots(2, 1)
            else:
                fig, aDataOff = plt.subplots(1, 1)
        else:
            if drawOn:
                fig, ((aBranchesOff, aDataOff), (aBranchesOn, aDataOn)) = \
                    plt.subplots(2, 2, gridspec_kw = {'width_ratios':[1, 20]})
            else:
                fig, (aBranchesOff, aDataOff) = \
                    plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1, 20]})

        fig.subplots_adjust(left=PAD, right=(1 - PAD), top=(1 - PAD), bottom=PAD, hspace=0.2)
        if title is not None:
            fig.suptitle(title)

        offAverage = epochAverage(data, hz, stimOffIdx, secBefore, secAfter)
        maxOff = np.max(offAverage)
        vmax = maxOff
        if drawOn:
            onAverage  = epochAverage(data, hz,  stimOnIdx, secBefore, secAfter)
            maxOn = np.max(onAverage)
            vmax = max(vmax, maxOff)

        if 'vmax' not in kwargs:
            kwargs['vmax'] = vmax
        if 'vmin' not in kwargs:
            kwargs['vmin'] = 0
            
        aDataOff.set_title("Av. OFF stim response (%.2fs, max = %.2f/%.2f)" % (secAfter, maxOff, kwargs['vmax']))
        _plotIntensityOnto(aDataOff, offAverage.clip(min=0), **kwargs)
        if drawOn:
            aDataOn.set_title("Av. ON stim response (%.2fs, max = %.2f/%.2f)" % (secAfter, maxOn, kwargs['vmax']))
            _plotIntensityOnto(aDataOn, onAverage.clip(min=0), **kwargs)
            
        if aBranchesOff is not None:
            aDataOff.get_yaxis().set_visible(False)
            aBranchesOff.get_xaxis().set_visible(False)
            aBranchesOff.set_ylabel("Node ID and Branch")
            if drawOn:
                aDataOn.get_yaxis().set_visible(False)
                aBranchesOn.get_xaxis().set_visible(False)
                aBranchesOn.set_ylabel("Node ID and Branch")

            fig.subplots_adjust(wspace=0.0)
            _plotBranchesOnto(aBranchesOff, branches, yLim=aDataOff.get_ylim())
            if drawOn:
                _plotBranchesOnto(aBranchesOn, branches, yLim=aDataOn.get_ylim())
        else:
            aDataOff.set_ylabel("Node ID")
            if drawOn:
                aDataOn.set_ylabel("Node ID")

        aDataOff.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
        if drawOn:
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

    with plt.style.context(('seaborn-dark-palette')):   
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
    return ax

def plotPlanarStructure(
    tree, rootID, nodeXYZ, 
    branchIDs=None, colors=None, title=None, 
    flipY=False, scale=1, savePath=None, 
    lineAlpha=0.8, flatten='Z', pixelMeters=None, 
    palette='seaborn-dark-palette', bgColor='black'):
    
    # Default to flatten Z
    idxA, idxB = 0, 1 # X, Y
    scaleA, scaleB = scale, scale * (-1 if flipY else 1)
    if flatten == 'X':
        idxA, idxB = 1, 2 # Y, Z
        scaleA, scaleB = scale * (-1 if flipY else 1), scale
    elif flatten == 'Y':
        idxA, idxB = 0, 2 # X, Z
        scaleA, scaleB = scale, scale
   
    with plt.style.context((palette)):   
        fig, ax = plt.subplots(1, 1)
        ax.patch.set_facecolor(bgColor)
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

        somaCol = nearestColor
        if isinstance(nearestColor, list):
            somaCol = (nearestColor[0], nearestColor[1], nearestColor[2], 0.6)
        ax.scatter(x, y, s=150, c=[somaCol], marker='o')

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
        
        if pixelMeters is not None:
            ax.add_artist(ScaleBar(pixelMeters, 'm', 'si-length', color='w', box_color='k'))

        if savePath is not None:
            fig.savefig(savePath)
        
    return ax

# Scatter plot of all samples from traces from a filopodia tip vs. filopodia base.
def plotBaseTipScatter(baseTrace, tipTrace, title=None, **kwargs):
    with plt.style.context(('seaborn-dark-palette')):   
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
    return ax

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

    with plt.style.context(('seaborn-dark-palette')):   
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
    return ax

# Debug helper to print tree strucuture to commandline:
def printTree(nodeAt, nodes, indent=''):
    print (indent + str(nodeAt))
    for child in nodes[nodeAt]['children']:
        printTree(child['id'], nodes, indent + ' ')

def scrollingAnimation(data, hz, branches, stim, title, savePath):
    fig, ((aBranches, aData), (aBlank, aStim)) = plt.subplots(2, 2, figsize=(8,10), gridspec_kw = {'height_ratios':[8, 1], 'width_ratios':[1, 20]})
    fig.suptitle(title)
    fig.subplots_adjust(left=PAD/2, right=(1 - PAD/2), top=(1 - PAD), bottom=PAD)

    aData.get_yaxis().set_visible(False)
    aBranches.get_xaxis().set_visible(False)
    aBranches.get_yaxis().set_visible(False)

    fig.subplots_adjust(wspace=0.0)
    _plotBranchesOnto(aBranches, branches, yLim=aData.get_ylim())

    aData.get_xaxis().set_visible(False)
    aStim.get_yaxis().set_visible(False)
    aStim.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: "%.2fs" % (x / hz)))
    aStim.set_xlabel("Time and Stimuli")
    fig.subplots_adjust(hspace=0.0)

    if aBlank is not None:
        aBlank.get_xaxis().set_visible(False)
        aBlank.get_yaxis().set_visible(False)

    DOWNSAMPLE = 1

    samples = data.shape[1]

    progressBar = tqdm(total=samples//DOWNSAMPLE)
    aLine = aData.axvline(-1)

    # HACK for intensity:
    data[data.shape[0] - 1, 0] = np.max(data)


    def _animFrame(i):
        f = i * DOWNSAMPLE
        cell = np.copy(data)
        cell[:, f:] = 0
        cell = cell.clip(min=0)
        _plotIntensityOnto(aData, cell)
        aLine.set_xdata(f)
        cStim = stim[stim[:, 0] <= f]
        aStim.set_xticks(cStim[:, 0])
        _plotStimOnto(aStim, cStim, xLim=aData.get_xlim())

        progressBar.update(1)
        return fig,

    intMS = DOWNSAMPLE * 1000.0 / hz
    anim = FuncAnimation(fig, _animFrame, frames=samples//DOWNSAMPLE, blit=True, interval=intMS)

    # Set up formatting for the movie files
    print ("Saving...")
    Writer = animation.writers['ffmpeg']
    anim.save(savePath, writer=Writer(fps=hz/DOWNSAMPLE)) #, metadata=dict(artist='Me'), bitrate=1800))
    # anim.save("fire.gif", dpi=80, writer='imagemagick')
