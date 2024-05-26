# pyNeuroTrace Documentation

The will file will provide and overview documentation of the pyNeuroTrace library


# Table of Contents
1. [Installation](#Installation)
2. [Overview](#Overview)
3. [Modules](#Modules)
    - [events](#events)
    - [files](#files)
    - [filters](#filters)
    - [morphology](#morphology)
    - [notebook](#notebook)
    - [viz](#viz)
4. [Examples][#Examples]        


# Installation
`pyNeuroTrace` can be installed with pip:

```
pip install pyNeuroTrace
```

If you wish to use optional GPU accelrated modules install `pyNeuroTrace` with this modification to the above command:

```
pip install pyNeuroTrace['GPU']
```
Or insure the `Cupy` is installed in your Python environment and that you have a CUDA enabled graphics card.

# Overview
This is an overview of the useful modules in the library

```plaintext
pyneurotrace
├── gpu
│   ├── events
│   ├── filters
├── events
├── files
├── filters
├── morphology
├── notebook
├── viz
```
# Modules
Here is an overview of the core modules and their functions
## events
Documentation for the functions in the events module
### ewma

`pyneurotrace.events.ewma(data, weight=0.1)`

Performs smoothing on each row using the Exponentially Weighted Moving Average (EWMA) method.

#### Parameters:

- `data`: array
  - Data array to be smoothed.
- `weight`: float, optional
  - Weight for new values versus existing average. Default is 0.1.

#### Returns:

- `smoothed_data`: array
  - Smoothed data array.
___

### cusum

`pyneurotrace.events.cusum(data, slack=1.0)`

Calculates the Cumulative Sum (CUSUM) of movement above the mean for each row in the data.

#### Parameters:

- `data`: array
  - Data array to be analyzed.
- `slack`: float, optional
  - Slack noise value. Default is 1.0.

#### Returns:

- `cusum_data`: array
  - CUSUM data array.
___

### matchedFilter

`pyneurotrace.events.matchedFilter(data, hz, windowSize, A=2.0, riseRate=0.028, decayRate=0.39)`

Performs a likelihood ratio test using a Matched Filter (MF) to see how well a previous window matches a desired signal over no signal.

#### Parameters:

- `data`: array
  - Data array to be filtered.
- `hz`: int
  - Sampling rate in Hz.
- `windowSize`: int
  - Size of the window to match.
- `A`: float, optional
  - Amplitude of the desired signal. Default is 2.0.
- `riseRate`: float, optional
  - Rise rate of the desired signal. Default is 0.028.
- `decayRate`: float, optional
  - Decay rate of the desired signal. Default is 0.39.

#### Returns:

- `mf_data`: array
  - Matched filter data array.
___

### thresholdEvents

`pyneurotrace.events.thresholdEvents(data, threshold, minBelowBefore=1)`

Turns an event detector into event yes/no indicators by applying a threshold and only keeping events that occur after a minimum number of non-event samples.

#### Parameters:

- `data`: array
  - Data array containing event detector outputs.
- `threshold`: float
  - Confidence threshold for events.
- `minBelowBefore`: int, optional
  - Minimum number of non-event samples before an event is considered. Default is 1.

#### Returns:

- `events`: array
  - Array indicating detected events.
___


## files 
Tools for loading and processing experimental data. These functions are to be used with the output of a custom two-photon AOD microscope.

### load2PData

`pyneurotrace.files.load2PData(path, hasLocation=True)`

Load Node IDs, positions, and raw data from an EXPT.TXT file.

#### Parameters:

- `path`
  - array
  - Path to the EXPT.TXT file.
- `hasLocation`
  - bool, optional
  - Flag indicating if location data is available. Default is `True`.

#### Returns:

- `nodeIDs`
  - array
  - List of node IDs.
- `positions`
  - array
  - Node positions (if `hasLocation` is `True`).
- `rawData`
  - array
  - Raw trace data.

___

### loadMetadata

`pyneurotrace.files.loadMetadata(path)`

Load stimulus [start, end] sample indices, plus sample rate, from the rscan_metadata .txt file.

#### Parameters:

- `path`
  - array
  - Path to the metadata file.

#### Returns:

- `stimIndices`
  - array
  - Array of stimulus start and end indices.
- `hz`
  - float
  - Sample rate in Hz.
- `xySizeM`
  - float
  - XY pixel size in meters.
- `zStackLocations`
  - array
  - Array of Z stack locations.

___

### loadTreeStructure

`pyneurotrace.files.loadTreeStructure(path)`

Load Tree structure (branch & parent details) from an interp-neuron-.txt file.

#### Parameters:

- `path`
  - array
  - Path to the tree structure file.

#### Returns:

- `rootId`
  - int
  - ID of the root node.
- `nodes`
  - dict
  - Dictionary of nodes with their details.

___

### loadNodeXYZ

`pyneurotrace.files.loadNodeXYZ(path)`

Load node IDs and their XYZ positions from a file.

#### Parameters:

- `path`
  - array
  - Path to the file containing node XYZ data.

#### Returns:

- `nodeIDs`
  - array
  - List of node IDs.
- `positions`
  - array
  - Node positions.

___

### loadKymograph

`pyneurotrace.files.loadKymograph(path, pxlPerNode=11)`

Load kymograph data from a file.

#### Parameters:

- `path`
  - array
  - Path to the kymograph file.
- `pxlPerNode`
  - int, optional
  - Number of pixels per node. Default is `11`.

#### Returns:

- `kymoData`
  - dict
  - Dictionary mapping node IDs to kymograph data.

___

### loadSingleStep

`loadSingleStep(stepPath, metaPath, treePath, xyzsPath=None, kymoPath=None, volumeXYZSource=None, convertXYZtoPx=False, normalizeXYZ=False)`

Loads as many of the above as possible for a single step of an experiment.

#### Parameters:

- `stepPath`
  - array
  - Path to EXPT.TXT raw data traces.
- `metaPath`
  - array
  - Path to metadata file containing sample rate, stim times, and pixel sizes.
- `treePath`
  - array
  - Path to tree structure file.
- `xyzsPath`
  - array, optional
  - Path to mapping from ID to x/y/z location of all points in the tree.
- `kymoPath`
  - array, optional
  - Path to kymograph intensity time series for all scanned points.
- `volumeXYZSource`
  - array, optional
  - Source for position data if not available for all points (e.g., planar scan).
- `convertXYZtoPx`
  - bool, optional
  - Flag indicating if XYZ should be converted to pixels. Default is `False`.
- `normalizeXYZ`
  - bool, optional
  - Flag indicating if XYZ should be normalized. Default is `False`.

#### Returns:

- `stepData`
  - dict
  - Dictionary containing the data for a single step of an experiment.

___

### loadHybrid

`pyneurotrace.files.loadHybrid(rootPath, loadKymoData=False, convertXYZtoPx=False, getPlanarXYZFromVolume=False, normalizeXYZ=False)`

Loads an entire hybrid experiment from a folder, containing many scan results.

#### Parameters:

- `rootPath`
  - array
  - Folder to load all the steps from.
- `loadKymoData`
  - bool, optional
  - Flag indicating whether to load kymographs. Default is `False`.
- `convertXYZtoPx`
  - bool, optional
  - Flag indicating if XYZ should be converted to pixels. Default is `False`.
- `getPlanarXYZFromVolume`
  - bool, optional
  - Flag indicating if planar XYZ should be obtained from volume. Default is `False`.
- `normalizeXYZ`
  - bool, optional
  - Flag indicating if XYZ should be normalized. Default is `False`.

#### Returns:

- `hybridData`
  - dict
  - Dictionary containing data for the entire hybrid experiment.

___

## filters
These functions are tools for smoothing and filtering timeseries data. 


### nndSmooth

`pyneurotrace.filters.nndSmooth(data, hz, tau, iterFunc=None)`

Performs fast non-negative temporal deconvolution for laser scanning microscopy.

#### Parameters:

- `data`
  - array
  - Data array to be smoothed.
- `hz`
  - int
  - Sampling rate in Hz.
- `tau`
  - float
  - Time constant for the exponential decay.
- `iterFunc`
  - function, optional
  - Optional iteration function. Default is `None`.

#### Returns:

- `smoothed_data`
  - array
  - Smoothed data array.

___

### okada

`pyneurotrace.filters.okada(data, iterFunc=None)`

A computationally efficient filter for reducing shot noise in low S/N data.

#### Parameters:

- `data`
  - array
  - Data array to be filtered.
- `iterFunc`
  - function, optional
  - Optional iteration function. Default is `None`.

#### Returns:

- `filtered_data`
  - array
  - Filtered data array.

___

### deltaFOverF0

`pyneurotrace.filters.deltaFOverF0(data, hz, t0=0.2, t1=0.75, t2=3.0, iterFunc=None)`

Calculates the change in fluorescence over baseline fluorescence. Optionally smoothed with an ewma.

#### Parameters:

- `data`
  - array
  - Data array to be analyzed.
- `hz`
  - int
  - Sampling rate in Hz.
- `t0`
  - float, optional
  - Time constant for exponential moving average. Default is `0.2`.
- `t1`
  - float, optional
  - Time window for calculating the mean baseline. Default is `0.75`.
- `t2`
  - float, optional
  - Time window for calculating the minimum baseline. Default is `3.0`.
- `iterFunc`
  - function, optional
  - Optional iteration function. Default is `None`.

#### Returns:

- `deltaF_over_F0`
  - array
  - Calculated change in fluorescence over baseline fluorescence.

___


## morphology
These functions are used to reconstruct a neuronal structure imaged using a custom AOD microscope. These file formats are unique to a microscope in the Haas lab.

### treePostProcessing

`pyneurotrace.morphology.treePostProcessing(nodeIDs, nodeXYZ, traceIDs, data, rootID, tree)`

Processes the tree structure, adding locations, calculating branches, and reordering nodes by branch.

#### Parameters:

- `nodeIDs`
  - array
  - List of node IDs.
- `nodeXYZ`
  - array
  - Array of node locations (XYZ coordinates).
- `traceIDs`
  - array
  - List of trace IDs.
- `data`
  - array
  - Data array containing raw traces.
- `rootID`
  - int
  - ID of the root node.
- `tree`
  - dict
  - Dictionary representing the tree structure.

#### Returns:

- `nodeIDs`
  - array
  - Processed list of node IDs.
- `nodeXYZ`
  - array
  - Processed array of node locations (XYZ coordinates).
- `finalTraceIDs`
  - array
  - Processed list of trace IDs.
- `finalTraceBranches`
  - array
  - List of trace branches.
- `data`
  - array
  - Processed data array containing raw traces.
- `branchIDs`
  - array
  - List of branch IDs.
- `branchIDMap`
  - dict
  - Dictionary mapping node IDs to branch IDs.

___

### buildBranchIDMap

`pyneurotrace.morphology.buildBranchIDMap(nodeID, nodes, splitAtBranch=False)`

Builds a map of branch IDs for the given tree structure.

#### Parameters:

- `nodeID`
  - int
  - ID of the starting node.
- `nodes`
  - dict
  - Dictionary representing the tree structure.
- `splitAtBranch`
  - bool, optional
  - Flag indicating whether to split at branches. Default is `False`.

#### Returns:

- `branchIDMap`
  - dict
  - Dictionary mapping node IDs to branch IDs.

___


### treeToFiloTipAndBase

`pyneurotrace.morphology.treeToFiloTipAndBase(nodeIDs, nodeXYZ, tree, rootID, filoDist=5.0)`

Maps nodes to filopodia tips and bases based on the specified distance.

#### Parameters:

- `nodeIDs`
  - array
  - List of node IDs.
- `nodeXYZ`
  - array
  - Array of node locations (XYZ coordinates).
- `tree`
  - dict
  - Dictionary representing the tree structure.
- `rootID`
  - int
  - ID of the root node.
- `filoDist`
  - float, optional
  - Distance threshold for identifying filopodia tips and bases. Default is `5.0`.

#### Returns:

- `mapping`
  - dict
  - Dictionary mapping node IDs to branch IDs, indicating filopodia tips and bases.
___

## notebook
Tools for setting up and interacting with IPython notebooks, including file dialogs and tabbed displays. These are generalizable and can be used for any project.
### filePicker

`pyneurotrace.notebook.filePicker(prompt="Select file", extension="", defaultPath='.')`

Show a dialog selection box to the user to pick one file.

#### Parameters:

- `prompt`
  - str, optional
  - The prompt message for the dialog. Default is `"Select file"`.
- `extension`
  - str, optional
  - File extension filter. Default is `""`.
- `defaultPath`
  - str, optional
  - Default path to start the file picker. Default is `'.'`.

#### Returns:

- `fname`
  - str
  - Path of the selected file.

___

### newFilePicker

`pyneurotrace.notebook.newFilePicker(prompt="New file", defaultPath='.')`

Show a dialog for the user to select the path of a new save file.

#### Parameters:

- `prompt`
  - str, optional
  - The prompt message for the dialog. Default is `"New file"`.
- `defaultPath`
  - str, optional
  - Default path to start the file picker. Default is `'.'`.

#### Returns:

- `fname`
  - str
  - Path of the new save file.

___

### folderPicker

`pyneurotrace.notebook.folderPicker(prompt="Output Folder", defaultPath='.')`

Show a folder selection dialog to the user and return the path.

#### Parameters:

- `prompt`
  - str, optional
  - The prompt message for the dialog. Default is `"Output Folder"`.
- `defaultPath`
  - str, optional
  - Default path to start the folder picker. Default is `'.'`.

#### Returns:

- `fname`
  - str
  - Path of the selected folder.

___

### showTabs

`pyneurotrace.notebook.showTabs(data, showOnTabFunc, titles=None, progressBar=False)`

Display data in multiple tabs in the same output area.

#### Parameters:

- `data`
  - list or dict
  - Data to be displayed in tabs.
- `showOnTabFunc`
  - function
  - Function to display data on each tab.
- `titles`
  - list, optional
  - List of titles for the tabs. Default is `None`.
- `progressBar`
  - bool, optional
  - Flag to show a progress bar. Default is `False`.

#### Returns:

- `None`
  - Displays data in tabs within the notebook.



## vis
Tools for visualizing experimental data, including intensity plots, line plots, and animations.

### plotIntensity

`pyneurotrace.vis.plotIntensity(data, hz, branches=None, colors=None, stim=None, title=None, overlayStim=False, savePath=None, hybridStimColours=False, forceStimWidth=None, **kwargs)`

Plots intensity data with optional branches and stimulus.

#### Parameters:

- `data`
  - array
  - Data to be plotted.
- `hz`
  - int
  - Sampling rate in Hz.
- `branches`
  - array, optional
  - Array of branch data. Default is `None`.
- `colors`
  - array, optional
  - Colors for the branches. Default is `None`.
- `stim`
  - array, optional
  - Array of stimulus data. Default is `None`.
- `title`
  - str, optional
  - Title for the plot. Default is `None`.
- `overlayStim`
  - bool, optional
  - Flag to overlay stimulus data. Default is `False`.
- `savePath`
  - str, optional
  - Path to save the plot. Default is `None`.
- `hybridStimColours`
  - bool, optional
  - Flag for hybrid stimulus colors. Default is `False`.
- `forceStimWidth`
  - int, optional
  - Force width for the stimulus. Default is `None`.
- `**kwargs`
  - Additional keyword arguments for customization.

#### Returns:

- `xAx`
  - object
  - x-axis object.
- `yAx`
  - object
  - y-axis object.

___

### plotLine

`pyneurotrace.vis.plotLine(data, hz, branches=None, stim=None, labels=None, colors=None, title=None, yTitle='DF/F0', split=True, limitSec=None, overlayStim=True, savePath=None, hybridStimColours=True, yTickScale=1, yTickPct=True, yPad=.05)`

Plots line data with optional branches and stimulus.

#### Parameters:

- `data`
  - array
  - Data to be plotted.
- `hz`
  - int
  - Sampling rate in Hz.
- `branches`
  - array, optional
  - Array of branch data. Default is `None`.
- `stim`
  - array, optional
  - Array of stimulus data. Default is `None`.
- `labels`
  - list, optional
  - Labels for the lines. Default is `None`.
- `colors`
  - array, optional
  - Colors for the lines. Default is `None`.
- `title`
  - str, optional
  - Title for the plot. Default is `None`.
- `yTitle`
  - str, optional
  - Title for the y-axis. Default is `'DF/F0'`.
- `split`
  - bool, optional
  - Flag to split the lines. Default is `True`.
- `limitSec`
  - tuple, optional
  - Time limits for the plot in seconds. Default is `None`.
- `overlayStim`
  - bool, optional
  - Flag to overlay stimulus data. Default is `True`.
- `savePath`
  - str, optional
  - Path to save the plot. Default is `None`.
- `hybridStimColours`
  - bool, optional
  - Flag for hybrid stimulus colors. Default is `True`.
- `yTickScale`
  - int, optional
  - Scale for the y-axis ticks. Default is `1`.
- `yTickPct`
  - bool, optional
  - Flag to format y-axis ticks as percentages. Default is `True`.
- `yPad`
  - float, optional
  - Padding for the y-axis. Default is `.05`.

#### Returns:

- `xAx`
  - object
  - x-axis object.
- `yAx`
  - object
  - y-axis object.

___

### plotAveragePostStimIntensity

`pyneurotrace.vis.plotAveragePostStimIntensity(data, hz, stimOffIdx, stimOnIdx, branches=None, title=None, secBefore=0, secAfter=3, savePath=None, **kwargs)`

Plots average post-stimulus intensity.

#### Parameters:

- `data`
  - array
  - Data to be plotted.
- `hz`
  - int
  - Sampling rate in Hz.
- `stimOffIdx`
  - int
  - Index of stimulus off.
- `stimOnIdx`
  - int, optional
  - Index of stimulus on. Default is `None`.
- `branches`
  - array, optional
  - Array of branch data. Default is `None`.
- `title`
  - str, optional
  - Title for the plot. Default is `None`.
- `secBefore`
  - int, optional
  - Seconds before the stimulus. Default is `0`.
- `secAfter`
  - int, optional
  - Seconds after the stimulus. Default is `3`.
- `savePath`
  - str, optional
  - Path to save the plot. Default is `None`.
- `**kwargs`
  - Additional keyword arguments for customization.

#### Returns:

- None

___

### plotAveragePostStimTransientParams

`pyneurotrace.vis.plotAveragePostStimTransientParams(dfof, hz, stimOffsets, secAfter, vizTrace=None, savePath=None)`

Plots average post-stimulus transient parameters.

#### Parameters:

- `dfof`
  - array
  - Data to be plotted.
- `hz`
  - int
  - Sampling rate in Hz.
- `stimOffsets`
  - array
  - Array of stimulus offsets.
- `secAfter`
  - int
  - Seconds after the stimulus.
- `vizTrace`
  - int, optional
  - Index of trace to visualize. Default is `None`.
- `savePath`
  - str, optional
  - Path to save the plot. Default is `None`.

#### Returns:

- `ax`
  - object
  - Axis object.

___

### plotPlanarStructure

`pyneurotrace.vis.plotPlanarStructure(tree, rootID, nodeXYZ, branchIDs=None, colors=None, title=None, flipY=False, scale=1, savePath=None, lineAlpha=0.8, flatten='Z', pixelMeters=None, palette='seaborn-dark-palette', bgColor='black')`

Plots a planar structure of the tree.

#### Parameters:

- `tree`
  - dict
  - Dictionary representing the tree structure.
- `rootID`
  - int
  - ID of the root node.
- `nodeXYZ`
  - array
  - Array of node locations (XYZ coordinates).
- `branchIDs`
  - array, optional
  - Array of branch IDs. Default is `None`.
- `colors`
  - array, optional
  - Colors for the branches. Default is `None`.
- `title`
  - str, optional
  - Title for the plot. Default is `None`.
- `flipY`
  - bool, optional
  - Flag to flip the y-axis. Default is `False`.
- `scale`
  - float, optional
  - Scale factor for the plot. Default is `1`.
- `savePath`
  - str, optional
  - Path to save the plot. Default is `None`.
- `lineAlpha`
  - float, optional
  - Alpha value for the lines. Default is `0.8`.
- `flatten`
  - str, optional
  - Axis to flatten. Default is `'Z'`.
- `pixelMeters`
  - float, optional
  - Pixel to meter conversion factor. Default is `None`.
- `palette`
  - str, optional
  - Color palette for the plot. Default is `'seaborn-dark-palette'`.
- `bgColor`
  - str, optional
  - Background color for the plot. Default is `'black'`.

#### Returns:

- `ax`
  - object
  - Axis object.

___

### plotBaseTipScatter

`pyneurotrace.vis.plotBaseTipScatter(baseTrace, tipTrace, title=None, **kwargs)`

Scatter plot of all samples from traces from a filopodia tip vs. filopodia base.

#### Parameters:

- `baseTrace`
  - array
  - Base trace data.
- `tipTrace`
  - array
  - Tip trace data.
- `title`
  - str, optional
  - Title for the plot. Default is `None`.
- `**kwargs`
  - Additional keyword arguments for customization.

#### Returns:

- `ax`
  - object
  - Axis object.

___

### planarAnimation

`pyneurotrace.vis.planarAnimation(tree, rootID, nodeXYZ, traceData, hz, flipY=False, stim=None, stimXY=(0,0), radius=0.005, savePath=None, scale=1, flatten='Z')`

Creates an animation of the planar structure with trace data.

#### Parameters:

- `tree`
  - dict
  - Dictionary representing the tree structure.
- `rootID`
  - int
  - ID of the root node.
- `nodeXYZ`
  - array
  - Array of node locations (XYZ coordinates).
- `traceData`
  - array
  - Array of trace data.
- `hz`
  - int
  - Sampling rate in Hz.
- `flipY`
  - bool, optional
  - Flag to flip the y-axis. Default is `False`.
- `stim`
  - array, optional
  - Array of stimulus data. Default is `None`.
- `stimXY`
  - tuple, optional
  - Stimulus position in the plot. Default is `(0,0)`.
- `radius`
  - float, optional
  - Radius of the nodes. Default is `0.005`.
- `savePath`
  - str, optional
  - Path to save the animation. Default is `None`.
- `scale`
  - float, optional
  - Scale factor for the plot. Default is `1`.
- `flatten`
  - str, optional
  - Axis to flatten. Default is `'Z'`.

#### Returns:

- `anim`
  - object
  - Animation object.

___


### kymograph

`pyneurotrace.vis.kymograph(kymoData, hz, smooth=False, title=None, widthInches=10, heightInches=60)`

Shows raw intensity across the pixel line scanned around the POI to check for drift.

#### Parameters:

- `kymoData`
  - array
  - Kymograph data.
- `hz`
  - int
  - Sampling rate in Hz.
- `smooth`
  - bool, optional
  - Flag to smooth the data. Default is `False`.
- `title`
  - str, optional
  - Title for the plot. Default is `None`.
- `widthInches`
  - int, optional
  - Width of the plot in inches. Default is `10`.
- `heightInches`
  - int, optional
  - Height of the plot in inches. Default is `60`.

#### Returns:

- `ax`
  - object
  - Axis object.

___


### scrollingAnimation

`pyneurotrace.vis.scrollingAnimation(data, hz, branches, stim, title, savePath)`

Creates a scrolling animation of the data.

#### Parameters:

- `data`
  - array
  - Data to be animated.
- `hz`
  - int
  - Sampling rate in Hz.
- `branches`
  - array
  - Array of branch data.
- `stim`
  - array
  - Array of stimulus data.
- `title`
  - str
  - Title for the animation.
- `savePath`
  - str
  - Path to save the animation.

#### Returns:

- `anim`
  - object
  - Animation object.



# Examples 
Please look at the jupyter notebooks in the GitHub repository for examples.
