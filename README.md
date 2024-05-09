# PyNeuroTrace: Python code for Neural Timeseries

## Installation

This library can be installed directly from github, using pip:
```
pip install --upgrade "git+https://github.com/padster/pyNeuroTrace#egg=pyneurotrace&subdirectory=pyneurotrace"
```

## Vizualization

Probably the most useful section of pynt, there are a number of visualization functions provided to help display the trace data in easy to understand formats. For more details, and visual examples of what is available, please consult the [README](https://github.com/padster/pyNeuroTrace/tree/master/pyneurotrace/pyneurotrace) next to viz.py.

## Notebook utilities

Unrelated to neuron time series, but useful when using this regardless, PyNeuroTrace also provides a collection of tools for running analysis in Jupyter notebooks.

These include:

`notebook.filePicker` to selecting an existing file, `notebook.newFilePicker` for indicating a new path, and `notebook.folderPicker` for selecting an existing folder.
> These all open PyQT file dialogs, to make it easy for users to interact with the file system. Customisation options exist, for example prompt and default locations.

`showTabs(data, func, titles, progressBar=False)`
> Useful for performing analysis repeated multiple times across e.g. different neurons, experiments, conditions, ...etc. Given either a list, or dictionary, all items will be iterated over, and each individually drawn onto their own tab with the provided `func`.
The method provided expects `func(idx, key, value)`, where `idx` is the (0-based) index of the tab, `key` is the list/dictionary key, and `value` is what is to be processed.

## Processing Data

Common per-trace processing filters are provided within [filters.py](https://github.com/padster/pyNeuroTrace/blob/master/pyneurotrace/pyneurotrace/filters.py). These are all designed to take a numpy array of traces, with each row an independent trace, and all return a filtered array of the same size.

These include:

`filters.deltaFOverF0(traces, hz, t0, t1, t2)`
> Converts raw signal to the standard Delta-F-over-F0, using the technique given in [Jia et al, 2011](http://doi.org/10.1038/nprot.2010.169). The smoothing parameters (t0, t1, t2) are as described in the paper, all with units in seconds. Sample rate must also be provided to convert these to sample units.

`filters.okada(traces)`
> Reduces noise in traces by smoothing single peaks or valleys, as described in [Okada et al, 2016](https://doi.org/10.1371/journal.pone.0157595)



## Event Detection

Python implementations of the three algorithms discussed in our paper [Sakaki et al, 2018](https://doi.org/10.1109/EMBC.2018.8512983) for finding events within Calcium fluorescent traces.

`ewma(data, weight)`
> calculates the Exponentially-Weighted Moving Average for each trace, given how strongly to weight new points (vs. the previous average).

`cusum(data, slack)`
> calculates the Cumulative Sum for each trace, taking a slack parameter which controls how far from the mean the signal the signal needs to be to not be considered noise.

`matchedFilter(data, windowSize, A, tA, tB)`
> calculates the likelihood ratio for each sample to be the end of a window of expected transient shape, being a double exponential with amplitude A, rise-time tA, and decay-time tB (in samples).

The results of each of these three detection filters can then be passed through `thresholdEvents(data, threshold)`, to register detected events whenever the filter strength increases above the given threshold.

## Reading Data (lab-specific)

The code within this repository was designed to read data from experiments performed by the [Kurt Haas lab](http://www.haaslab.com/) at UBC. If you're from this lab, read below. If not, this part is probably not relevant, but fee free to ask if you'd be interested in loading your own data file formats.

A number of options for loading data files are available within [files.py](https://github.com/padster/pyNeuroTrace/blob/master/pyneurotrace/pyneurotrace/files.py), including:
* `load2PData(path)` takes an experiment output file (e.g STEP_5_EXPT.TXT) and returns ID, XYZ location, and raw intensity values for each node in the experiment.
* `loadMetadata(path)` takes a metadata file (e.g. rscan_metadata_step_5.txt) and returns the stimulus start/stop samples, as well as the sample rate for the experiment.
* `loadTreeStructure(path)` takes a tree structure file (e.g. interp-neuron-.txt) and returns the mapping of node IDs to tree information about that node (e.g. node type, children, ...).
