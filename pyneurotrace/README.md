# PyNeuroTrace: Python code for Neural Time-series

## Visualisation

A number of options for visualising trace data are provided, including:

![Intensity plot](https://padster.github.io/pyNeuroTrace/img/pyntIntensity.png)

`plotIntensity(data, hz, branches, stim, title)`
> Generates a 2D intensity image, with rows being nodes and columns being samples of data. The intensity is normalised, ranging from black (lowest), to red, to yellow, and finally white (highest). Options include:
* Title for plot
* Groups (e.g. branch, neuron type, ...) shown on the left
* Stimuli shown below
* Path to save the file to
* Arguments passed through to matplotlib (e.g. vmin/vmax intensity bounds)

Note: Stimuli are all of 2D arrays of the form (S, 2),
where each row is a stimulus, and the two entries are the start and end sample.

&nbsp;

![Line plot](https://padster.github.io/pyNeuroTrace/img/pyntLines.png)

`plotLine(data, hz, branches, stim, title, split)`
> Generates line graphs for the provided traces. This accepts similar parameters to the intensity graph, as well as:
* Whether to draw lines over each-other, or separated along the Y axis

&nbsp;

![Planar structure plot](https://padster.github.io/pyNeuroTrace/img/pyntPlanar.png)

`plotPlanarStructure(tree, rootID, xyz)`
> Plots a pyNeuroTrace tree on a 2D plane, showing node positions, optional colours, and parent-child relationships. Options include:
* Title for plot
* Whether to flip the Y direction (up & down)
* Which axis to flatten along (default=Z)
* Individual colours for points
* Path to save the file to

The tree model required for this is a mapping:

    nodes = {
      <node id>: {
        'location': [x, y, z]
        'children': [list of child IDs]
      }
    }

&nbsp;

Other more specialised visualisations are also available, including stim response analysis and animations. Consult viz.py file.
