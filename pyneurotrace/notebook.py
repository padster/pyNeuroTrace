# Set up environment
import ipywidgets
import matplotlib.pyplot as plt
import os

from PyQt5 import QtWidgets
from IPython.display import display, HTML
from tqdm import tqdm_notebook

# Show a dialog selection box to the user so they can pick one file.
def filePicker(prompt="Select file", extension="", defaultPath='.'):
    """
    Show a dialog selection box to the user to pick one file.

    Parameters
    ----------
    prompt : str, optional
        The prompt message for the dialog. Default is `"Select file"`.
    extension : str, optional
        File extension filter. Default is `""`.
    defaultPath : str, optional
        Default path to start the file picker. Default is `'.'`.

    Returns
    -------
    fname : str
        Path of the selected file.
    """

    if os.path.isfile(defaultPath):
        defaultPath = os.path.dirname(defaultPath)
    app = QtWidgets.QApplication([])
    fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, prompt, defaultPath, extension)
    return str(fname)
    
# Dialog for user to select path of *new* save file
def newFilePicker(prompt="New file", defaultPath='.'):
    """
    Show a dialog for the user to select the path of a new save file.

    Parameters
    ----------
    prompt : str, optional
        The prompt message for the dialog. Default is `"New file"`.
    defaultPath : str, optional
        Default path to start the file picker. Default is `'.'`.

    Returns
    -------
    fname : str
        Path of the new save file.
    """

    if os.path.isfile(defaultPath):
        defaultPath = os.path.dirname(defaultPath)
    app = QtWidgets.QApplication([])
    fname, _ = QtWidgets.QFileDialog.getSaveFileName(None, prompt, defaultPath)
    return str(fname)

# Show a folder selection dialog to the user and return the path.
def folderPicker(prompt="Output Folder", defaultPath='.'):
    """
    Show a folder selection dialog to the user and return the path.

    Parameters
    ----------
    prompt : str, optional
        The prompt message for the dialog. Default is `"Output Folder"`.
    defaultPath : str, optional
        Default path to start the folder picker. Default is `'.'`.

    Returns
    -------
    fname : str
        Path of the selected folder.
    """

    if os.path.isfile(defaultPath):
        defaultPath = os.path.dirname(defaultPath)
    app = QtWidgets.QApplication([])
    fname = QtWidgets.QFileDialog.getExistingDirectory(None, prompt, defaultPath)
    return str(fname)
    
# Display data in multiple tabs in the same output area.
# Performed as a mapping, using dictionary/list and a custom map function.
def showTabs(data, showOnTabFunc, titles=None, progressBar=False):
    """
    Display data in multiple tabs in the same output area.

    Parameters
    ----------
    data : list or dict
        Data to be displayed in tabs.
    showOnTabFunc : function
        Function to display data on each tab.
    titles : list, optional
        List of titles for the tabs. Default is `None`.
    progressBar : bool, optional
        Flag to show a progress bar. Default is `False`.

    Returns
    -------
    None
        Displays data in tabs within the notebook.
    """
    dataIsList = isinstance(data, list)
    
    progressBarFunc = tqdm_notebook if progressBar else (lambda x: x)
    
    # Default titles if none provided.
    if titles is None:
        if dataIsList:
            titles = [str(i) for i in range(len(data))]
        else:
            titles = ['%s: Step %d' % ('P' if value['planar'] else 'V', key) for (key, value) in data.items()]
    
    pbar = None
    if progressBar:
        pbar = tqdm_notebook(total=len(data))
    
    # Create tabs, set their titles.
    allTabs = [ipywidgets.Output() for i in range(len(data))]
    tab = ipywidgets.Tab(children = allTabs)
    for i, title in enumerate(titles):
        tab.set_title(i, title)
    display(tab)

    # And map each data point into a display function.
    if dataIsList:
        for idx, value in enumerate(data):
            with allTabs[idx]:
                if showOnTabFunc(idx, idx, value) is None:
                    plt.show()
                if pbar is not None:
                    pbar.update(1)
    else:
        for idx, (key, value) in progressBarFunc(enumerate(data.items())):
            with allTabs[idx]:
                if showOnTabFunc(idx, key, value) is None:
                    plt.show()
                if pbar is not None:
                    pbar.update(1)
    
    if pbar is not None:
        pbar.close()
