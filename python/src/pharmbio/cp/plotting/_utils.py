import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import math
import numpy as np
from sklearn.utils import check_consistent_length
import pandas as pd


using_seaborn = False
# Try to import sns as they create somewhat nicer plots
try:
    import seaborn as sns
    sns.set()
    logging.debug('Using Seaborn plotting defaults')
    using_seaborn = True
except ImportError as e:
    logging.debug('Seaborn not available - using Matplotlib defaults')
    pass 


def get_fig_and_axis(ax=None, figsize = (10,8)):
    '''Function for instantiating a Figure / axes object

    Validates the input parameters, instantiates a Figure and axes if not
    sent as a parameter to the plotting function.

    Parameters
    ----------
    ax : matplotlib axes
        An existing axes object or None
    
    figsize : float or (float, float)
        A figure size to generate. If a single number is given, the figure will
        be a square with each side of that size.
    
    Returns
    -------
    fig : Figure
    
    ax : matplotlib axes
    '''
    
    if ax is None:
        # No current axes, create a new Figure
        if isinstance(figsize, (int, float)):
            fig = plt.figure(figsize = (figsize, figsize))
        elif isinstance(figsize, tuple):
            fig = plt.figure(figsize = figsize)
        else:
            raise TypeError('parameter figsize must either be float or (float, float), was: {}'.fromat(type(figsize)))
        # Add an axes spanning the entire Figure
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    
    return fig, ax

def cm_as_list(cm, default_cm):
    if cm is None:
        return default_cm
    elif isinstance(cm, mpl.colors.ListedColormap):
        return list(cm.colors)
    elif isinstance(cm, list):
        return cm
    else:
        return [cm]
