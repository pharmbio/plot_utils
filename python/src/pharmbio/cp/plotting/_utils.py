import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
from warnings import warn
import math
import numpy as np
from sklearn.utils import check_consistent_length
import pandas as pd


_using_seaborn = False
# Try to import sns as they create somewhat nicer plots
try:
    import seaborn as sns
    sns.set_theme()
    logging.debug('Using Seaborn plotting defaults')
    _using_seaborn = True
except ImportError as e:
    logging.debug('Seaborn not available - using Matplotlib defaults')
    pass 

__DEFAULT_FIG_SIZE = (10,8)

def get_fig_and_axis(ax=None, figsize = __DEFAULT_FIG_SIZE):
    '''Function for instantiating a Figure / axes object

    Validates the input parameters, instantiates a Figure and axes if not
    sent as a parameter to the plotting function.

    Parameters
    ----------
    ax : matplotlib axes or None
        An existing axes object that the user may send,
        to write the plot in
    
    figsize : float or (float, float)
        A figure size to generate. If a single number is given, the figure will
        be a square with each side of that size.
    
    Returns
    -------
    fig : matplotlib Figure
    
    ax : matplotlib Axes

    Raises
    ------
    TypeError
        If `figsize` or `ax` is of invalid type 
    '''
    # Override if figsize is None
    if figsize is None:
        figsize = __DEFAULT_FIG_SIZE
    
    if ax is None:
        # No current axes, create a new Figure
        if isinstance(figsize, (int, float)):
            fig = plt.figure(figsize = (figsize, figsize))
        elif isinstance(figsize, tuple):
            fig = plt.figure(figsize = figsize)
        else:
            raise TypeError('parameter figsize must either be float or (float, float), was: {}'.format(type(figsize)))
        # Add an axes spanning the entire Figure
        ax = fig.add_subplot(111)
    elif not isinstance(ax, mpl.axes.Axes):
        raise TypeError('parameter ax must be either None or a matplotlib.axes object, was: {}'.format(type(ax)))
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

def _set_label_if_not_set(ax, text, x_axis=True):
    """Sets the x or y axis labels if not set before

    Checks if the axis label has been set previously, 
    does not overwrite any existing label

    Parameters
    ----------
    ax : matplotlib Axes
        The Axes to write to, must not be None
    text : str or None
        Optional text to write to the label, function simply
        returns if None is given
    x_axis : bool
        If it the x axis (True) or y axis (False) that should be written
    """
    if text is None or not isinstance(text,str):
        return
    if x_axis:
        if len(ax.xaxis.get_label().get_text()) == 0:
            # Label not set before
            ax.set_xlabel(text)
    else:
        if len(ax.yaxis.get_label().get_text()) == 0:
            ax.set_ylabel(text)

def _set_title(ax, title=None):
    """Sets the title if given and not previously set

    """
    if title is None:
        return
    if len(ax.get_title()) == 0:
        if not isinstance(title,str):
            title = str(title)
        ax.set_title(title,fontdict={'fontsize':'x-large'})

def _set_chart_size(ax, 
    x_vals, 
    y_vals, 
    padding = 0.025, 
    flip_x=False, 
    flip_y=False):
    """Sets the chart drawing limits

    Handles padding and finds the max and min values

    Parameters
    ----------
    ax : matplotlib Axes
    
    x_vals, y_vals : array_like
        The values to find limits of, can optionally be calculated prior to this function and sent as a list of e.g. [min,max] to save computation time
    
    padding : float or (float,float), default = 0.025
        Padding as percentage of the value range, if a single value is given the same padding is applied to both axes. For two values, the first is applied to x-axes and the second to the y-axes.
    
    flip_x : bool, default False
        If the x-axes should display significance level (`False`) or confidence (`True`)
    
    flip_y : bool, default False
        If the y-axes should display error-rate (`False`) or accuracy (`True`)
    """
    x_min,x_max = np.min(x_vals), np.max(x_vals)
    y_min,y_max = np.min(y_vals), np.max(y_vals)
    if flip_x:
        x_min,x_max = 1-x_max, 1-x_min
    if flip_y:
        y_min,y_max = 1-y_max, 1-y_min
    x_w, y_w = x_max - x_min, y_max - y_min

    if padding is None:
        x_padd = 0
        y_padd = 0
    elif isinstance(padding,float):
        x_padd = x_w*padding
        y_padd = y_w*padding
    elif isinstance(padding,tuple) or isinstance(padding,list):
        if len(padding) == 1:
            x_padd = x_w*padding[0]
            y_padd = y_w*padding[0]
        elif len(padding) > 1:
            x_padd = x_w*padding[0]
            y_padd = y_w*padding[1]
        else:
            raise TypeError('padding should be a float or list/tuple of 2 floats')
    else:
        raise TypeError('padding should be a float or list/tuple of 2 floats, got {}'.format(type(padding)))
    
    ax.axis([x_min-x_padd, 
        x_max+x_padd, 
        y_min-y_padd, 
        y_max+y_padd])

def _plot_vline(x,y_min,y_max,ax,color='gray',alpha=0.7,line_cap=None):
    # The vertical line itself
    ax.vlines(x = x,
        ymin = y_min,
        ymax = y_max,
        color = color,
        alpha = alpha)
    if (isinstance(line_cap,bool) and not line_cap) or line_cap is None:
        return
    elif (isinstance(line_cap,bool) and line_cap) or line_cap == 1:
        m_u = m_l = '_'
    elif line_cap == 2:
        m_u,m_l = 6,7 
    elif isinstance(line_cap,str):
        m_u, m_l = line_cap,line_cap
    else:
        warn('Invalid argument for line_cap {}, falling back to not printing any'.format(line_cap))
    # Upper 'cap'
    ax.plot(x, y_max,
        marker = m_u,
        lw = 0,
        alpha = alpha,
        color = color)
    # Lower 'cap'
    ax.plot(x, y_min,
        marker = m_l,
        lw = 0,
        alpha = alpha,
        color = color)