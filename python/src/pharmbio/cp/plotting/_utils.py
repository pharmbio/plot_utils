import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
from warnings import warn
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


def set_chart_size(ax, x_vals, y_vals, padding = 0.025):
    x_min = np.min(x_vals)
    x_max = np.max(x_vals)
    y_min = np.min(y_vals)
    y_max = np.max(y_vals)
    
    if isinstance(padding,float):
        x_padd = (x_max - x_min)*padding
        y_padd = (y_max - y_min)*padding
    elif isinstance(padding,tuple) or isinstance(padding,list):
        if len(padding) == 1:
            x_padd = (x_max - x_min)*padding[0]
            y_padd = (y_max - y_min)*padding[0]
        elif len(padding) == 2:
            x_padd = (x_max - x_min)*padding[0]
            y_padd = (y_max - y_min)*padding[1]
    else:
        warn('Padding only allowed as None, float or (x_padd,y_padd), falling back to default 2.5%')
    # If no correct given, set default 2.5%
    if x_padd is None:
        x_padd = (x_max - x_min)*0.025
        y_padd = (y_max - y_min)*0.025

    # [x_min,x_max,y_min,y_max]
    ax.axis([x_min - x_padd, x_max + x_padd, y_min-y_padd, y_max+y_padd])

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
        m_u,m_l = 6,7 #'^', 'v'
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