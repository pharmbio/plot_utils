"""CP Regression plots
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraysetops import isin
from sklearn.utils import check_consistent_length
from warnings import warn

# The following import sets seaborn etc if available 
from ._utils import get_fig_and_axis, set_chart_size, _plot_vline


def plot_calibration_curve_reg(error_rates,
    sign_vals,
    color = 'blue',
    ax = None,
    figsize = (10,8),
    chart_padding = None,
    title = None,
    tight_layout = True,
    **kwargs):

    check_consistent_length((sign_vals,error_rates))
    error_fig, ax = get_fig_and_axis(ax, figsize)

    # TODO Check sorting ?
    
    if chart_padding is None:
        chart_padding = (sign_vals[-1] - sign_vals[0]) * 0.025
    
    # Set chart range
    set_chart_size(ax,
        sign_vals,
        sign_vals,
        chart_padding)
    ax.plot(sign_vals, sign_vals, '--k', alpha=0.25, linewidth=1)

    ax.plot(sign_vals, error_rates, color=color, **kwargs)

    # Print some labels and title if appropriate
    if len(ax.yaxis.get_label().get_text()) == 0:
        ax.set_ylabel("Error rate")
    if len(ax.xaxis.get_label().get_text()) == 0:
        ax.set_xlabel("Significance")
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})

    if tight_layout:
        error_fig.tight_layout()
    
    return error_fig



def plot_pred_widths(pred_widths,
    sign_vals,
    color = 'blue',
    ax = None,
    figsize = (10,8),
    chart_padding = 0.025,
    title = None,
    y_label = 'Median Prediction interval width',
    tight_layout = True,
    **kwargs):
    
    check_consistent_length((sign_vals, pred_widths))
    error_fig, ax = get_fig_and_axis(ax, figsize)

    # Set chart range
    set_chart_size(ax,
        sign_vals,
        pred_widths,
        chart_padding)

    ax.plot(sign_vals, pred_widths, color=color, **kwargs)

    # Print some labels and title if appropriate
    if (len(ax.yaxis.get_label().get_text()) == 0) and (y_label is not None):
        ax.set_ylabel(y_label)
    if len(ax.xaxis.get_label().get_text()) == 0:
        ax.set_xlabel("Significance")
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})

    if tight_layout:
        error_fig.tight_layout()
    
    return error_fig



def plot_pred_intervals(y_true,
    predictions,
    ax = None,
    figsize = (10,8),

    correct_color = 'blue',
    correct_marker = 'o',
    correct_alpha = 0.75,
    correct_ci = 'gray',
    correct_ci_alpha = 0.7,

    incorrect_color = 'red',
    incorrect_marker = 'o',
    incorrect_alpha = 0.75,
    incorrect_ci ='gray',
    incorrect_ci_alpha = 0.7,

    line_cap = 1,
    chart_padding = 0.025,
    title = None,
    x_label = "Predicted examples",
    y_label = None,
    
    tight_layout = True,
    **kwargs):
    """**Regression**
    """

    check_consistent_length((y_true, predictions))
    ys = y_true.copy()
    ys.shape = (ys.shape[0],1)

    fig, ax = get_fig_and_axis(ax, figsize)
    
    joined = np.hstack((ys,predictions))
    # sorted by the true labels
    joined = joined[joined[:,0].argsort()]
    xs = np.arange(0,len(joined),1).reshape((len(joined),1))
    joined = np.hstack((joined,xs))

    # find the correct and incorrect predictions
    corr_ind = (joined[:,1]<= joined[:,0]) & (joined[:,0]<= joined[:,2])
    incorr_ind = ~corr_ind
    # Columns in the gathered result matrix
    y_ind = 0
    ci_min = 1
    ci_max = 2
    x_ind = 3
    
    # Set the chart size
    set_chart_size(ax,
        [0,len(y_true)],
        [np.max(predictions), np.max(y_true), np.min(predictions), np.min(y_true)],
        chart_padding)

    # VERTICAL INTERVALS
    # plot corrects
    _plot_vline(x = joined[corr_ind,x_ind],
        y_min = joined[corr_ind,ci_min],
        y_max = joined[corr_ind,ci_max],
        ax = ax,
        color = correct_ci,
        alpha = correct_ci_alpha,
        line_cap=line_cap)
    
    # plot incorrect intervals
    _plot_vline(x = joined[incorr_ind,x_ind],
        y_min = joined[incorr_ind,ci_min],
        y_max = joined[incorr_ind,ci_max],
        ax = ax,
        color = incorrect_ci,
        alpha = incorrect_ci_alpha,
        line_cap=line_cap)

    # plot the true values
    # corrects
    ax.plot(joined[corr_ind,x_ind],
        joined[corr_ind,y_ind], 
        label = 'Correct',
        marker = correct_marker,
        alpha = correct_alpha,
        lw = 0,
        color = correct_color)
    # incorrects
    ax.plot(joined[incorr_ind,x_ind],
        joined[incorr_ind,y_ind],
        label = 'Incorrect',
        marker = incorrect_marker,
        alpha = incorrect_alpha,
        lw = 0,
        color = incorrect_color)

    # Print some labels and title if appropriate
    if (len(ax.yaxis.get_label().get_text()) == 0) and (y_label is not None):
        ax.set_ylabel(y_label)
    if (len(ax.xaxis.get_label().get_text()) == 0) and x_label is not None:
        ax.set_xlabel(x_label)
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})

    if tight_layout:
        fig.tight_layout()
    
    return fig
