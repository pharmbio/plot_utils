"""CP Regression plots
"""
# from os import stat_result
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# from numpy.lib.arraysetops import isin
from sklearn.utils import check_consistent_length
# from warnings import warn

from pharmbio.cp.utils import to_numpy1D, to_numpy2D

# The following import sets seaborn etc if available 
from ._utils import get_fig_and_axis, _set_chart_size, _plot_vline,_set_label_if_not_set, _set_title
from ._common import add_calib_curve


def plot_calibration_curve_reg(error_rates,
    sign_vals,
    color = 'blue',
    ax = None,
    figsize = (10,8),
    chart_padding = 0.025,
    title = None,
    tight_layout = True,
    x_significance = True,
    y_error = True,
    **kwargs):
    """**Regression** 
    """
    check_consistent_length((sign_vals,error_rates))
    error_fig, ax = get_fig_and_axis(ax, figsize)
    ax.set_aspect('equal','box')

    (x_lab, y_lab) = add_calib_curve(ax,error_rates,sign_vals,
        plot_expected=True,
        chart_padding=chart_padding,
        set_chart_size=True)
    
    # Print some labels and title if appropriate
    _set_label_if_not_set(ax,x_lab,True)
    _set_label_if_not_set(ax,y_lab,False)
    _set_title(ax,title)

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
    x_significance = True,
    **kwargs):
    
    check_consistent_length((sign_vals, pred_widths))
    fig, ax = get_fig_and_axis(ax, figsize)

    if x_significance:
        xs = sign_vals
        x_label = 'Significance'
    else:
        xs = 1 - sign_vals
        x_label = 'Confidence'

    # Set chart range
    _set_chart_size(ax,
        xs,
        pred_widths,
        chart_padding)

    ax.plot(xs, pred_widths, color=color, **kwargs)

    # Print some labels and title if appropriate
    _set_label_if_not_set(ax,x_label,True)
    _set_label_if_not_set(ax,y_label,False)
    # if (len(ax.yaxis.get_label().get_text()) == 0) and (y_label is not None):
    #     ax.set_ylabel(y_label)
    # if len(ax.xaxis.get_label().get_text()) == 0:
    #     ax.set_xlabel("Significance")
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})

    if tight_layout:
        fig.tight_layout()
    
    return fig



def plot_pred_intervals(y_true,
    predictions,
    ax = None,
    figsize = (10,8),

    correct_color = 'blue',
    correct_marker = 'o',
    correct_alpha = 0.75,
    correct_ci = 'gray',
    correct_ci_alpha = 0.7,
    correct_label = 'Correct',

    incorrect_color = 'red',
    incorrect_marker = 'o',
    incorrect_alpha = 0.75,
    incorrect_ci ='gray',
    incorrect_ci_alpha = 0.7,
    incorrect_label = 'Incorrect',

    line_cap = 1,
    chart_padding = 0.025,
    title = None,
    x_label = 'Predicted examples',
    y_label = None,
    
    x_start_index = 0,
    tight_layout = True,
    **kwargs):
    """**Regression**
    """

    check_consistent_length((y_true, predictions))
    ys = to_numpy1D(y_true, "y_true",return_copy=True)
    preds = to_numpy2D(predictions,"predictions")

    fig, ax = get_fig_and_axis(ax, figsize)
    
    # sorted by the true labels
    sort_order = ys.argsort()
    ys = ys[sort_order]
    preds = preds[sort_order]
    xs = np.arange(x_start_index,x_start_index+len(ys),1)

    # find the correct and incorrect predictions
    corr_ind = (preds[:,0] <= ys) & (ys<= preds[:,1])
    incorr_ind = ~corr_ind
    
    # Set the chart size
    _set_chart_size(ax,
        [0,len(y_true)],
        [np.max(predictions), np.max(y_true), np.min(predictions), np.min(y_true)],
        chart_padding)

    # VERTICAL INTERVALS
    # plot corrects
    _plot_vline(x = xs[corr_ind],
        y_min = preds[corr_ind,0],
        y_max = preds[corr_ind,1],
        ax = ax,
        color = correct_ci,
        alpha = correct_ci_alpha,
        line_cap=line_cap)
    
    # plot incorrect intervals
    _plot_vline(x = xs[incorr_ind],
        y_min = preds[incorr_ind,0],
        y_max = preds[incorr_ind,1],
        ax = ax,
        color = incorrect_ci,
        alpha = incorrect_ci_alpha,
        line_cap=line_cap)

    # plot the true values
    # corrects
    ax.plot(xs[corr_ind],
        ys[corr_ind], 
        label = correct_label,
        marker = correct_marker,
        alpha = correct_alpha,
        lw = 0,
        color = correct_color)
    # incorrects
    ax.plot(xs[incorr_ind],
        ys[incorr_ind],
        label = incorrect_label,
        marker = incorrect_marker,
        alpha = incorrect_alpha,
        lw = 0,
        color = incorrect_color)

    # Print some labels and title if appropriate
    _set_label_if_not_set(ax, y_label, x_axis=False)
    _set_label_if_not_set(ax, x_label, x_axis=True)

    # if (len(ax.yaxis.get_label().get_text()) == 0) and (y_label is not None):
    #     ax.set_ylabel(y_label)
    # if (len(ax.xaxis.get_label().get_text()) == 0) and x_label is not None:
    #     ax.set_xlabel(x_label)
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})

    if tight_layout:
        fig.tight_layout()
    
    return fig
