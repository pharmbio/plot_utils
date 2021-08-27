"""CP Regression plots
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_consistent_length

# The following import sets seaborn etc if available 
from ._utils import get_fig_and_axis


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
    
    # Set chart range and add dashed diagonal
    ax.axis([sign_vals[0]-chart_padding, sign_vals[-1]+chart_padding, sign_vals[0]-chart_padding, sign_vals[-1]+chart_padding]) 
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
    y_min = np.min(pred_widths)
    y_max = np.max(pred_widths)
    chart_padding_x = (sign_vals[-1] - sign_vals[0]) * chart_padding
    chart_padding_y = (y_max - y_min)*chart_padding
    
    # Set chart range 
    # [x_min,x_max,y_min,y_max]
    ax.axis([np.min(sign_vals)-chart_padding_x,
        np.max(sign_vals)+chart_padding_x, 
        y_min-chart_padding_y, 
        y_max+chart_padding_y]) 

    ax.plot(sign_vals, pred_widths, color=color, **kwargs)

    # Print some labels and title if appropriate
    if (len(ax.yaxis.get_label().get_text()) == 0) & (y_label is not None):
        ax.set_ylabel(y_label)
    if len(ax.xaxis.get_label().get_text()) == 0:
        ax.set_xlabel("Significance")
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})

    if tight_layout:
        error_fig.tight_layout()
    
    return error_fig

