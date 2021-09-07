import numpy as np
from numpy.core.fromnumeric import sort
from ._utils import _set_chart_size

def add_calib_curve(ax, 
    sign_vals,
    error_rates,
    legend = None,
    color = 'k',
    flip_x = False,
    flip_y = False,
    chart_padding = 0.025,
    set_chart_size = False,
    plot_expected = True,
    zorder = None,
    **kwargs):
    """Utility function for adding a single line to an Axes

    Solves setting chart size, error rate vs significance or Accuracy vs confidence
    Parameters
    ----------
    ax : matplotlib Axes
        Axes to plot in
    
    sign_vals : 1d ndarray

    error_rates : 1d ndarray

    legend : str or None
        An optional legend to add to the plotted values
    
    color : str or matplotlib recognized color-input
    
    flip_x : bool, default False
        If the x-axes should display significance level (`False`) or confidence (`True`)
    
    flip_y : bool, default False
        If the y-axes should display error-rate (`False`) or accuracy (`True`)
    
    chart_padding : float, (float,float) or None
        padding added to the chart-area outside of the min and max values found in data. If two values the first value will be used as x-padding and second y-padding. E.g. 0.025 means 2.5% on both sides

    set_chart_size : bool, default False
        If the chart size should be set
    
    plot_expected : bool, default True
        If the dashed 'expected' error/accuracy line should be plotted
    
    zorder : float or None
        An zorder (controlling the order of plotted artists), a higher zorder means plotting on-top of other objects plotted
    
    Returns
    -------
    (x_label, y_label) : the str labels for what is plotted
    """

    # Handle x-axis
    if flip_x:
        x_label, xs = 'Confidence', 1 - np.array(sign_vals)
    else:
        x_label, xs = 'Significance', sign_vals
    # Handle y-axis
    if flip_y:
        y_label, ys = 'Accuracy', 1 - np.array(error_rates)
    else:
        y_label, ys = 'Error rate', error_rates
    
    if set_chart_size:
        min = np.min([np.min(ys), np.min(xs)])
        max = np.max([np.max(ys), np.max(xs)])
        _set_chart_size(ax,
            [min,max],
            [min,max],
            chart_padding)
    
    if plot_expected:
        if flip_x == flip_y:
            # If both flipped or both normal
            ax.plot(xs, xs, '--', color='gray', linewidth=1)
        else:
            ax.plot(xs, 1-np.array(xs), '--', color='gray', linewidth=1)
    
    # If there's an explicit zorder - add it to a new dict
    if zorder is not None:
        kwargs = dict(kwargs, zorder=zorder)

    # Plot the values
    if color is not None:
        ax.plot(xs, ys, 
            label=(legend if legend is not None else y_label), 
            color=color,
            **kwargs)

    return (x_label,y_label)