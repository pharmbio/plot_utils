import numpy as np
from ._utils import _set_chart_size

def add_calib_curve(ax, 
    error_rates,
    sign_vals,
    legend = None,
    color = 'k',
    std_orientation = True,
    chart_padding = 0.025,
    set_chart_size = False,
    plot_expected = True,
    zorder = None,
    **kwargs):
    """Utility function for adding a single line to an Axes

    Solves setting chart size, error rate vs significance or Accuracy vs confidence

    Returns
    -------
    (x_label, y_label) : the str labels for what is plotted
    """
    
    if std_orientation:
        x_label='Significance'
        y_label='Error rate'
        xs = sign_vals
        ys = error_rates
    else:
        x_label='Confidence'
        y_label='Accuracy'
        xs = 1 - sign_vals
        ys = 1 - error_rates
    
    if set_chart_size:
        min = np.min([np.min(ys), np.min(xs)])
        max = np.max([np.max(ys), np.max(xs)])
        _set_chart_size(ax,
            [min,max],
            [min,max],
            chart_padding)
    
    if plot_expected:
        ax.plot(xs, xs, '--', color='gray', linewidth=1) # alpha=0.25,
    
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