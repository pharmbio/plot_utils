from collections.abc import Iterable
import matplotlib as mpl
import numpy as np
from sklearn.utils import check_consistent_length
from numpy.core.fromnumeric import sort
from ._utils import get_fig_and_axis, cm_as_list, _set_title, _set_label_if_not_set,_set_chart_size
from ..utils import to_numpy2D, validate_sign, to_numpy1D

_default_color_map = list(mpl.rcParams['axes.prop_cycle'].by_key()['color'])

def update_plot_settings(theme_style = 'ticks', context = 'notebook', font_scale = 1):
    '''
    Update the global plot-settings, requires having seaborn available.

    This is simply a convenience wrapper of the functions `seaborn.set_context` and `seaborn.set_theme`. If seaborn is not available, this function 
    will not make any alterations 
    '''
    try:
        import seaborn as sns
        sns.set_context(context=context)
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style=theme_style, rc=custom_params, font_scale=font_scale)
    except ImportError as e:
        pass

def add_calib_curve(ax, 
    sign_vals,
    error_rates,
    label = None,
    color = 'k',
    flip_x = False,
    flip_y = False,
    chart_padding = 0.025,
    set_chart_size = False,
    plot_expected = True,
    zorder = None,
    **kwargs):
    """Utility function for adding a single line to an Axes

    Solves setting chart size, error rate vs significance or Accuracy vs confidence
    Parameters
    ----------
    ax : matplotlib Axes
        Axes to plot in
    
    sign_vals : 1d ndarray

    error_rates : 1d ndarray

    label : str or None
        An optional label to add to the plotted values
    
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
            label=(label if label is not None else y_label), 
            color=color,
            **kwargs)

    return (x_label,y_label)

def plot_calibration(sign_vals = None, 
    error_rates = None, 
    error_rates_sd = None, 
    conf_vals = None,
    accuracy_vals = None,
    accuracy_sd = None,
    labels = None,
    ax = None, 
    figsize = (10,8),
    chart_padding=0.025,
    cm = None,
    flip_x = False,
    flip_y = False,
    title=None,
    tight_layout=True,
    plot_expected = True,
    sd_alpha = .3,
    **kwargs):
    '''
    **Classification and regression ** - Create a calibration plot from computed values

    This function creates a plot of calibration curves, given precomputed values for either
    accuracy or error rates given significance or confidence. Note that either accuracy _or_ error_rate
    can be given (not both) and significance _or_ confidence (not both) must be given. Additionally 
    a standard-deviation "_sd" parameters can be given which will be displayed with the same color (according to `cm` argument)
    behind the error/accuracy values.

    Parameters
    ----------
    sign_vals : a 1D Iterable, default None
        Significance values for the corresponding accuracy/error rates
    
    error_rates : 1D or 2D list like, default None
        Error rates, either a single (e.g. overall value) or multiple (i.e. one for class)

    error_rates_sd : 1D or 2D list like, default None
        Standard deviations for the `error_rates`, used for plotting `error_rate +/- SD` areas

    conf_vals : a 1D Iterable, default None
        Confidence values for the corresponding accuracy/error rates
    
    accuracy_vals : 1D or 2D list like, default None
        Accuracy values, either a single (e.g. overall value) or multiple (i.e. one for class)

    accuracy_sd : 1D or 2D list like, default None
        Standard deviations for the `accuracy_vals`, used for plotting `accuracy +/- SD` areas

    labels : list of str, optional
        Descriptive labels for the input, for regression input it can be a single str, for classification
        the number of columns in `error_rates` or `accuracy_vals` should match the number of labels

    ax : matplotlib Axes, optional
        An existing matplotlib Axes to plot in (default None)

    figsize : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    chart_padding : float, (float,float) or None, default 0.025
        padding added to the chart-area outside of the min and max values found in data. 
        If two values the first value will be used as x-padding and second y-padding. E.g. 0.025 means 2.5% on both sides

    cm : color, list of colors or ListedColorMap, optional
        The colors to use. First color will be for class 0, second for class 1, ..
    
    flip_x : bool, default False
        If the x-axes should be 'flipped', i.e. if `sign_vals` was given the default is to display "Significance" on the x-axis, 
        this will flip it and display "Confidence instead", or vise versa in case conf_vals was given
    
    flip_y : bool, default False
        If the y-axes should be 'flipped', i.e. if accuracy_vals was given, the default is to display "Accuracy" on the y-axis, 
        this will flip it and instead display "Error rate". And vise versa if `error_rates` is given.

    title : str, optional
        Optional title that will be printed in 'x-large' font size (default None)

    tight_layout : bool, optional
        Set `tight_layout` on the matplotlib Figure object
    
    plot_expected : bool, optional
        Plot the diagonal, representing the expected error/accuracy (default `True`)

    **kwargs : dict, optional
        Keyword arguments, passed to matplotlib
    
    Returns
    -------
    fig : Figure
        matplotlib.figure.Figure object
    
    See Also
    --------
    matplotlib.colors.ListedColormap
    '''

    colors = cm_as_list(cm, _default_color_map)

    # Validate either significance or confidence values were given
    if (sign_vals is None and conf_vals is None) or (sign_vals is not None and conf_vals is not None):
        raise ValueError('Either sign_vals or conf_vals must be given (not both)')
    # Validate either error_rates or accuracy are given
    if (error_rates is None and accuracy_vals is None) or (error_rates is not None and accuracy_vals is not None):
        raise ValueError('Either error_rates or accuracy_vals must be given (not both)')
    
    # ======================================================
    # Deduce the x-values + label
    if sign_vals is not None:
        # Using sign input
        validate_sign(sign_vals)
        if len(sign_vals) < 2:
            raise ValueError('Must have at least 2 significance values to plot a calibration curve')
        x_lab = 'Confidence' if flip_x else 'Significance'
        x_vals = 1 - to_numpy1D(sign_vals,'sign_vals') if flip_x else to_numpy1D(sign_vals,'sign_vals')
    else:
        # Using conf input
        validate_sign(conf_vals)
        if len(conf_vals) < 2:
            raise ValueError('Must have at least 2 confidence values to plot a calibration curve')
        x_lab = 'Significance' if flip_x else 'Confidence'
        x_vals = 1 - to_numpy1D(conf_vals,'conf_vals') if flip_x else to_numpy1D(conf_vals,'conf_vals')
    
    # ======================================================
    # Deduce the y-values + label
    if error_rates is not None:
        # Using error rate input
        y_lab = 'Accuracy' if flip_y else 'Error rate'
        y_vals = 1 - to_numpy2D(error_rates,'error_rates', unravel=True, min_num_cols=1) if flip_y else to_numpy2D(error_rates,'error_rates', unravel=True, min_num_cols=1)
        y_SD = None if error_rates_sd is None else to_numpy2D(error_rates_sd,'error_rates_sd', unravel=True, min_num_cols=1)
    else:
        # Using accuracy input
        y_lab = 'Error rate' if flip_y else 'Accuracy'
        y_vals = 1 - to_numpy2D(accuracy_vals,'accuracy_vals', unravel=True, min_num_cols=1) if flip_y else to_numpy2D(accuracy_vals,'accuracy_vals', unravel=True, min_num_cols=1)
        y_SD = None if accuracy_sd is None else to_numpy2D(accuracy_sd,'accuracy_sd', unravel=True, min_num_cols=1)

    # Create the figure and axis to plot in
    error_fig, ax = get_fig_and_axis(ax, figsize)

    # Create labels if not set
    if labels is None:
        labels = ['Overall']
        if y_vals.shape[1]>1:
            labels += ['Label {}'.format(i-1) for i in range(1,y_vals.shape[1])]
    elif isinstance(labels,Iterable):
        if len(labels) < y_vals.shape[1]:
            raise ValueError('Invalid number of labels given, should be {}'.format(y_vals.shape[1]))
        if isinstance(labels,str):
            # str is iterable, which forces us to special case this
            if y_vals.shape[1]==1:
                labels = [labels]
            else:
                raise ValueError("Invalid 'labels' argument, should be a list of labels of length {}".format(y_vals.shape[1]))

    elif y_vals.shape[1]==1:
        # Single line to be plotted, wrap in a list
        labels = [labels]
    else:
        raise ValueError("Invalid 'labels' argument, should be a list of labels")


    # Check consistent length of x and y points
    check_consistent_length(y_vals,x_vals)

    # Set the chart size, flipping handled before, set to False
    _set_chart_size(ax,x_vals,y_vals,
        padding=chart_padding,
        flip_x=False,
        flip_y=False)
    
    # Plot the expected
    if plot_expected:
        if flip_x == flip_y:
            # If both flipped or both normal
            ax.plot(x_vals, x_vals, '--', color='gray', linewidth=1)
        else:
            ax.plot(x_vals, 1-np.array(x_vals), '--', color='gray', linewidth=1)
    
    z_offset = 20
    # Plot all curves
    for col in range(0,y_vals.shape[1]):
        # Plot SD area
        if y_SD is not None and y_SD.shape[1]>= col:
            ax.fill_between(x_vals, y_vals[:,col]-y_SD[:,col], y_vals[:,col]+y_SD[:,col], interpolate=True, zorder = col+z_offset, color = colors[col], alpha = sd_alpha)
        # Plot the mean line
        ax.plot(x_vals, y_vals[:,col], color=colors[col],zorder=col+1, label=labels[col],**kwargs)

    if flip_x != flip_y:
        ax.legend(loc='lower left')
    else:
        ax.legend(loc='lower right')
    
    _set_label_if_not_set(ax,x_lab, True)
    _set_label_if_not_set(ax,y_lab, False)
    _set_title(ax,title)
    
    if tight_layout:
        error_fig.tight_layout()
    
    return error_fig