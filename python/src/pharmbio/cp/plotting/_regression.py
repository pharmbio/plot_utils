"""CP Regression plots
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_consistent_length

from pharmbio.cp.utils import to_numpy1D, to_numpy2D, validate_sign

# The following import sets seaborn etc if available 
from ._utils import get_fig_and_axis, _set_chart_size, _plot_vline,_set_label_if_not_set, _set_title
from ._common import add_calib_curve


def plot_pred_widths(sign_vals,
                     pred_widths,
                     pred_widths_sd = None,
                     color = 'blue',
                     sd_alpha = 0.3,
                     flip_x = False,
                     ax = None,
                     figsize = (10,8),
                     chart_padding = 0.025,
                     title = None,
                     y_label = 'Median Prediction interval width',
                     tight_layout = True,
                     **kwargs):
    """**Regression** - Plot prediction widths at different significance levels
    
    Parameters
    ----------
    sign_vals : array like
        List of significance levels for each of the `pred_widths`
    
    pred_widths : array like
        List or 1D array of prediction widths, typically generated from `metrics.pred_width`
    
    pred_widths_sd : array like, optional
        List or 1D array of standard deviations for the prediction widths, to depict the width+/-std area
    
    color : str or matplotlib recognized color-input
        Color of the plotted curve
    
    flip_x : bool, default False
        If the x-axes should display significance level (`False`) or confidence (`True`)

    ax : matplotlib Axes
        Axes to plot in
    
    figsize : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given
    
    chart_padding : float, (float,float) or None
        padding added to the chart-area outside of the min and max values found in data. If two values the first value will be used as x-padding and second y-padding. E.g. 0.025 means 2.5% on both sides. None means no padding at all
    
    title : str, optional
        Optional title that will be printed in 'x-large' font size (default None)
    
    y_label : str or None
        Label for the y-axis, default is 'Median Prediction interval width'

    tight_layout : bool, optional
        Set `tight_layout` on the matplotlib Figure object
    
    **kwargs : dict, optional
        Keyword arguments, passed to matplotlib

    Returns
    -------
    Figure
        matplotlib.figure.Figure object

    See Also
    --------
    metrics.pred_width
        Calculates median or mean prediction interval widths
    """
    
    check_consistent_length(sign_vals, pred_widths)
    if pred_widths_sd is not None:
        check_consistent_length(sign_vals, pred_widths_sd)

    validate_sign(sign_vals)
    fig, ax = get_fig_and_axis(ax, figsize)

    if flip_x:
        xs = 1 - sign_vals if isinstance(sign_vals,np.ndarray) else 1 - np.array(sign_vals)
        x_label = 'Confidence' 
    else:
        xs = sign_vals
        x_label = 'Significance'

    # Set chart range
    _set_chart_size(ax,
        xs,
        pred_widths,
        chart_padding)

    if pred_widths_sd is not None:
        ax.fill_between(xs, pred_widths-pred_widths_sd, pred_widths+pred_widths_sd, interpolate=True, alpha = sd_alpha)
    ax.plot(xs, pred_widths, color=color,label=y_label, **kwargs)

    # Print some labels and title if appropriate
    _set_label_if_not_set(ax,x_label,True)
    _set_label_if_not_set(ax,y_label,False)
    _set_title(ax,title)

    if tight_layout:
        fig.tight_layout()
    
    return fig



def plot_pred_intervals(y_true,
    predictions,
    ax = None,
    figsize = (10,8),
    chart_padding = 0.025,

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
    
    title = None,
    x_label = 'Predicted examples',
    y_label = None,
    
    x_start_index = 0,
    tight_layout = True,
    **kwargs):
    """**Regression** - Plot predictions and their confidence intervals

    Sorts the predictions after the size of the `y_true` values and plots both the true labels and the prediction/confidence intervals (CI) for each prediction. Erronious and correctly predicted examples can be discerned by using different colors and markers for the CI and/or the true value-points.

    Parameters
    ----------
    y_true : 1D numpy array, list or pandas Series
        True labels
    
    predictions : 2D ndarray
        2D array of shape (n_samples, 2) where the second dimension should have min interval limit at index 0 and max interval limit at index 1
    
    ax : matplotlib Axes, optional
        An existing matplotlib Axes to plot in (default None)

    figsize : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    chart_padding : float, (float,float) or None, default 0.025
        padding added to the chart-area outside of the min and max values found in data. If two values the first value will be used as x-padding and second y-padding. E.g. 0.025 means 2.5% on both sides
    
    correct_color,incorrect_color : str of matplotlib recognized color-input
        Color of the points for the true values
    
    correct_marker, incorrect_marker : str of matplotlib recognized marker 
        The shape of the true values
    
    correct_alpha, incorrect_alpha : float, default 0.75
        The alpha (transparency) of the true values
    
    correct_ci, incorrect_ci : str of matplotlib recognized color-input
        Color of the confidence/prediction intervals
    
    correct_ci_alpha, incorrect_ci_alpha : float
        The alpha (transparency) of the confidence/prediction intervals
    
    correct_label,incorrect_label : str
        The label, if any, that should be added to the correct/incorrect true examples, which will end up in the plot if adding a legend in the figure
    
    line_cap : {None, 1, 2 or str}, default 1
        The end of the confidence/prediction intervals, 1:'_', 2:6 / 7 out of the accepted markers list: https://matplotlib.org/stable/api/markers_api.html
    
    title : str, default None
        A title to add to the figure (default None)

    x_label, y_label : str, optional
        label for the x/y-axis, default is None for the y-axis and 'Predicted examples' on the x-axis
    
    x_start_ind : int, default 0
        The starting index on the x-axis

    Returns
    -------
    Figure
        matplotlib.figure.Figure object

    See Also
    --------
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
        [x_start_index,len(y_true)+x_start_index],
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
    ax.scatter(xs[corr_ind],
        ys[corr_ind], 
        label = correct_label,
        marker = correct_marker,
        alpha = correct_alpha,
        color = correct_color,
        zorder = 2.5)
    # incorrects
    ax.scatter(xs[incorr_ind],
        ys[incorr_ind],
        label = incorrect_label,
        marker = incorrect_marker,
        alpha = incorrect_alpha,
        color = incorrect_color,
        zorder = 2.5)

    # Print some labels and title if appropriate
    _set_label_if_not_set(ax, y_label, x_axis=False)
    _set_label_if_not_set(ax, x_label, x_axis=True)
    _set_title(ax,title)

    if tight_layout:
        fig.tight_layout()
    
    return fig
