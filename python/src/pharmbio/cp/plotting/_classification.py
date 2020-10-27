import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import math
import numpy as np
from sklearn.utils import check_consistent_length
import pandas as pd

# Package stuff
from ..utils import *

from ..metrics import frac_error, frac_single_label_preds
from ..metrics import frac_multi_label_preds


__using_seaborn = False
# Try to import sns as they create somewhat nicer plots
try:
    import seaborn as sns
    sns.set()
    logging.debug('Using Seaborn plotting defaults')
    __using_seaborn = True
except ImportError as e:
    logging.debug('Seaborn not available - using Matplotlib defaults')
    pass 

# Set some defaults that will be used amongst the plotting functions
__default_color_map = list(mpl.rcParams['axes.prop_cycle'].by_key()['color'])
__default_single_label_color = __default_color_map.pop(2) # green
__default_multi_label_color = __default_color_map.pop(2) # red
__default_empty_prediction_color = "gainsboro"

__default_incorr_single_label_color = __default_color_map.pop(2) 
__default_incorr_multi_label_color = __default_color_map.pop(2)

####################################
### INTERNAL UTILS FUNCTIONS
####################################

def _get_fig_and_axis(ax, fig_size = (10,8)):
    '''Internal function for instantiating a Figure / axes object
    
    Returns
    -------
    fig : Figure
    
    ax : matplotlib axes
    '''
    
    if ax is None:
        # No current axes, create a new Figure
        if isinstance(fig_size, (int, float)):
            fig = plt.figure(figsize = (fig_size, fig_size))
        elif isinstance(fig_size, tuple):
            fig = plt.figure(figsize = fig_size)
        else:
            raise TypeError('parameter fig_size must either be float or (float, float), was: ' +
                str(type(fig_size)))
        # Add an axes spanning the entire Figure
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    
    return fig, ax

def _cm_as_list(cm):
    if cm is None:
        return __default_color_map
    elif isinstance(cm, mpl.colors.ListedColormap):
        return list(cm.colors)
    elif isinstance(cm, list):
        return cm
    else:
        return [cm]

def _get_default_labels(labels, unique_labels):
    sorted_labels = sorted(unique_labels)
    if labels is not None:
        if not isinstance(labels, (np.ndarray, list)):
            raise TypeError('parameter labels must be either a list or 1D numpy array')
        if len(labels) < sorted_labels[-1]:
            raise TypeError('parameter labels and number of classes does not match')
        return np.array(labels).astype(str)
    else:
        # No labels, use the unique_labels found
        labels = []
        for lab in range(0, unique_labels[-1]+1):
            labels.append('Label ' + str(lab))
        return np.array(labels)


####################################
### CLASSIFICATION
####################################

def plot_pvalues(y_true,
                    p_values,
                    cols = [0,1],
                    labels = None,
                    ax = None,
                    fig_size = (10,8),
                    cm = None,
                    markers = None,
                    sizes = None,
                    title = None,
                    order = "freq",
                    x_label = 'p-value {class}',
                    y_label = 'p-value {class}',
                    add_legend = True,
                    tight_layout = True,
                    fontargs = None,
                    **kwargs):
    """Plot p-values agains each other

    Plot p-values against each other, switch the axes by setting the `cols` parameter
    and handle multi-class predictions by deciding which p-values should be plotted.
    
    Parameters
    ----------
    y_true : 1D numpy array, list or pandas Series
        True labels

    p_values : 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..

    cols : list of int
        Colums in the `p_values` matrix to plot

    labels : list of str, optional
        Textual labels for the classes, will affect the x- and y-labels and the legend

    ax : matplotlib Axes, optional
        An existing matplotlib Axes to plot in (default None)

    fig_size : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    cm : color, list of colors or ListedColorMap, optional
        Colors to plot each class with, index 0 for class 0, etc.

    markers : str or list of str, optional
        Markers to use, if a single one is given, all points/classes will get that marker,
        if a list is given index 0 will be used for class 0, etc. If the list is of the same length
        as the number of predictions each example will get their own marker

    sizes : float or list of float, optional
        Size(s) to use for all predictions or for predictions for each class

    title : str, optional
        A title to add to the figure (default None)

    order : {'freq', 'class', 'label', None}
        Order in which the points are plotted, options:
        'freq' : plot each half of the plot independently, choosing the order by
            the frequency of each class - so the smallest class is plotted last.
            This will make it less likely that outliers are hidden by points plotted later
        'class' / 'label' / None : Plot based on order of classes, i.e. plot class 0, 1, 2,..

    x_label : str, optional
        label for the x-axis, default is 'p-value {class x}' where x is based on the `cols` parameter
        If None is given, no y-label is added to the figure

    y_label : str, optional
        label for the y-axis, default is 'p-value {class y}' where y is based on the `cols` parameter.
        If None is given, no y-label is added to the figure

    add_legend : bool, optional
        If a legend should be added to the figure (Default True)

    fontargs : dict, optional
        Font arguments passed to matplotlib

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
    matplotlib.colors.ListedColormap
    """
    # Verify and convert to correct format
    y_true = to_numpy1D_int(y_true, 'y_true')
    unique_labels = np.sort(np.unique(y_true).astype(int))
    p_values = to_numpy2D(p_values, 'p_values')
    check_consistent_length(y_true,p_values)

    n_class = p_values.shape[1]

    # Verify cols argument
    if cols is None or not isinstance(cols, list):
        raise ValueError('parameter cols must be a list of integer')
    if len(cols) != 2:
        raise ValueError('parameter cols must only have 2 values (can only plot 2D)')
    for col in cols:
        if not isinstance(col, int) or col < 0 or col > n_class:
            raise ValueError('parameter col must be a list of int, all in the range [0,'+
                str(n_class) + ']')
    
    fig, ax = _get_fig_and_axis(ax, fig_size)
    fontargs = fontargs if fontargs is not None else {}

    # Validate the order-argument
    if order is not None and not isinstance(order, str):
        raise TypeError('parameter order must be None or str type, was ' + str(type(order)))

    # Set the color-map (list)
    colors = _cm_as_list(cm)

    # Verify the labels
    labels = _get_default_labels(labels, unique_labels)
    
    # Set the markers to a list
    if markers is None:
        # default
        plt_markers = [mpl.rcParams["scatter.marker"]]
    elif isinstance(markers, list):
        # Unique markers for all
        plt_markers = markers
    else:
        # not a list - same marker for all
        plt_markers = [markers]
    
    # Set the size of the markers
    if sizes is None:
        plt_sizes = [None] # Default
    elif isinstance(sizes, list):
        plt_sizes = sizes
    else:
        plt_sizes = [sizes]
    
    # Do the plotting
    if order is None or (order.lower() == 'class' or order.lower() == 'label') :
        # Use the order of the labels simply 
        for lab in unique_labels:
            label_filter = np.array(y_true == lab)
            x = p_values[label_filter, cols[0]]
            y = p_values[label_filter, cols[1]]
            ax.scatter(x, y, 
                s = plt_sizes[lab % len(plt_sizes)], 
                color = colors[lab % len(colors)], 
                marker = plt_markers[lab % len(plt_markers)], 
                label = labels[lab],
                **kwargs)
        # The order is now based on the unique_labels list-so adding a legend is straight forward
        if add_legend:
            ax.legend(loc='upper right',**fontargs)
    elif order.lower() == 'freq':
        # Try to use a smarter apporach
        num_ul = []
        num_lr = []
        num_per_label = []
        # Compute the order of classes to plot
        for lab in unique_labels:
            label_filter = np.array(y_true == lab)
            x = p_values[label_filter, cols[0]]
            y = p_values[label_filter, cols[1]]
            num_per_label.append(len(x))
            # the number in lower-right part and upper-left
            num_lr.append((x>y).sum())
            num_ul.append(x.shape[0] - num_lr[-1])
        
        # Normalize for size
        num_ul = np.array(num_ul) / np.array(num_per_label)
        num_lr = np.array(num_lr) / np.array(num_per_label)
        
        # Plot the upper-left section
        lab_order, handles = [], []
        ul_sorting = np.argsort(num_ul)[::-1]
        for lab in unique_labels[ul_sorting]:
            label_filter = np.array(y_true == lab)
            x = p_values[label_filter, cols[0]]
            y = p_values[label_filter, cols[1]]
            upper_left_filter = y >= x
            x = x[upper_left_filter]
            y = y[upper_left_filter]
            handle = ax.scatter(x, y, 
                s = plt_sizes[lab % len(plt_sizes)], 
                color = colors[lab % len(colors)], 
                marker = plt_markers[lab % len(plt_markers)], 
                **kwargs)
            lab_order.append(lab)
            handles.append(handle)

        # Plot the lower-right section
        lr_sorting = np.argsort(num_lr)[::-1]
        for lab in unique_labels[lr_sorting]:
            label_filter = np.array(y_true == lab)
            x = p_values[label_filter, cols[0]]
            y = p_values[label_filter, cols[1]]
            lower_right_filter = y < x
            x = x[lower_right_filter]
            y = y[lower_right_filter]
            # Skip the labels in second one!
            ax.scatter(x, y, 
                s = plt_sizes[lab % len(plt_sizes)], 
                color = colors[lab % len(colors)], 
                marker = plt_markers[lab % len(plt_markers)], 
                **kwargs)
        # Add legend - in correct order
        if add_legend:
            ls, hs = zip(*sorted(zip(lab_order, handles), key=lambda t: t[0]))
            ax.legend(hs, [labels[x] for x in ls], loc='upper right', **fontargs)
    else:
        raise ValueError('parameter order not any of the allowed values: ' + str(order))

    ax.set_ylim([-.025, 1.025])
    ax.set_xlim([-.025, 1.025])
    
    # Set some labels and title

    if y_label is not None:
        y_label = y_label.replace('{class}', str(labels[cols[1]]))
        ax.set_ylabel(y_label,**fontargs)

    if x_label is not None:
        x_label = x_label.replace('{class}', str(labels[cols[0]]))
        ax.set_xlabel(x_label,**fontargs)

    if title is not None:
        if not bool(fontargs):
            # No font-args given, use larger font for the title
            ax.set_title(title, {'fontsize': 'x-large'})
        else:
            ax.set_title(title, **fontargs)
    
    if tight_layout:
        fig.tight_layout()

    return fig


def plot_calibration_curve(y_true,
                            p_values,
                            labels = None,
                            ax = None,
                            fig_size = (10,8),
                            sign_min=0,
                            sign_max=1,
                            sign_step=0.01,
                            sign_vals=None,
                            cm = None,
                            overall_color = 'black',
                            chart_padding=None,
                            plot_all_labels=True,
                            title=None,
                            tight_layout=True,
                            **kwargs):
    
    """**Classification** - Create a calibration curve
    
    Parameters
    ----------
    y_true : list or 1D numpy array
        The true labels (with values 0, 1, etc for each class)

    p_values : A 2D numpy array
        P-values, first column with p-value for class 0, second for class 1, ..

    labels : list of str, optional
        Descriptive labels for the classes

    ax : matplotlib Axes, optional
        An existing matplotlib Axes to plot in (default None)

    fig_size : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    title : str, optional
        A title to add to the figure (default None)

    sign_min : float range [0,1], optional
        The smallest significance level to include (default 0)

    sign_max : float range [0,1], optional
        The largest significance level to include (default 1)

    sign_step : float in range [1e-4, 0.5], optional
        Spacing between evaulated significance levels (default 0.01)

    sign_vals : list of float, optional
        A list of significance values to use, the `significance_` values will be ignored if this parameter was passed (default None)

    cm : color, list of colors or ListedColorMap, optional
        The colors to use. First color will be for class 0, second for class 1, ..

    overall_color : color, optional
        The color to use for the overall error rate (Default 'black')

    chart_padding : float, optional
        Padding added to the drawing area (default None will add 2.5% of padding)

    plot_all_labels : boolean, optional
        Plot the error rates for each class (default True). If False, only the 'overall' error rate is plotted

    title : str, optional
        Optional title that will be printed in 'x-large' font size (default None)

    tight_layout : bool, optional
        Set `tight_layout` on the matplotlib Figure object

    **kwargs : dict, optional
        Keyword arguments, passed to matplotlib
    
    Returns
    -------
    fig : Figure
        matplotlib.figure.Figure object
    
    See Also
    --------
    matplotlib.colors.ListedColormap
    """
    
    # Create a list with all significances 
    sign_vals = get_sign_vals(sign_vals, sign_min,sign_max,sign_step)
    
    # Verify and convert to correct format
    y_true = to_numpy1D_int(y_true, 'y_true')
    unique_labels = np.sort(np.unique(y_true).astype(int))
    p_values = to_numpy2D(p_values, 'p_values')
    labels = _get_default_labels(labels, unique_labels)
    check_consistent_length(y_true,p_values)

    if chart_padding is None:
        chart_padding = (sign_vals[-1] - sign_vals[0])*0.025
    
    overall_error_rates = []
    if plot_all_labels:
        label_based_rates = np.zeros((len(sign_vals), p_values.shape[1]))
    
    for ind, s in enumerate(sign_vals):
        overall, label_based = frac_error(y_true,p_values,s)
        overall_error_rates.append(overall)
        if plot_all_labels:
            # sets all values in a row
            label_based_rates[ind] = label_based
    
    error_fig, ax = _get_fig_and_axis(ax, fig_size)
    # Set chart range and add dashed diagonal
    ax.axis([sign_vals[0]-chart_padding, sign_vals[-1]+chart_padding, sign_vals[0]-chart_padding, sign_vals[-1]+chart_padding]) 
    ax.plot(sign_vals, sign_vals, '--k', alpha=0.25, linewidth=1)
    
    # Plot overall (high zorder to print it on top)
    ax.plot(sign_vals, overall_error_rates, c=overall_color, label='Overall', zorder=10, **kwargs)

    if plot_all_labels:
        colors = _cm_as_list(cm)
        for i in range(label_based_rates.shape[1]):
            ax.plot(sign_vals, label_based_rates[:,i], color=colors[i], label=labels[i], **kwargs)
    
    ax.legend(loc='lower right')
    
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Significance")
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})
    
    if tight_layout:
        error_fig.tight_layout()
    
    return error_fig

def plot_label_distribution(y_true,
                            p_values,
                            ax=None,
                            fig_size=(10,8),
                            title=None,
                            sign_min=0,
                            sign_max=1,
                            sign_step=0.01,
                            sign_vals=None,
                            cm=None,
                            display_incorrects=False,
                            mark_best=True,
                            tight_layout=True,
                            **kwargs):
    """**Classification** - Create a stacked plot with label ratios
    
    Parameters
    ----------
    y_true : A list or 1D numpy array
        The true classes/labels using values 0, 1, etc for each class

    p_values : 2D numpy array
        The predicted p-values, first column for the class 0, second for class 1, ..

    ax : matplotlib Axes, optional
        An existing matplotlib Axes to plot in (default None)

    fig_size : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    title : str, optional
        A title to add to the figure (default None)

    sign_min : float range [0,1], optional
        The smallest significance level to include (default 0)

    sign_max : float range [0,1], optional
        The largest significance level to include (default 1)

    sign_step : float in range [1e-4, 0.5], optional
        Spacing between evaulated significance levels (default 0.01)

    sign_vals : list of float, optional
        Significance values to use, the `significance_` parameters will be ignored if this 
        parameter was passed (default None)

    cm : list of colors or ListedColorMap, optional
        Colors to use, given in the order: [single, multi, empty, (incorrect-single), (incorrect-multi)]  
        (if `display_incorrects`=True a list of at least 5 is required, otherwise 3 is sufficient)

    display_incorrects : boolean, optional
        Plot the incorrect predictions intead of only empty/single/multi-label predictions (default True)

    mark_best : boolean
        Mark the best significance value with a line and textbox (default True)
    
    tight_layout : bool, optional
        Set `tight_layout` on the matplotlib Figure object

    **kwargs : dict, optional
        Keyword arguments, passed to matplotlib
    
    Returns
    -------
    fig : Figure
        matplotlib.figure.Figure object
    """
    
    # Set color-mapping
    if cm is not None:
        pal = list(cm)
    else:
        pal = [__default_single_label_color, __default_multi_label_color, __default_empty_prediction_color, __default_incorr_single_label_color,__default_incorr_multi_label_color]

    # Create a list with all significances
    sign_vals = get_sign_vals(sign_vals, sign_min,sign_max,sign_step)

    # Validate format
    p_values = to_numpy2D(p_values,'p_values')
    y_true = to_numpy1D_int(y_true, 'y_true')
    check_consistent_length(y_true,p_values)
    
    # Calculate the values
    s_label = []
    si_label = [] # Incorrects
    m_label = []
    mi_label = [] # Incorrects
    empty_label = []
    highest_single_ratio = -1
    best_sign = -1
    for s in sign_vals:
        _, s_corr, s_incorr = frac_single_label_preds(y_true, p_values,s)
        _, m_corr, m_incorr = frac_multi_label_preds(y_true, p_values, s)
        if display_incorrects:
            s_l = s_corr
            m_l = m_corr
        else:
            # Here these should be summed to gain the totals
            s_l = s_corr + s_incorr
            m_l = m_corr + m_incorr
        # Update lists
        s_label.append(s_l)
        si_label.append(s_incorr)
        m_label.append(m_l)
        mi_label.append(m_incorr)
        sum_labels = s_corr+s_incorr+m_corr+m_incorr
        
        # Update the best significance value
        if s_l > highest_single_ratio:
            highest_single_ratio = s_l
            best_sign = s
        # The empty labels are the remaing predictions
        empty_label.append(1 - sum_labels)
    
    # Convert all to numpy arrays
    s_label = np.array(s_label)
    si_label = np.array(si_label)
    m_label = np.array(m_label)
    mi_label = np.array(mi_label)
    empty_label = np.array(empty_label)
    
    ys = []
    labels = []
    colors = []

    if not np.all(s_label == 0):
        ys.append(s_label)
        if display_incorrects:
            labels.append('Correct single-label')
        else:
            labels.append('Single-label')
        colors.append(pal[0 % len(pal)])
    if display_incorrects and not np.all(si_label == 0):
        ys.append(si_label)
        labels.append('Incorrect single-label')
        colors.append(pal[3 % len(pal)])
    if not np.all(m_label == 0):
        ys.append(m_label)
        if (p_values.shape[1] == 2):
            labels.append('Both')
        else:
            labels.append('Multi-label')
        colors.append(pal[1 % len(pal)])
    if p_values.shape[1] > 2 and display_incorrects and not np.all(mi_label == 0):
        ys.append(mi_label)
        labels.append('Incorrect multi-label')
        colors.append(pal[4 % len(pal)])
    if not np.all(empty_label == 0):
        ys.append(empty_label)
        labels.append('Empty')
        colors.append(pal[2 % len(pal)])
    
    fig, ax = _get_fig_and_axis(ax, fig_size)

    ax.axis([sign_vals[0],sign_vals[-1],0,1])
    ax.stackplot(sign_vals,ys,
                  labels=labels, 
                  colors=colors,**kwargs) #
    
    if mark_best:
        ax.plot([best_sign,best_sign],[0,1],'--k', alpha=0.75)
        props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        ax.text(best_sign+0.02, 0.1, str(best_sign), bbox=props)
    
    ax.legend()
    
    ax.set_ylabel("Label distribution")
    ax.set_xlabel("Significance")
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})
    
    if tight_layout:
        fig.tight_layout()
    
    return fig


def plot_confusion_matrix_bubbles(confusion_matrix,
                                  ax=None,
                                  fig_size=(10,8),
                                  title=None,
                                  bubble_size_scale_factor = 1,
                                  color_scheme = 'prediction_size',
                                  tight_layout = True,
                                  **kwargs):
    """**Classification** - Create a Confusion matrix bubble plot 

    Render a confusion matrix with bubbles, the size of the bubbles are related to the frequency
    of that prediction type. The confusion matrix is made at a specific significance level. 
    
    Parameters
    ----------
    confusion_matrix : DataFrame
        Confusion matrix in pandas DataFrame, from `metrics.confusion_matrix`

    ax : matplotlib Axes, optional
        An existing matplotlib Axes to plot in (default None)

    fig_size : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    bubble_size_scale_factor : number, optional
        Scaling to be applied on the size of the bubbles, default scale factor works OK for the default figure size

    color_scheme : { None, 'None', 'prediction_size', 'label', 'class', 'full' }
        None/'None':=All in the same color 
        'prediction_size':=Color single/multi/empty in different colors
        'label'/'class':=Each class colored differently
        'full':=Correct single, correct multi, incorrect single, incorrect multi and empty colored differently
    
    tight_layout : bool, optional
        Set `tight_layout` on the matplotlib Figure object

    **kwargs : dict, optional
        Keyword arguments, passed to matplotlib
    
    Returns
    -------
    fig : Figure
        matplotlib.figure.Figure object
    
    See Also
    --------
    metrics.calc_confusion_matrix : Calculating confusion matrix
    """
    
    # Make sure CM exists and correct type
    if confusion_matrix is None:
        raise ValueError('parameter confusion_matrix is required')
    if not isinstance(confusion_matrix, pd.DataFrame):
        raise TypeError('argument confusion_matrix must be a pandas DataFrame')

    # Create Figure and Axes if not given
    fig, ax = _get_fig_and_axis(ax, fig_size)
    
    x_coords = []
    y_coords = []
    sizes = confusion_matrix.to_numpy().ravel(order='F')
    n_rows = confusion_matrix.shape[0]
    for x in confusion_matrix.columns:
        x_coords.extend([x]*n_rows)
        y_coords.extend(confusion_matrix.index)
    
    # Set the colors
    colors = None
    if color_scheme is None:
        pass
    elif color_scheme.lower() == "none":
        pass
    elif color_scheme.lower() == 'prediction_size':
        n_class = confusion_matrix.shape[1]
        colors = [__default_single_label_color]*n_class
        colors.append(__default_empty_prediction_color)
        colors.append(__default_multi_label_color)
        if n_class > 2:
            colors.append(__default_multi_label_color)
        colors = colors*n_class
    elif color_scheme.lower() == 'class' or color_scheme.lower() == 'label':
        n_class = confusion_matrix.shape[1]
        n_rows = confusion_matrix.shape[0]
        colors = []
        for c in range(n_class):
            c_color = [__default_color_map[c]]*n_rows
            colors.extend(c_color)
    elif color_scheme.lower() == 'full':
        colors = []
        n_class = confusion_matrix.shape[1]

        for c in range(n_class):
            c_cols = [__default_incorr_single_label_color]*n_class
            c_cols[c] = __default_single_label_color # Only index c should be 'correct' color
            c_cols.append(__default_empty_prediction_color)
            c_cols.append(__default_multi_label_color)
            if n_class > 2:
                c_cols.append(__default_multi_label_color)
            colors.extend(c_cols)
    else:
        print('color_scheme=' +str(color_scheme) + " not recognized, falling back to None")

    # Convert the x and y coordinates to strings
    x_coords = np.array(x_coords, dtype=object).astype(str)
    y_coords = np.array(y_coords, dtype=object).astype(str)
    sizes_scaled = bubble_size_scale_factor * 2500 * sizes / sizes.max()
    
    ax.scatter(x_coords, y_coords, s=sizes_scaled,c=colors,edgecolors='black',**kwargs)

    ax.margins(.3)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.invert_yaxis()
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})

    for xi, yi, zi, z_si in zip(x_coords, y_coords, sizes, sizes_scaled):
        if isinstance(zi, float):
            zi = round(zi,2)
        ax.annotate(zi, xy=(xi, yi), xytext=(np.sqrt(z_si)/2.+5, 0),
                 textcoords="offset points", ha="left", va="center")
    if tight_layout:
        fig.tight_layout()
    
    return fig


def plot_confusion_matrix_heatmap(confusion_matrix, 
                                    ax=None, 
                                    fig_size=(10,8), 
                                    title=None,
                                    cmap=None,
                                    cbar_kws=None,
                                    tight_layout=True,
                                    **kwargs):
    """**Classification** - Plots the Conformal Confusion Matrix in a Heatmap
    
    Note that this method requires the Seaborn to be available and will fail otherwise
    
    Parameters
    ----------
    confusion_matrix : DataFrame
        Confusion matrix in pandas DataFrame, from metrics.calc_confusion_matrix

    ax : matplotlib Axes, optional
        An existing matplotlib Axes to plot in (default None)

    fig_size : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    title : str, optional
        Optional title that will be printed in 'x-large' font size (default None)

    cmap : matplotlib colormap name or object, or list of colors, optional
        Colormap to use for the heatmap, argument passed to Seaborn heatmap

    cbar_kws : dict, optional
        Arguments passed to the color-bar element
    
    tight_layout : bool, optional
        Set `tight_layout` on the matplotlib Figure object

    **kwargs : dict, optional
        Keyword arguments, passed to matplotlib
    
    Returns
    -------
    fig : Figure
        matplotlib.figure.Figure object
    
    See Also
    --------
    metrics.calc_confusion_matrix : Calculating confusion matrix
    """
    
    if not __using_seaborn:
        raise RuntimeError('Seaborn is required when using this function')
    
    fig, ax = _get_fig_and_axis(ax, fig_size)
    
    if title is not None:
        ax.set_title(title, fontdict={'fontsize':'x-large'})
    ax = sns.heatmap(confusion_matrix, ax=ax, annot=True, cmap=cmap, cbar_kws=cbar_kws, **kwargs)
    ax.set(xlabel='Predicted', ylabel='Observed')

    if tight_layout:
        fig.tight_layout()
    
    return fig


