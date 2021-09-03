"""CP Classification plots
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_consistent_length
import pandas as pd

from ..utils import get_n_classes,get_str_labels,to_numpy2D,to_numpy1D_int, validate_sign

# The following import sets seaborn etc if available 
from ._utils import get_fig_and_axis, cm_as_list, _using_seaborn, _set_title, _set_label_if_not_set,_set_chart_size
from ._common import add_calib_curve

from ..metrics import frac_errors, frac_single_label_preds
from ..metrics import frac_multi_label_preds


# Set some defaults that will be used amongst the plotting functions
__default_color_map = list(mpl.rcParams['axes.prop_cycle'].by_key()['color'])
__default_single_label_color = __default_color_map.pop(2) # green
__default_multi_label_color = __default_color_map.pop(2) # red
__default_empty_prediction_color = "gainsboro"

__default_incorr_single_label_color = __default_color_map.pop(2) 
__default_incorr_multi_label_color = __default_color_map.pop(2)


####################################
### CLASSIFICATION
####################################

def plot_pvalues(y_true,
    p_values,
    cols = [0,1],
    labels = None,
    ax = None,
    figsize = (10,8),
    chart_padding = 0.025,
    cm = None,
    markers = None,
    sizes = None,
    order = "freq",
    title = None,
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

    figsize : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    cm : color, list of colors or ListedColorMap, optional
        Colors to plot each class with, index 0 for class 0, etc.

    markers : str or list of str, optional
        Markers to use, if a single one is given, all points/classes will get that marker,
        if a list is given index 0 will be used for class 0, etc. If the list is of the same length
        as the number of predictions each example will get their own marker

    sizes : float or list of float, optional
        Size(s) to use for all predictions or for predictions for each class

    order : {'freq', 'class', 'label', None}
        Order in which the points are plotted, options:
        'freq' : plot each half of the plot independently, choosing the order by
            the frequency of each class - so the smallest class is plotted last.
            This will make it less likely that outliers are hidden by points plotted later
        'class' / 'label' / None : Plot based on order of classes, i.e. plot class 0, 1, 2,..

    title : str, optional
        A title to add to the figure (default None)

    x_label, y_label : str, optional
        label for the x/y-axis, default is 'p-value {class x/y}' where x/y is based on the `cols` parameter
        If None is given, no x/y-label is added to the figure

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

    n_pvals = p_values.shape[1]

    # Verify cols argument
    if cols is None or not isinstance(cols, list):
        raise ValueError('parameter cols must be a list of integer')
    if len(cols) != 2:
        raise ValueError('parameter cols must only have 2 values (can only plot 2D)')
    for col in cols:
        if not isinstance(col, int) or col < 0 or col >= n_pvals:
            raise ValueError('parameter col must be a list of int, all in the range [0,{}]'.format(n_pvals-1))
    
    fig, ax = get_fig_and_axis(ax, figsize)
    _set_chart_size(ax,[0,1], [0,1], chart_padding)
    fontargs = fontargs if fontargs is not None else {}

    # Validate the order-argument
    if order is not None and not isinstance(order, str):
        raise TypeError('parameter order must be None or str, was {}'.format(type(order)))

    # Set the color-map (list)
    colors = cm_as_list(cm, __default_color_map)

    # Verify the labels
    n_class = get_n_classes(y_true, p_values)
    labels = get_str_labels(labels, n_class)
    
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
    
    # Set some labels and title
    if y_label is not None:
        y_label = y_label.replace('{class}', str(labels[cols[1]]))
        ax.set_ylabel(y_label,**fontargs)

    if x_label is not None:
        x_label = x_label.replace('{class}', str(labels[cols[0]]))
        ax.set_xlabel(x_label,**fontargs)

    _set_title(ax,title)
    
    if tight_layout:
        fig.tight_layout()

    return fig


def plot_calibration_curve(y_true,
    p_values,
    labels = None,
    ax = None,
    figsize = (10,8),
    chart_padding=0.025,
    sign_vals=np.arange(0,1,0.01),
    cm = None,
    overall_color = 'black',
    std_orientation=True,
    plot_all_labels=True,
    title=None,
    tight_layout=True,
    **kwargs):
    
    """**Classification** - Create a calibration plot

    By default, all class-wise calibration curves will be printed as well as the 'overall' error rate/accuracy for all examples
    
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

    figsize : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    chart_padding : float, (float,float) or None, default 0.025
        padding added to the chart-area outside of the min and max values found in data. If two values the first value will be used as x-padding and second y-padding. E.g. 0.025 means 2.5% on both sides

    sign_vals : list of float, default np.arange(0,1,0.01)
        A list of significance values to use, 

    cm : color, list of colors or ListedColorMap, optional
        The colors to use. First color will be for class 0, second for class 1, ..

    overall_color : color, optional
        The color to use for the overall error rate (Default 'black')
    
    std_orientation : bool, optional
        If the axes should have the standard 'error rate vs significance' (`True`) or
        alternative 'Accuracy vs Confidence' (`False`) orientation

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
    validate_sign(sign_vals)
    if len(sign_vals) < 2:
        raise ValueError('Must have at least 2 significance values to plot a calibration curve')
    
    # Verify and convert to correct format
    y_true = to_numpy1D_int(y_true, 'y_true')
    p_values = to_numpy2D(p_values, 'p_values')
    check_consistent_length(y_true,p_values)

    n_class = get_n_classes(y_true,p_values)
    labels = get_str_labels(labels, n_class)

    # Calculate error rates
    overall_frac, cls_frac = frac_errors(y_true,p_values,sign_vals)
    
    # Create the figure and axis to plot in
    error_fig, ax = get_fig_and_axis(ax, figsize)
    
    (x_lab,y_lab) = add_calib_curve(ax,overall_frac,sign_vals,
        legend='Overall',zorder=100,
        color=overall_color,
        std_orientation=std_orientation,
        set_chart_size=True,
        chart_padding=chart_padding,
        plot_expected=True,
        **kwargs)

    if plot_all_labels:
        colors = cm_as_list(cm, __default_color_map)
        for i in range(cls_frac.shape[1]):
            add_calib_curve(ax,cls_frac[:,i],sign_vals,
                legend=labels[i],
                color=colors[i],
                set_chart_size=False,
                plot_expected=False,
                std_orientation=std_orientation,
                **kwargs)
    
    ax.legend(loc='lower right')
    
    _set_label_if_not_set(ax,x_lab, True)
    _set_label_if_not_set(ax,y_lab, False)
    _set_title(ax,title)
    
    if tight_layout:
        error_fig.tight_layout()
    
    return error_fig

def plot_label_distribution(y_true,
    p_values,
    ax=None,
    figsize=(10,8),
    title=None,
    x_label = 'Significance',
    y_label = 'Label distribution',
    sign_vals=np.arange(0,1,0.01),
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

    figsize : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    title : str, optional
        A title to add to the figure (default None)
    
    x_label,y_label : str, optional
        Labels for the x and y axes. Defaults are given

    sign_vals : list of float, default np.arange(0,1,0.01)
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

    # Validate the significance values
    validate_sign(sign_vals)

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
    
    fig, ax = get_fig_and_axis(ax, figsize)

    ax.axis([sign_vals[0],sign_vals[-1],0,1])
    ax.stackplot(sign_vals,ys,
                  labels=labels, 
                  colors=colors,**kwargs) #
    
    if mark_best:
        ax.plot([best_sign,best_sign],[0,1],'--k', alpha=0.75)
        props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        ax.text(best_sign+0.02, 0.1, str(best_sign), bbox=props)
    
    ax.legend()
    
    _set_title(ax,title)
    _set_label_if_not_set(ax,x_label,x_axis=True)
    _set_label_if_not_set(ax,y_label,x_axis=False)
    
    if tight_layout:
        fig.tight_layout()
    
    return fig


def plot_confusion_matrix_bubbles(confusion_matrix,
    ax=None,
    figsize=(10,8),
    title=None,
    scale_factor = 1,
    annotate = True,
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

    figsize : float or (float, float), optional
        Figure size to generate, ignored if `ax` is given

    scale_factor : number, optional
        Scaling to be applied on the size of the bubbles, default scale factor works OK for the default figure size
    
    annotate : boolean, optional
        If the actual numbers should be printed next to each bubble.

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
    fig, ax = get_fig_and_axis(ax, figsize)
    
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
    sizes_scaled = scale_factor * 2500 * sizes / sizes.max()
    
    ax.scatter(x_coords, y_coords, s=sizes_scaled,c=colors,edgecolors='black',**kwargs)

    ax.margins(.3)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.invert_yaxis()
    _set_title(ax,title)

    # Write the number for each bubble
    if annotate is not None and annotate: 
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
    figsize=(10,8), 
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

    figsize : float or (float, float), optional
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

    seaborn.heatmap : Seaborn is used for generating the heatmap. Here you can find available parameters
        for the `cbar_kws` and other possible customizations. 
    """
    
    if not _using_seaborn:
        raise RuntimeError('Seaborn is required when using this function')
    import seaborn as sns
    
    fig, ax = get_fig_and_axis(ax, figsize)
    
    _set_title(ax,title)
    ax = sns.heatmap(confusion_matrix, ax=ax, annot=True, cmap=cmap, cbar_kws=cbar_kws, **kwargs)
    ax.set(xlabel='Predicted', ylabel='Observed')

    if tight_layout:
        fig.tight_layout()
    
    return fig


