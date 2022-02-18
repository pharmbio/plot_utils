"""CP Classification plots
"""
import matplotlib as mpl
#import matplotlib.pyplot as plt
#from matplotlib.patches import Patch
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


def __plot_freq(ax,y,ps,s,c,m,unique_labels,cols,str_labels,add_legend=False,**kwargs):
    '''Helper function for `plot_pvalues`, used for plotting using order="freq" '''
    # Count them
    counts = np.zeros(len(unique_labels))
    for i,l in enumerate(unique_labels):
        counts[i] = (y==l).sum()
    l_order = np.argsort(- counts)

    for l in unique_labels[l_order]:

        if add_legend:
            kw = kwargs.copy()
            kw['label']=str_labels[l]
        else:
            kw = kwargs
        l_mask = y==l
        ax.scatter(x = ps[l_mask,cols[0]],
            y = ps[l_mask,cols[1]],
            s = s[l_mask], 
            c = c[l_mask], 
            marker = m[l],
            **kw)


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
    split_chart = True,
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

    markers : None, str or list of str, optional
        Markers to use, if a single one is given, all points/classes will get that marker,
        if a list is given index 0 will be used for class 0, etc. 

    sizes : float or list of float, optional
        Size(s) to use for all predictions or for predictions for each class

    order : {'freq', 'class', 'label', None}
        Order in which the points are plotted, options:
        'freq', None : Use the frequency of the classes in order to plot the most frequent first, then
            the second most frequent class etc. When `split_chart`=`True` each half of the plot
            is treated independently.
        'class' / 'label' : Plot based on order of classes, i.e. plot class 0, 1, 2,..
        'rev class' / 'rev label' : Plot based on the reverse order of classes, highest, next-highest,..,0
    
    split_chart : bool, optional
        Treat the upper-left and lower-right halves of the chart independently
        with respect to the `order` of the plotting of points. I.e. the frequency is 
        calculated for each half of the chart independently. Ignored if `order` != "freq"

    title : str, optional
        A title to add to the axes (default None)

    x_label, y_label : str, optional
        label for the x/y-axis, default is 'p-value {class x/y}' where x/y is based on the `cols` parameter
        If None is given, no x/y-label is added to the axes

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
    unique_labels = np.sort(np.unique(y_true).astype(np.int16))
    p_values = to_numpy2D(p_values, 'p_values')
    check_consistent_length(y_true,p_values)

    n_pvals = p_values.shape[1]

    # Verify cols argument
    if cols is None or not isinstance(cols, (list,tuple)):
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
    if order is None:
        pass
    elif isinstance(order, str):
        order = order.lower()
    else: 
        raise TypeError('parameter order must be None or str, was {}'.format(type(order)))

    # Set the color-map (list)
    colors = cm_as_list(cm, __default_color_map)

    # Verify the labels
    n_class = get_n_classes(y_true, p_values)
    labels = get_str_labels(labels, n_class)
    n_ex = len(y_true)
    
    # plt_markers as list of length n_class
    if markers is None:
        # default
        plt_markers = [mpl.rcParams["scatter.marker"]]*n_class
    elif isinstance(markers, (list,tuple)):
        if len(markers) >= n_class:
            plt_markers = markers
        else:
            raise TypeError('parameter markers must be None, str or list of same size as number of classes, was list of length {}'.format(len(markers)))
    else:
        # not a list - same marker for all
        plt_markers = [markers]*n_class
    
    # plt_sizes as ndarray of shape (n_ex,) 
    if sizes is None:
        plt_sizes = np.array([mpl.rcParams['lines.markersize'] ** 2]*n_ex) # Default
    elif isinstance(sizes, list) or isinstance(sizes,np.ndarray):
        if len(sizes) == n_ex:
            plt_sizes = np.array(sizes)
        elif len(sizes) == n_class:
            plt_sizes = np.empty(n_ex, dtype=np.float64)
            for l in unique_labels:
                plt_sizes[y_true == l] = sizes[l]
        elif len(sizes) == 1:
            plt_sizes = np.full(n_ex, fill_value=sizes)
        else:
            raise TypeError('parameter sizes must be None, float or list of same size as number of classes or number of plotted points, was list of length {}'.format(len(sizes)))
    else:
        plt_sizes = np.full(n_ex, fill_value=sizes)
    
    # Color as ndarray of shape (n_ex,)
    if len(colors) == n_ex:
        # One color per instance, have to convert to ndarray
        plt_c = np.array(n_ex)
    elif len(colors) == 1:
        # Same color for all instances
        if isinstance(colors[0],(tuple,list)):
            plt_c = np.full((n_ex,len(colors[0])),fill_value=colors[0])
        else:
            plt_c = np.full(n_ex,fill_value=colors[0])
    else:
        # One color per class
        if isinstance(colors[0],(str,np.str_)):
            plt_c = np.empty(n_ex, dtype=np.str_)
        elif isinstance(colors[0], (tuple,list)):
            # RGB or RGBA color tuples
            plt_c = np.empty((n_ex,len(colors[0])),dtype=np.float64)
        else:
            raise ValueError('color input of un-recognized type: {}'.format(type(colors[0])))
        for l in unique_labels:
            plt_c[y_true==l] = colors[l]

    # --------------------
    # Do the plotting
    # --------------------
    if order is None or (order == 'none') or (order == 'freq'):

        # If sides of chart should be treated independently
        if split_chart:
            # Create a mask for upper-left
            ul_mask = p_values[:,cols[0]] < p_values[:,cols[1]]
            __plot_freq(ax,y_true[ul_mask],p_values[ul_mask],plt_sizes[ul_mask],
                plt_c[ul_mask],plt_markers,unique_labels,
                cols,labels,add_legend=False,**kwargs)
            # Do lower-right 
            __plot_freq(ax,y_true[~ul_mask],p_values[~ul_mask],plt_sizes[~ul_mask],
                plt_c[~ul_mask],plt_markers,unique_labels,
                cols,labels,add_legend=True,**kwargs)
        else:
            # Here look at overall frequency of each class
            __plot_freq(ax,y_true,p_values,
                plt_c,plt_markers,unique_labels,
                cols,labels,add_legend=True,**kwargs)

    elif ('class' in order or 'label' in order) and 'rev' in order:
        # Use the reerse order of the labels 
        rev = np.flip(unique_labels)
        for lab in rev:
            l_mask = y_true == lab
           
            ax.scatter(
                p_values[l_mask, cols[0]], 
                p_values[l_mask, cols[1]], 
                s = plt_sizes[l_mask], 
                color = plt_c[l_mask], 
                marker = plt_markers[lab], 
                label = labels[lab],
                **kwargs)
    elif order == 'class' or order == 'label' :
        # Use the order of the labels 
        for lab in unique_labels:
            l_mask = y_true == lab
            
            ax.scatter(
                p_values[l_mask, cols[0]], 
                p_values[l_mask, cols[1]], 
                s = plt_sizes[l_mask], 
                color = plt_c[l_mask], 
                marker = plt_markers[lab],
                label = labels[lab],
                **kwargs)
    else:
        raise ValueError('parameter order not any of the allowed values: ' + str(order))
    # --------------------
    # End of plotting 
    # --------------------

    if add_legend:
        ax.legend(loc='upper right',**{'fontsize':'large',**fontargs})
    
    # Set some labels and title
    if y_label is not None:
        y_label = y_label.replace('{class}', str(labels[cols[1]]))
        ax.set_ylabel(y_label,**{'fontsize':'large',**fontargs})

    if x_label is not None:
        x_label = x_label.replace('{class}', str(labels[cols[0]]))
        ax.set_xlabel(x_label,**{'fontsize':'large',**fontargs})

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
    flip_x = False,
    flip_y = False,
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
    
    flip_x : bool, default False
        If the x-axes should display significance level (`False`) or confidence (`True`)
    
    flip_y : bool, default False
        If the y-axes should display error-rate (`False`) or accuracy (`True`)

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
    if not isinstance(sign_vals,np.ndarray):
        sign_vals = np.array(sign_vals)
    
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

    # Find the x and y ranges and set the chart-size
    y_vals = np.concatenate((overall_frac,sign_vals))
    # x_vals = np.array([np.min(sign_vals), np.max(sign_vals)])
    # y_vals = np.array([np.min(overall_frac),(overall_frac)])
    # y_vals = np.append(y_vals,x_vals)
    # else:
    #     x_vals = [np.min(sign_vals), np.max(sign_vals)]
    # if flip_y:
    #     y_vals = [1-np.min(overall_frac), np.max(overall_frac)]+ x_vals
    # print(y_vals.shape)
    if plot_all_labels:
        y_vals = np.concatenate((y_vals,cls_frac.reshape(-1)))
    # print(x_vals)
    # print(y_vals)
        # y_vals.append(np.max(cls_frac))
    _set_chart_size(ax,sign_vals,y_vals,
        padding=chart_padding,
        flip_x=flip_x,
        flip_y=flip_y)
    
    (x_lab,y_lab) = add_calib_curve(ax,
        sign_vals,
        overall_frac,
        legend='Overall',
        zorder=100,
        color=overall_color,
        flip_x=flip_x,
        flip_y=flip_y,
        set_chart_size=False,
        plot_expected=True,
        **kwargs)

    if plot_all_labels:
        colors = cm_as_list(cm, __default_color_map)
        for i in range(cls_frac.shape[1]):
            add_calib_curve(ax,
                sign_vals,
                cls_frac[:,i],
                legend=labels[i],
                color=colors[i],
                set_chart_size=False,
                plot_expected=False,
                flip_x=flip_x,
                flip_y=flip_y,
                **kwargs)
    
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
    
    ax.scatter(x_coords, y_coords, s=sizes_scaled, c=colors, edgecolors='black', **kwargs)

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


