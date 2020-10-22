import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import math
import numpy as np
import pandas as pd
from .metrics import calc_error_rate,calc_single_label_preds
from .metrics import calc_multi_label_preds,calc_confusion_matrix
from .metrics import calc_multi_label_preds_ext,calc_single_label_preds_ext


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
### INTERNAL FUNCTIONS
####################################

def __get_significance_values(sign_min=0,sign_max=1,sign_step=0.01):
    '''Internal function for generation of a list of significance values
    
    Returns
    -------
    list of float
    '''
    
    # Do some validation
    if sign_min<0:
        sign_min = 0
    if sign_max > 1:
        sign_max = 1
    if sign_max < sign_min:
        raise ValueError('sign_max < sign_min not allowed')
    if sign_step < 1e-4 or sign_step > 1:
        sign_step = 0.01
    
    significances = list(np.arange(sign_min,sign_max,sign_step))
    if significances[-1] < sign_max:
        significances.append(sign_max)
    return significances

def __get_fig_and_axis(ax, fig_size = (10,8)):
    '''Internal function for instantiating a Figure / axes object
    
    Returns
    -------
    tuple (Figure, axes)
    '''
    
    if ax is None:
        # No current axes, create a new Figure
        fig = plt.figure(figsize = fig_size)
        # Add an axes spanning the entire Figure
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    
    return fig, ax

####################################
### CLASSIFICATION
####################################

def plot_pvalues(true_labels,
                    p_values,
                    ax = None,
                    fig_size = (10,8),
                    cm = None,
                    markers = None,
                    sizes = None,
                    labels = None,
                    title = None,
                    order = None,
                    x_label = 'p-value 0',
                    y_label = 'p-value 1',
                    fontargs = None,
                    **kwargs):
    '''
    Plot p0 vs p1 (or others if multiclass)
    
    Parameters
    ----------
    
    Returns
    -------
    Figure
        matplotlib.figure.Figure object
    '''

    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values must be a numpy ndarray')
    if p_values.shape[1] != 2:
        raise ValueError('p_values must be a (n_examples,2) shaped numpy ndarray')
    
    fig, ax = __get_fig_and_axis(ax, fig_size)

    # Set the color-map
    if cm is None:
        pal = __default_color_map
    else:
        pal = list(cm)
    # Color individual examples
    # Find all unique labels
    if len(pal) == 1:
        # Single color for all
        colors = [pal[0]]*len(true_labels)
    else:
        # Unique color for each example
        colors = []
        for ex in true_labels:
            colors.append(pal[int(ex) % len(pal)])
    colors = np.array(colors)

    unique_labels = sorted(list(np.unique(true_labels).astype(int)))

    # Verify the labels
    if labels is not None:
        if not isinstance(labels, list):
            raise TypeError('labels must be a list if supplied')
        if len(labels) != len(unique_labels):
            raise TypeError('labels and number of classes does not match') 
    
    # Set the markers to a list
    if markers is None:
        # default
        plt_markers = [mpl.rcParams["scatter.marker"]]
    elif not isinstance(markers, list):
        # not a list - same marker for all
        plt_markers = [markers]
    else:
        # Unique markers for all
        plt_markers = markers
    
    # Set the size of the markers
    if sizes is None:
        plt_sizes = [None] #[mpl.rcParams['lines.markersize']] # Default
    elif isinstance(sizes, list):
        plt_sizes = sizes
    else:
        plt_sizes = [sizes]
    
    # Do the plotting
    if order is not None and order.lower() == 'order':
        # Use the order of the labels simply 
        for lab in unique_labels:
            label_filter = np.array(true_labels == lab)
            x = p_values[label_filter, 0]
            y = p_values[label_filter, 1]
            c = colors[label_filter]
            m = plt_markers[int(lab) % len(plt_markers)]
            #if plt_sizes is None:
            #    s = None
            #else:
            s = plt_sizes[int(lab) % len(plt_sizes)]
            label = None
            if labels is not None:
                label = labels[int(lab)]
            ax.scatter(x, y, s=s, c=c, marker = m, label=label, **kwargs)
    else:
        # Try to use a smarter apporach
        unique_labels = np.array(unique_labels)
        num_ul = []
        num_lr = []
        num_per_label = []
        # Compute the order of classes to plot
        for lab in unique_labels:
            label_filter = np.array(true_labels == lab)
            x = p_values[label_filter, 0]
            y = p_values[label_filter, 1]
            num_per_label.append(len(x))
            # the number in lower-right part and upper-left
            num_lr.append((x>y).sum())
            num_ul.append(x.shape[0] - num_lr[-1])
        
        # For small datasets, normalize for size
        if order is not None and order.lower() == 'rel':
            num_ul = np.array(num_ul) / np.array(num_per_label)
            num_lr = np.array(num_lr) / np.array(num_per_label)
        
        # Plot the upper-left section
        ul_sorting = np.argsort(num_ul)[::-1]
        for lab in unique_labels[ul_sorting]:
            label_filter = np.array(true_labels == lab)
            x = p_values[label_filter, 0]
            y = p_values[label_filter, 1]
            upper_left_filter = y >= x
            x = x[upper_left_filter]
            y = y[upper_left_filter]
            c = colors[label_filter][upper_left_filter]
            m = plt_markers[int(lab) % len(plt_markers)]
            s = plt_sizes[int(lab) % len(plt_sizes)] #[None if plt_sizes is None else
            label = None
            if labels is not None:
                label = labels[int(lab)]
            ax.scatter(x, y, s=s, c=c, marker = m, label=label, **kwargs)

        # Plot the lower-right section
        lr_sorting = np.argsort(num_lr)[::-1]
        for lab in unique_labels[lr_sorting]:
            label_filter = np.array(true_labels == lab)
            x = p_values[label_filter, 0]
            y = p_values[label_filter, 1]
            lower_right_filter = y < x
            x = x[lower_right_filter]
            y = y[lower_right_filter]
            c = colors[label_filter][lower_right_filter]
            m = plt_markers[int(lab) % len(plt_markers)]
            s = plt_sizes[int(lab) % len(plt_sizes)] #[None if plt_sizes is None else
            # Skip the labels in second one!
            ax.scatter(x, y, s=s, c=c, marker = m, **kwargs)

    ax.set_ylim([-.025, 1.025])
    ax.set_xlim([-.025, 1.025])
    
    if labels is not None:
        if fontargs is not None:
            ax.legend(loc='upper right',**fontargs)
        else:
            ax.legend(loc='upper right')
    
    # Set some labels and title
    if y_label is not None:
        if fontargs is not None:
            ax.set_ylabel(y_label,**fontargs)
        else:
            ax.set_ylabel(y_label)
    if x_label is not None:
        if fontargs is not None:
            ax.set_xlabel(x_label,**fontargs)
        else:
            ax.set_xlabel(x_label)
    if title is not None:
        if fontargs is None:
            ax.set_title(title, {'fontsize': 'x-large'})
        else:
            ax.set_title(title, **fontargs)

    return fig


def plot_calibration_curve(true_labels,
                            p_values,
                            ax = None,
                            fig_size = (10,8),
                            title = None,
                            significance_min=0,
                            significance_max=1,
                            significance_step=0.01,
                            significance_values=None,
                            cm = None,
                            fig_padding=None,
                            plot_all_labels=True,
                            class_labels=None,
                            title=None,
                            **kwargs):
    
    '''Create a calibration curve **(Classification)**
    
    Parameters
    ----------
    true_labels : list or 1D numpy array
        The true labels (with values 0, 1, etc for each class)
    p_values : A 2D numpy array
        P-values, first column with p-value for class 0, second for class 1, ..
    ax : matplotlib Axes, optional
        An existing matplotlib Axes to plot in (default None)
    fig_size : tuple of 2 values, optional
        Figure size that should be generated, ignored if *ax* is given
    title : str, optional
        A title to add to the figure (default None)
    significance_min : float range [0,1], optional
        The smallest significance level to include (default 0)
    significance_max : float range [0,1], optional
        The largest significance level to include (default 1)
    significance_step : float > 1, optional
        Spacing between evaulated significance levels (default 0.01)
    significance_values : list of float, optional
        A list of significance values to use, the `significance_` values will be ignored if this parameter was passed (default None)
    cm : colormap (list of colors), optional
        The color-mapping to use. First color will the the overall, then the classes 0,1,..
    fig_padding : float, optional
        Padding added to the drawing area (default None will add 2.5% of padding)
    plot_all_labels : boolean, optional
        Plot the error rates for each class (default True). If False, only the 'overall' error rate is plotted
    class_labels : list of str, optional
        Descriptive labels for the classes (default will use 0, 1, .. for the classes)
    **kwargs : kwargs, optional
        kwargs dict passed to matplotlib
    
    Returns
    -------
    Figure
        matplotlib.figure.Figure object
    '''
    
    # Create a list with all significances 
    significances = __get_significance_values(significance_min,significance_max,significance_step)
    sign_max = significances[len(significances)-1]
    sign_min = significances[0]
    
    if class_labels is not None:
        if len(class_labels) != p_values.shape[1]:
            raise ValueError('Number of class labels must be equal to number of p-values ' + 
                                 str(len(class_labels)) + " != " + str(p_values.shape[1]))
    if cm is None:
        pal = __default_color_map
    else:
        pal = list(cm)

    if fig_padding is None:
        fig_padding = (sign_max - sign_min)*0.025
    
    overall_error_rates = []
    if plot_all_labels:
        label_based_rates = np.zeros((len(significances), p_values.shape[1]))
    
    for ind, s in enumerate(significances):
        overall, label_based = calc_error_rate(true_labels,p_values,s)
        overall_error_rates.append(overall)
        if plot_all_labels:
            # sets all values in a row
            label_based_rates[ind] = label_based
    
    error_fig, ax = __get_fig_and_axis(ax, fig_size)

    ax.axis([sign_min-fig_padding, sign_max+fig_padding, sign_min-fig_padding, sign_max+fig_padding]) 
    ax.plot(significances, significances, '--k', alpha=0.25, linewidth=1)
    
    if plot_all_labels:
        # Specify a higher zorder for the overall to print it on the top (most important)
        ax.plot(significances, overall_error_rates,c="black", label='Overall',zorder=10, **kwargs)
        for i in range(label_based_rates.shape[1]):
            label = 'Label ' + str(i)
            if class_labels is not None:
                label = class_labels[i]
            ax.plot(significances,label_based_rates[:,i], c=pal[i+1], label=str(label),**kwargs)
        ax.legend(loc='lower right')
    else:
        ax.plot(significances, overall_error_rates,c=pal[0], **kwargs)
    
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Significance")
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})
    
    return error_fig

def plot_label_distribution(true_labels,
                            p_values,
                            ax=None,
                            fig_size=(10,8),
                            title=None,
                            significance_min=0,
                            significance_max=1,
                            significance_step=0.01,
                            significance_values=None,
                            cm=None,
                            display_incorrects=False,
                            mark_best=True,
                            **kwargs):
    '''Create a stacked plot with label ratios **(Classification)**
    
    Parameters
    ----------
    true_labels : A list or 1D numpy array
        The true classes/labels using values 0, 1, etc for each class
    p_values : 2D numpy array
        The predicted p-values, first column for the class 0, second for class 1, ..
    ax : matplotlib Axes, optional
        An existing matplotlib Axes to plot in (default None)
    fig_size : tuple of 2 values, optional
        Figure size that should be generated, ignored if *ax* is given
    title : str, optional
        A title to add to the figure (default None)
    significance_min : float range [0,1], optional
        The smallest significance level to include (default 0)
    significance_max : float range [0,1], optional
        The largest significance level to include (default 1)
    significance_step : float > 1, optional
        Spacing between evaulated significance levels (default 0.01)
    significance_values : list of float, optional
        A list of significance values to use, the *significance_* values will be ignored if this parameter was passed (default None)
    cm : colormap (list of colors), optional
        A color map (list of colors) in the order: [single, multi, empty, (incorrect-single), (incorrect-multi)]  (if *display_incorrects=True* a list of at least 5 is required, otherwise 3 is sufficient)
    display_incorrects : boolean, optional
        Plot the incorrect predictions intead of only empty/singletons/multi-label predictions (default True)
    mark_best : boolean
        Mark the best significance value with a line and textbox (default True)
    **kwargs : kwargs
        kwargs dict passed to matplotlib
    
    Returns
    -------
    Figure
        matplotlib.figure.Figure object
    '''
    
    # Set color-mapping
    if cm is not None:
        pal = list(cm)
    else:
        pal = [__default_single_label_color, __default_multi_label_color, __default_empty_prediction_color, __default_incorr_single_label_color,__default_incorr_multi_label_color]

    # Create a list with all significances
    if significance_values is not None:
        # Validate the given significance values
        
    else:
        significances = __get_significance_values(significance_min,significance_max,significance_step)
        sign_max = significances[-1]
        sign_min = significances[0]
    
    # Calculate the values
    s_label = []
    si_label = [] # Incorrects
    m_label = []
    mi_label = [] # Incorrects
    empty_label = []
    highest_single_ratio = -1
    best_sign = -1
    for s in significances:
        s_corr, s_incorr = calc_single_label_preds_ext(true_labels, p_values,s)
        m_corr, m_incorr = calc_multi_label_preds_ext(true_labels, p_values, s)
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
    
    fig, ax = __get_fig_and_axis(ax, fig_size)
    
    ax.axis([sign_min,sign_max,0,1])
    ax.stackplot(significances,ys,
                  labels=labels, 
                  colors=colors,**kwargs) #
    
    if mark_best:
        ax.plot([best_sign,best_sign],[0,1],'--k')
        props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        ax.text(best_sign+0.02, 0.1, str(best_sign), bbox=props)
    
    ax.legend()
    
    ax.set_ylabel("Label distribution")
    ax.set_xlabel("Significance")
    if title is not None:
        ax.set_title(title, {'fontsize': 'x-large'})
    
    return fig


def plot_confusion_matrix_bubbles(confusion_matrix,
                                  ax=None,
                                  fig_size=(10,8),
                                  title=None,
                                  bubble_size_scale_factor = 2500,
                                  color_scheme = 'prediction_size',
                                  **kwargs):
    '''Create a Bubble plot over predicted labels at a fixed significance level (Classification)
    
    Parameters
    ----------
    confusion_matrix -- A precomputed confusion matrix in pandas DataFrame, from pharmbio.cp.metrics.calc_confusion_matrix
    ax -- (Optional) An existing matplotlib Axes to plot in
    fig_size -- (Optional) Figure size, ignored if *ax* is given
    bubble_size_scale_factor -- (Optional) Scaling to be applied on the size of the bubbles, default scaling works OK for the default figure size
    color_scheme -- (Optional) String - allowed values: 
        None/'None':=All in the same color 
        'prediction_size':=Color single/multi/empty in different colors
        'label'/'class':=Each class colored differently
        'full':=Correct single, correct multi, incorrect single, incorrect multi and empty colored differently
    **kwargs -- kwargs passed along to matplot-lib
    
    Returns
    -------
    Figure
        matplotlib.figure.Figure object
    '''
    
    # Make sure CM exists
    if confusion_matrix is None:
        # We need to calculate the CM
        if true_labels is None or p_values is None or significance is None:
            raise TypeError('Either a precomputed confusion matrix or {labels,p_values,significance} must be sent')
        if class_labels is not None:
            confusion_matrix = calc_confusion_matrix(true_labels, p_values,significance, class_labels=class_labels)
        else:
            confusion_matrix = calc_confusion_matrix(true_labels, p_values,significance)
    
    if not isinstance(confusion_matrix, pd.DataFrame):
        raise TypeError('argument confusion_matrix must be a DataFrame - otherwise give labels and p-values so it can be generated')

    # Create Figure and Axes if not given
    fig, ax = __get_fig_and_axis(ax, fig_size)
    
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
    sizes_scaled = bubble_size_scale_factor * sizes / sizes.max()
    
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
    
    return fig


def plot_heatmap(confusion_matrix, 
                 ax=None, fig_size=(10,8), 
                 title=None,
                 cbar_kws=None,
                 **kwargs):
    '''Plots the Conformal Confusion Matrix in a Heatmap **(Classification)**
    
    Parameters
    ----------
    confusion_matrix -- A precomputed confusion matrix in pandas DataFrame, from pharmbio.cp.metrics.calc_confusion_matrix
    ax -- (Optional) An existing matplotlib Axes to plot in
    fig_size -- (Optional) Figure size as a tuple, ignored if *ax* is given
    title -- (Optional) An optional title that will be printed in 'x-large' font size
    cbar_kws -- (Optional) Arguments passed to the color-bar element
    **kwargs -- kwargs passed along to matplotlib
    
    Returns
    -------
    Figure
        matplotlib.figure.Figure object
    '''
    if not __using_seaborn:
        raise RuntimeException('Seaborn is required when using this function')
    
    fig, ax = __get_fig_and_axis(ax, fig_size)
    
    if title is not None:
        ax.set_title(title, fontdict={'fontsize':'x-large'})
    
    ax = sns.heatmap(confusion_matrix, ax=ax, annot=True,cbar_kws=cbar_kws, **kwargs)
    ax.set(xlabel='Predicted', ylabel='Observed')
    
    return fig


