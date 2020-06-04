import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from .metrics import calc_error_rate,calc_single_label_preds
from .metrics import calc_multi_label_preds,calc_confusion_matrix


__using_seaborn = False
# Try to import sns as they create somewhat nicer plots
try:
    import seaborn as sns
    sns.set()
    print('Using Seaborn plotting defaults')
    __using_seaborn = True
except ImportError as e:
    print('Seaborn not available - using default Matplot-lib settings')
    pass 

__version__ = '0.0.1'


####################################
### CLASSIFICATION
####################################

def __get_significance_values(sign_min=0,sign_max=1,sign_step=0.01):
    '''Internal function for generation of a list of significance values
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
    if significances[len(significances)-1] < sign_max:
        significances.append(sign_max)
    return significances

def plot_calibration_curve(true_labels, p_values, 
                           figure = None, fig_size = (10,8),
                           significance_min=0, significance_max=1,
                           significance_step=0.01, fig_padding=None,
                           plot_all_labels=True,
                           class_labels=None,
                           **kwargs):
    
    '''Create a calibration curve (Classification)
    
    Arguments:
    true_labels -- A list or 1D numpy array, with values 0, 1, etc for each class
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    figure -- (Optional) An existing matplotlib figure to plot in
    fig_size -- (Optional) Figure size, ignored if *figure* is given
    significance_min -- (Optional) The smallest significance level to include
    significance_max -- (Optional) The largest significance level to include
    significance_step -- (Optional) The spacing between calculated values
    fig_padding-- (Optional) The padding added to the drawing area, *None* will add 2.5% of padding 
    plot_all_labels -- (Optional) If *True* all classes will get their own calibration curve plotted together with the 'overall' error rate, if *False* only the overall will be plotted
    class_labels -- (Optional) Descriptive labels for the classes 
    **kwargs -- kwargs passed along to matplot-lib
    
    '''
    # Create a list with all significances 
    significances = __get_significance_values(significance_min,significance_max,significance_step)
    sign_max = significances[len(significances)-1]
    sign_min = significances[0]
    
    if class_labels is not None:
        if len(class_labels) != p_values.shape[1]:
            raise ValueError('Number of class labels must be equal to number of p-values ' + 
                                 str(len(class_labels)) + " != " + str(p_values.shape[1]))
    
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
    
    if figure is None:
        error_fig = plt.figure(figsize = fig_size)
    else:
        error_fig = figure
        plt.figure(error_fig.number)
    
    plt.axis([sign_min-fig_padding, sign_max+fig_padding, sign_min-fig_padding, sign_max+fig_padding]) 
    plt.plot(significances, significances, '--k')
    
    if plot_all_labels:
        plt.plot(significances, overall_error_rates,label='overall',**kwargs)
        for i in range(label_based_rates.shape[1]):
            label = 'label ' + str(i)
            if class_labels is not None:
                label = class_labels[i]
            plt.plot(significances,label_based_rates[:,i], label=str(label),**kwargs)
        plt.legend(loc='lower right')
    else:
        plt.plot(significances, overall_error_rates,**kwargs)
    
    plt.ylabel("Error rate")
    plt.xlabel("Significance")
    
    return error_fig


def plot_label_distribution(true_labels, p_values, 
                            figure=None, fig_size=(10,8),
                            significance_min=0, significance_max=1,
                            significance_step=0.01,
                            single_label_color='green', 
                            multi_label_color='red', 
                            empty_label_color='white', 
                            mark_best=True,
                            **kwargs):
    '''Create a stacked plot with label ratios (Classification)
    
    Arguments:
    true_labels -- A list or 1D numpy array, with values 0, 1, etc for each class
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    figure -- (Optional) An existing matplotlib figure to plot in
    fig_size -- (Optional) Figure size, ignored if *figure* is given
    significance_min -- (Optional) The smallest significance level to include
    significance_max -- (Optional) The largest significance level to include
    significance_step -- (Optional) The spacing between calculated values
    single_label_color -- (Optional) The color to use for the area with single-label predictions
    multi_label_color -- (Optional) The color to use for the area with multi-label predictions
    empty_label_color -- (Optional) The color to use for the area with empty predictions
    mark_best -- (Optional) If *True* adds a line and textbox with the significance with the largest ratio of single-label predictions
    **kwargs -- kwargs passed along to matplot-lib
    '''
    # Create a list with all significances 
    significances = __get_significance_values(significance_min,significance_max,significance_step)
    sign_max = significances[len(significances)-1]
    sign_min = significances[0]
    
    # Calculate the values
    s_label = []
    m_label = []
    empty_label = []
    highest_single_ratio = -1
    best_sign = -1
    for s in significances:
        s_l = calc_single_label_preds(true_labels, p_values, s)
        if s_l > highest_single_ratio:
            highest_single_ratio = s_l
            best_sign = s
        m_l = calc_multi_label_preds(true_labels,p_values,s)
        s_label.append(s_l)
        m_label.append(m_l)
        empty_label.append(1 - s_l - m_l)
    
    y = [s_label, m_label, empty_label]
    
    if figure is None:
        fig = plt.figure(figsize = fig_size)
    else:
        fig = figure
        plt.figure(fig.number)
    
    plt.axis([sign_min,sign_max,0,1])
    plt.stackplot(significances,y,
                  labels=['Single-label predictions','Multi-label predictions','Empty predictions'], 
                  colors=[single_label_color,multi_label_color,empty_label_color],**kwargs)
    
    if mark_best:
        plt.plot([best_sign,best_sign],[0,1],'--k')
        props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        plt.text(best_sign+0.02, 0.1, str(best_sign), bbox=props)
    
    plt.legend()
    
    plt.ylabel("Label distribution")
    plt.xlabel("Significance")
    
    return fig


def plot_confusion_matrix_bubbles(confusion_matrix,
                                  figure=None, fig_size=(10,8),
                                  bubble_size_scale_factor = 2500,
                                  **kwargs):
    '''Create a Bubble plot over predicted labels at a fixed significance (Classification)
    
    Arguments:
    confusion_matrix -- A precomputed confusion matrix in pandas DataFrame, from pharmbio.cp.metrics.calc_confusion_matrix
    figure -- (Optional) An existing matplotlib figure to plot in
    fig_size -- (Optional) Figure size, ignored if *figure* is given
    bubble_size_scale_factor -- (Optional) Scaling to be applied on the size of the bubbles, default scaling works OK for the default figure size
    **kwargs -- kwargs passed along to matplot-lib
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

    # Make sure figure is set
    if figure is None:
        figure = plt.figure(figsize = fig_size)
    else:
        plt.figure(fig.number)
    
    x_coords = []
    y_coords = []
    sizes = confusion_matrix.to_numpy().ravel(order='F')
    n_rows = confusion_matrix.shape[0]
    for x in confusion_matrix.columns:
        x_coords.extend([x]*n_rows)
        y_coords.extend(confusion_matrix.index)
    
    # Convert the x and y coordinates to strings
    x_coords = np.array(x_coords, dtype=object).astype(str)
    y_coords = np.array(y_coords, dtype=object).astype(str)
    sizes_scaled = bubble_size_scale_factor * sizes / sizes.max()
    
    plt.scatter(x_coords, y_coords, s=sizes_scaled,**kwargs)
    plt.margins(.3)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.gca().invert_yaxis()

    for xi, yi, zi, z_si in zip(x_coords, y_coords, sizes, sizes_scaled):
        if isinstance(zi, float):
            zi = round(zi,2)
        plt.annotate(zi, xy=(xi, yi), xytext=(np.sqrt(z_si)/2.+5, 0),
                 textcoords="offset points", ha="left", va="center")
    
    return figure


def plot_heatmap(confusion_matrix, 
                 figure=None, fig_size=(10,8), title=None,
                 cbar_kws=None,
                 **kwargs):
    '''Plots the Conformal Confusion Matrix in a Heatmap (Classification)
    
    Arguments:
    confusion_matrix -- A precomputed confusion matrix in pandas DataFrame, from pharmbio.cp.metrics.calc_confusion_matrix
    figure -- (Optional) An existing matplotlib figure to plot in
    fig_size -- (Optional) Figure size as a tuple, ignored if *figure* is given
    title -- (Optional) An optional title that will be printed in 'x-large' font size
    cbar_kws -- (Optional) Arguments passed to the color-bar element
    **kwargs -- kwargs passed along to matplotlib
    
    '''
    if not __using_seaborn:
        raise RuntimeException('Seaborn is required when using this function')
    
    if figure is None:
        figure = plt.figure(figsize = fig_size)
    else:
        plt.figure(figure.number)
    
    if title is not None:
        plt.title(title, fontdict={'fontsize':'x-large'})
    
    ax = sns.heatmap(confusion_matrix, annot=True,cbar_kws=cbar_kws, **kwargs)
    ax.set(xlabel='Predicted', ylabel='Observed')
    
    return figure


