import matplotlib.pyplot as plt
#import math
import numpy as np
from .metrics import calc_error_rate


__using_seaborn = False
# Try to import sns as they create somewhat nicer plots
try:
    import seaborn as sns; 
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

def plot_calibration_curve(true_labels, p_values, 
                           sign_min=0, sign_max=1,
                           sign_step=0.01, fig_padding=None,
                           plot_all_labels=True,
                           fig_size = (10,8),
                           class_labels=None,
                           **kwargs):
    
    '''Create a calibration curve (Classification)
    
    
    
    '''
    # Do some validation
    if sign_min<0:
        sign_min = 0
    if sign_max > 1:
        sign_max = 1
    if sign_max < sign_min:
        raise ValueException('sign_max < sign_min not allowed')
    if sign_step < 1e-4 or sign_step > 1:
        sign_step = 0.01
    if class_labels is not None:
        if len(class_labels) != p_values.shape[1]:
            raise ValueException('Number of class labels must be equal to number of p-values ' + 
                                 str(len(class_labels)) + " != " + str(p_values.shape[1]))
    
    if fig_padding is None:
        fig_padding = (sign_max - sign_min)*0.025
    
    # Create a list with all significances 
    significances = list(np.arange(sign_min,sign_max,sign_step))
    if significances[len(significances)-1] < sign_max:
        significances.append(sign_max)
    
    overall_error_rates = []
    if plot_all_labels:
        label_based_rates = np.zeros((len(significances), p_values.shape[1]))
    
    for ind, s in enumerate(significances):
        overall, label_based = calc_error_rate(true_labels,p_values,s)
        overall_error_rates.append(overall)
        if plot_all_labels:
            # sets all values in a row
            label_based_rates[ind] = label_based
    
    error_fig = plt.figure(figsize = fig_size)
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
    
