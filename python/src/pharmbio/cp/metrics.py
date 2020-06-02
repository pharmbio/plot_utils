import numpy as np

__version__ = '0.0.1'

####################################
### CLASSIFICATION
####################################

def calc_error_rate(true_labels, p_values, sign):
    '''Calculate the error rate (classification)
    
    Arguments:
    true_labels -- A 1D numpy array, with values 0, 1, etc for each class
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    sign -- the significance the metric should be calculated for
    
    returns -- tuple (overall_error_rate, label_wise_error_rates) 
    '''
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values argument must be a numpy ndarray')
    
    if (len(true_labels) != p_values.shape[0]):
        raise ValueException('arguments true_labels and p_values must have the same length')
    
    total_errors = 0
    # lists containing errors/counts for each class label
    label_wise_errors = [0] * p_values.shape[1]
    label_wise_counts = [0] * p_values.shape[1]
    
    for test_ex in range(0,p_values.shape[0]):
        ex_value = true_labels[test_ex]
        if p_values[test_ex, ex_value] < sign:
            total_errors += 1
            label_wise_errors[ex_value] += 1
        label_wise_counts[ex_value] += 1
    
    label_wise_erro_rate = np.array(label_wise_errors) / np.array(label_wise_counts)
    
    return (total_errors / true_labels.shape[0], label_wise_erro_rate)

def calc_single_label_preds(true_labels, p_values, sign):
    '''Calculate the fraction of single label predictions (classification)
    
    Arguments:
    true_labels -- A list or 1D numpy array, with values 0, 1, etc for each class
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    sign -- the significance the metric should be calculated for
    
    returns -- The fraction of single label predictions (single value) 
    '''
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values argument must be a numpy ndarray')
    
    if (len(true_labels) != p_values.shape[0]):
        raise ValueException('arguments true_labels and p_values must have the same length')
    
    single_labels = 0
    for i in range(0,p_values.shape[0]):
        if (p_values[i,:] > sign).sum() == 1:
            single_labels += 1
    return single_labels / len(true_labels)

def calc_multi_label_preds(true_labels, p_values, sign):
    '''Calculate the fraction of multi-label predictions (classification)
    
    Arguments:
    true_labels -- A list or 1D numpy array, with values 0, 1, etc for each class
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    sign -- the significance the metric should be calculated for
    
    returns -- The fraction of multi-label predictions
    '''
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values argument must be a numpy ndarray')
    
    if (len(true_labels) != p_values.shape[0]):
        raise ValueException('arguments true_labels and p_values must have the same length')
    
    multi_labels = 0
    for i in range(0,p_values.shape[0]):
        if (p_values[i,:] > sign).sum() > 1:
            multi_labels += 1
    return multi_labels / len(true_labels)

def calc_OF(true_labels, p_values):
    ''' Calculates the Observed Fuzziness (significance independent)
    
    Arguments:
    true_labels -- A list or 1D numpy array, with values 0, 1, etc for each class
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    
    returns -- The Observed Fuzziness 
    '''
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values argument must be a numpy ndarray')
    
    if (len(true_labels) != p_values.shape[0]):
        raise ValueException('arguments true_labels and p_values must have the same length')
    
    of_sum = 0
    for i in range(0,p_values.shape[0]):
        # Mask the p-value of the true label
        p_vals_masked = np.ma.array(p_values[i,:], mask=False)
        p_vals_masked.mask[true_labels[i]] = True
        # Sum the remaining p-values
        of_sum += p_vals_masked.sum()
    
    return of_sum / len(true_labels)
    
    

####################################
### REGRESSION - TODO
####################################

def calc_error_rate_regression(true_labels, prediction_ranges):
    # what is the best way to do this? prediction-ranges as 2D numpy array? [lower bound, upper bound] ?
    raise NotImplementedError('TODO')
