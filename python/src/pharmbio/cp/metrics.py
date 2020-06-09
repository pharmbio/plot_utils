import numpy as np
import pandas as pd
from collections import Counter


__default_significance = 0.8

####################################
### UTILS
####################################

def __convert_to_1D_or_raise_error(labels):
    error_obj = TypeError('labels argument must be either a list or numpy 1D array')
    if isinstance(labels, list):
        return np.array( [round(x) for x in labels] )
    elif isinstance(labels, np.ndarray):
        # Make sure the shape is correct
        if len(labels.shape) == 1:
            # 1D numpy - correct!
            return labels.astype(int)
        elif labels.shape[1] > 1:
            raise error_obj
        else:
            # convert to 1D array
            return np.squeeze(labels).astype(int)
    else:
        raise error_obj
   

######################################
### CLASSIFICATION - OBSERVED METRICS
######################################


def calc_error_rate(true_labels, p_values, sign):
    '''Calculate the error rate (classification)
    
    Arguments:
    true_labels -- A 1D numpy array, with values 0, 1, etc for each class. note that dtype must be integer!
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    sign -- the significance the metric should be calculated for
    
    returns -- tuple (overall_error_rate, label_wise_error_rates) 
    '''
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values argument must be a numpy ndarray')
    
    if (len(true_labels) != p_values.shape[0]):
        raise ValueError('arguments true_labels and p_values must have the same length')
    
    true_labels = __convert_to_1D_or_raise_error(true_labels)

    total_errors = 0
    # lists containing errors/counts for each class label
    label_wise_errors = [0] * p_values.shape[1]
    label_wise_counts = [0] * p_values.shape[1]
    
    for test_ex in range(0,p_values.shape[0]):
        ex_value = true_labels[test_ex]
        #print(ex_value)
        if p_values[test_ex, ex_value] < sign:
            total_errors += 1
            label_wise_errors[ex_value] += 1
        label_wise_counts[ex_value] += 1
    
    label_wise_erro_rate = np.array(label_wise_errors) / np.array(label_wise_counts)
    
    return (total_errors / true_labels.shape[0], label_wise_erro_rate)


 
    #true_labels = __convert_to_1D_or_raise_error(true_labels)
    # 
    #multi_labels = 0
    #for i in range(0,p_values.shape[0]):
    #    if (p_values[i,:] > sign).sum() > 1:
    #        multi_labels += 1
    #return multi_labels / len(true_labels)

def calc_single_label_preds_ext(true_labels, p_values, sign):
    '''Calculate the fraction of single label predictions (classification), but calculating the correct and incorrect classifications
    
    Arguments:
    true_labels -- A list or 1D numpy array, with values 0, 1, etc for each class
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    sign -- the significance the metric should be calculated for
    
    returns -- Tuple (ratio correct single label, ratio incorrect single label) 
    '''
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values argument must be a numpy ndarray')
    
    if (len(true_labels) != p_values.shape[0]):
        raise ValueError('arguments true_labels and p_values must have the same length')

    true_labels = __convert_to_1D_or_raise_error(true_labels)
    n_total = len(true_labels)

    predictions = p_values > sign
    s_label_filter = np.sum(predictions, axis=1) == 1
    s_preds = predictions[s_label_filter]
    s_trues = true_labels[s_label_filter]

    n_corr = 0
    n_incorr = 0
    for i in range(0, s_trues.shape[0]):
        if s_preds[i, s_trues[i]]:
            n_corr +=1
        else:
            n_incorr += 1 
    
    return n_corr/n_total, n_incorr/n_total

def calc_multi_label_preds_ext(true_labels, p_values, sign):
    '''Calculate the fraction of multi-label predictions (classification), but calculating the correct and incorrect classifications
    
    Arguments:
    true_labels -- A list or 1D numpy array, with values 0, 1, etc for each class
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    sign -- the significance the metric should be calculated for
    
    returns -- Tuple (ratio correct multi-label, ratio incorrect multi-label) 
    '''
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values argument must be a numpy ndarray')
    
    if (len(true_labels) != p_values.shape[0]):
        raise ValueError('arguments true_labels and p_values must have the same length')

    true_labels = __convert_to_1D_or_raise_error(true_labels)
    n_total = len(true_labels)

    predictions = p_values > sign
    m_label_filter = np.sum(predictions, axis=1) > 1
    m_preds = predictions[m_label_filter]
    m_trues = true_labels[m_label_filter]

    n_corr = 0
    n_incorr = 0
    for i in range(0, m_trues.shape[0]):
        if m_preds[i, m_trues[i]]:
            n_corr +=1
        else:
            n_incorr +=1 
    
    return n_corr/n_total, n_incorr/n_total

    #multi_labels = 0
    #for i in range(0,p_values.shape[0]):
    #    if (p_values[i,:] > sign).sum() > 1:
    #        multi_labels += 1
    #return multi_labels / len(true_labels)

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
        raise ValueError('arguments true_labels and p_values must have the same length')
    
    true_labels = __convert_to_1D_or_raise_error(true_labels)

    of_sum = 0
    for i in range(0,p_values.shape[0]):
        # Mask the p-value of the true label
        p_vals_masked = np.ma.array(p_values[i,:], mask=False)
        p_vals_masked.mask[true_labels[i]] = True
        # Sum the remaining p-values
        of_sum += p_vals_masked.sum()
    
    return of_sum / len(true_labels)

def calc_confusion_matrix(true_labels, p_values, significance, 
                class_labels=None, normalize_per_class = False):
    ''' Calculates a conformal confusion matrix with number of predictions for each class and number of both and none. 

    Arguments:
    true_labels -- A list or 1D numpy array, with values 0, 1, etc for each class
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class
    significance -- The significance value to use, a value between 0 and 1
    class_labels -- (Optional) A list with the class names
    normalize_per_class -- (Optional) Normalizes the count so that each column sums to 1 (good when visualizing imbalanced datasets)
    
    returns -- A Pandas dataframe with a Conformal Confusion Matrix
    '''
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values argument must be a numpy ndarray')
    
    if len(true_labels) != p_values.shape[0]:
        raise ValueError('arguments true_labels and p_values must have the same length')
    if p_values.shape[1] < 2:
        raise ValueError('Number of classes must be at least 2')

    true_labels = __convert_to_1D_or_raise_error(true_labels)
    
    predictions = p_values > significance
    
    n_class = p_values.shape[1]
    
    # We create two different 'multi-label' predictions, either including or excluding the correct label
    if n_class == 2:
        result_matrix = np.zeros((n_class+2, n_class))
    else:
        result_matrix = np.zeros((n_class+3, n_class))
    
    if class_labels is None:
        class_labels = list(range(n_class))
    elif len(class_labels) != n_class:
        raise ValueError('class_labels must have the same length as the number of classes')
    
    # For every observed class - t
    for t in range(n_class):
        
        # Get the predictions for this class
        t_filter = true_labels == t
        t_preds = predictions[t_filter]
        
        # For every (single) predicted label - p
        for p in range(n_class):
            predicted_p = [False]*n_class
            predicted_p[p] = True
            result_matrix[p,t] = (t_preds == predicted_p).all(axis=1).sum()
        
        # Empty predictions for class t
        result_matrix[n_class,t] = ( t_preds.sum(axis=1) == 0 ).sum()
        
        # multi-label predictions for class t
        t_multi_preds = t_preds.sum(axis=1) > 1
        t_num_all_multi = t_multi_preds.sum()
        if n_class == 2:
            result_matrix[n_class+1,t] = t_num_all_multi
        else:
            # For multi-class we have two different multi-sets - correct or incorrect!
            # first do a filter of rows that are multi-labeled then check t was predicted
            t_num_correct_multi = (t_preds[t_multi_preds][:,t] == True).sum()
            t_num_incorrect_multi = t_num_all_multi - t_num_correct_multi
            
            result_matrix[n_class+1,t] = t_num_correct_multi
            result_matrix[n_class+2,t] = t_num_incorrect_multi
    
    row_labels = list(class_labels)
    row_labels.append('Empty')
    if n_class == 2:
        row_labels.append('Both')
    else:
        row_labels.append('Correct Multi-set')
        row_labels.append('Incorrect Multi-set')
    
    if normalize_per_class:
        result_matrix = result_matrix / result_matrix.sum(axis=0)
    else:
        # Convert to int values!
        result_matrix = result_matrix.astype(int)
    
    return pd.DataFrame(result_matrix, columns=class_labels, index = row_labels)

########################################
### CLASSIFICATION - UNOBSERVED METRICS
########################################

def calc_single_label_preds(p_values, sign):
    '''Calculate the fraction of single label predictions (classification)
    
    Arguments:
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    sign -- the significance the metric should be calculated for
    
    returns -- The fraction of single label predictions (single value) 
    '''
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values argument must be a numpy ndarray')
    
    predictions = p_values > sign
    return np.mean(np.sum(predictions, axis=1) == 1)

def calc_multi_label_preds(p_values, sign):
    '''Calculate the fraction of multi-label predictions (classification)
    
    Arguments:
    p_values -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class etc..
    sign -- the significance the metric should be calculated for
    
    returns -- The fraction of multi-label predictions
    '''
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values argument must be a numpy ndarray')
    
    predictions = p_values > sign
    return np.mean(np.sum(predictions, axis=1) > 1)

def __check_p_vals_correct_type(p_values):
    if not isinstance(p_values, np.ndarray):
        raise TypeError('p_values must be a numpy 2D array')
    if len(p_values.shape) < 2:
        raise TypeError('p_values must be a numpy 2D array')
    if p_values.shape[1] < 2:
        raise TypeError('p_values must be a numpy 2D array')

def calc_credibility(p_values):
    '''CP Credibility - Mean of the largest p-values
    '''
    __check_p_vals_correct_type(p_values)
    sorted_matrix = np.sort(p_values, axis=1)
    return np.mean(sorted_matrix[:,-1]) # last index is the largest

def calc_confidence(p_values):
    '''CP Confidence - Mean of 1-'second largest p-value'
    '''
    __check_p_vals_correct_type(p_values)
    sorted_matrix = np.sort(p_values, axis=1)
    return np.mean(1-sorted_matrix[:,-2])

def calc_s_criterion(p_values):
    '''S criterion - Mean of the sum of all pvalues
    '''
    __check_p_vals_correct_type(p_values)
    return np.mean(np.sum(p_values, axis=1))

def calc_n_criterion(p_values, significance=__default_significance):
    '''N criterion - "Number" criterion - the average number of predicted labels

    Significance dependent metric
    '''
    __check_p_vals_correct_type(p_values)
    return np.mean(np.sum(p_values > significance, axis=1))

def calc_u_criterion(p_values):
    '''U criterion - "Unconfidence"

    Smaller values are preferable
    '''
    __check_p_vals_correct_type(p_values)
    sorted_matrix = np.sort(p_values, axis=1)
    return np.mean(sorted_matrix[:,-2])

def calc_f_criteria(p_values):
    '''F criterion - average fuzziness. Average of the sum of all p-values appart from the largest one

    '''
    __check_p_vals_correct_type(p_values)
    sorted_matrix = np.sort(p_values, axis=1)
    if sorted_matrix.shape[1] == 2:
        # Mean of only the smallest p-value
        return np.mean(sorted_matrix[:,0]) 
    else:
        # Here we must take the sum of the values appart from the first column
        return np.mean(np.sum(sorted_matrix[:,:-1], axis=1))

####################################
### REGRESSION - TODO
####################################

