"""CP Classification metrics

Module with classification metrics for CP. See https://arxiv.org/abs/1603.04416 
for references. Note that some metrics are 'unobserved' - i.e. a metric 
that can be calculated without knowing the ground truth (correct) labels
for all predictions. 

"""

import numpy as np
import pandas as pd
from collections import Counter

from ..utils import *
from sklearn.utils import check_consistent_length

_default_significance = 0.8


######################################
### OBSERVED METRICS
######################################


def frac_error(y_true, p_values, sign):
    """**Classification** - Calculate the fraction of errors

    Calculate the fraction of erronious predictions at a given significance level `sign`
    
    Parameters
    ----------
    y_true : 1D numpy array, list or pandas Series
        True labels

    p_values : 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..

    sign : float in [0,1]
        Significance the metric should be calculated for
    
    Returns
    -------
    frac_error : float
        Overall fraction of errors

    label_wise_fraction_error : array, shape = (n_classes,)
        Fraction of errors for each true label, first index for class 0, ...
    """
    validate_sign(sign)
    p_values = to_numpy2D(p_values,'p_values')
    y_true = to_numpy1D_int(y_true, 'y_true')
    
    check_consistent_length(y_true, p_values)

    total_errors = 0
    # lists containing errors/counts for each class label
    label_wise_errors = [0] * p_values.shape[1]
    label_wise_counts = [0] * p_values.shape[1]
    
    for test_ex in range(0,p_values.shape[0]):
        ex_value = y_true[test_ex]
        #print(ex_value)
        if p_values[test_ex, ex_value] < sign:
            total_errors += 1
            label_wise_errors[ex_value] += 1
        label_wise_counts[ex_value] += 1
    
    label_wise_erro_rate = np.array(label_wise_errors) / np.array(label_wise_counts)
    
    return total_errors / y_true.shape[0], label_wise_erro_rate


def _unobs_frac_single_label_preds(p_values, sign):
    """**Classification** - Calculate the fraction of single label predictions
    
    Parameters
    ----------
    p_values : array, 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..

    sign : float in [0,1]
        Significance the metric should be calculated for
    
    Returns
    ------- 
    score : float
    """
    validate_sign(sign)
    p_values = to_numpy2D(p_values,'p_values')
    
    predictions = p_values > sign
    return np.mean(np.sum(predictions, axis=1) == 1)

def frac_single_label_preds(y_true, p_values, sign):
    """**Classification** - Calculate the fraction of single label predictions
    
    It is possible to both calculate this as an observed and un-observed metric,
    the `y_true` is given the function returns three values - if no true values
    are known - only the fraction of multi-label predictions is returned. 
    
    Parameters
    ----------
    y_true : 1D numpy array, list, pandas Series or None
        True labels or None. If given, the fraction of correct and incorrect 
        single label predictions can be calculated as well. Otherwise this will
        be calculated in an unobserved fashion.

    p_values : 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..
    
    sign : float in [0,1]
        Significance the metric should be calculated for
    
    Returns
    -------
    frac_single : float
        Overall fraction of single-labelpredictio
    
    frac_correct_single : float, optional
        Fraction of correct single label predictions, not returned if no `y_true` was given
    
    frac_incorrect_single : float, optional
        Fraction of incorrect single label predictions, not returned if no `y_true` was given
    """
    # If no y_true - calculate in an un-observed fashion
    if y_true is None:
        return _unobs_frac_single_label_preds(p_values, sign),
    
    validate_sign(sign)
    p_values = to_numpy2D(p_values,'p_values')
    y_true = to_numpy1D_int(y_true, 'y_true')
    check_consistent_length(y_true, p_values)

    n_total = len(y_true)

    predictions = p_values > sign
    s_label_filter = np.sum(predictions, axis=1) == 1
    s_preds = predictions[s_label_filter]
    s_trues = y_true[s_label_filter]

    n_corr = 0
    n_incorr = 0
    for i in range(0, s_trues.shape[0]):
        if s_preds[i, s_trues[i]]:
            n_corr +=1
        else:
            n_incorr += 1 
    
    return (n_corr+n_incorr)/n_total, n_corr/n_total, n_incorr/n_total

def _unobs_frac_multi_label_preds(p_values, sign):
    """**Classification** - Calculate the fraction of multi-label predictions
    
    Calculates the fraction of multi-label predictions in an un-observed fashion - 
    i.e. disregarding the true labels

    Parameters
    ----------
    p_values : array, 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..

    sign : float in [0,1]
        Significance the metric should be calculated for
    
    Returns
    ------- 
    float

    See Also
    --------
    frac_multi_label_preds

    """
    p_values = to_numpy2D(p_values,'p_values')
    validate_sign(sign)
    
    predictions = p_values > sign
    return np.mean(np.sum(predictions, axis=1) > 1)

def frac_multi_label_preds(y_true, p_values, sign):
    """**Classification** - Calculate the fraction of multi-label predictions
    
    It is possible to both calculate this as an observed and un-observed metric,
    if the `y_true` is given the function returns three values - if no true values
    are known - only the fraction of multi-label predictions is returned. 
    
    Parameters
    ----------
    y_true : 1D numpy array, list, pandas Series, optional
        True labels or None. If given, the fraction of correct and incorrect 
        multi-label predictions can be calculated as well. Otherwise this will
        be calculated in an unobserved fashion
    
    p_values : 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..

    sign : float in [0,1]
        Significance the metric should be calculated for
    
    Returns
    ------- 
    frac_multi_label : float
        Fraction of multi-label predictions

    frac_correct : float or None
        Fraction of correct multi-label predictions (i.e. the true label is part of the set of predictions)
        Not returned if no `y_true` was given

    frac_incorrect : float or None
        Fraction of incorrect multi-label predictions. Not returned if no `y_true` was given
    """
    # If no y_true - calculate in an un-observed fashion
    if y_true is None:
        return _unobs_frac_multi_label_preds(p_values, sign), 
    
    validate_sign(sign)
    p_values = to_numpy2D(p_values,'p_values')
    y_true = to_numpy1D_int(y_true, 'y_true')
    check_consistent_length(y_true, p_values)
    
    n_total = len(y_true)

    predictions = p_values > sign
    m_label_filter = np.sum(predictions, axis=1) > 1
    m_preds = predictions[m_label_filter]
    m_trues = y_true[m_label_filter]

    n_corr = 0
    n_incorr = 0
    for i in range(0, m_trues.shape[0]):
        if m_preds[i, m_trues[i]]:
            n_corr +=1
        else:
            n_incorr +=1 
    
    return (n_corr+n_incorr)/n_total, n_corr/n_total, n_incorr/n_total

def obs_fuzziness(y_true, p_values):
    """**Classification** - Calculate the Observed Fuzziness (OF)
    
    Significance independent metric, smaller is better
    
    Parameters
    ----------
    y_true : 1D numpy array, list or pandas Series
        True labels

    p_values : 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..
    
    Returns
    -------
    obs_fuzz : float 
        Observed fuzziness
    """
    p_values = to_numpy2D(p_values,'p_values')
    y_true = to_numpy1D_int(y_true, 'y_true')
    check_consistent_length(y_true, p_values)

    of_sum = 0
    for i in range(0,p_values.shape[0]):
        # Mask the p-value of the true label
        p_vals_masked = np.ma.array(p_values[i,:], mask=False)
        p_vals_masked.mask[y_true[i]] = True
        # Sum the remaining p-values
        of_sum += p_vals_masked.sum()
    
    return of_sum / len(y_true)

def confusion_matrix(y_true, 
                            p_values, 
                            sign, 
                            labels=None, 
                            normalize_per_class = False):
    """**Classification** - Calculate a conformal confusion matrix
    
    A conformal confusion matrix includes the number of predictions for each class, empty predition sets and
    multi-prediction sets.

    Parameters
    ----------
    y_true : 1D numpy array, list or pandas Series
        True labels
    
    p_values : 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..
    
    sign : float in [0,1]
        Significance the confusion matrix should be calculated for
    
    labels : list of str, optional
        Descriptive labels for the classes
    
    normalize_per_class : bool, optional
        Normalizes the count so that each column sums to 1, good when visualizing imbalanced datasets (default False)
    
    Returns
    -------
    cm : pandas DataFrame
        The confusion matrix 
    """
    validate_sign(sign)
    p_values = to_numpy2D(p_values,'p_values')
    y_true = to_numpy1D_int(y_true, 'y_true')
    check_consistent_length(y_true, p_values)
    
    predictions = p_values > sign
    
    n_class = p_values.shape[1]
    
    # We create two different 'multi-label' predictions, either including or excluding the correct label
    if n_class == 2:
        result_matrix = np.zeros((n_class+2, n_class))
    else:
        result_matrix = np.zeros((n_class+3, n_class))
    
#    if labels is None:
    labels = get_str_labels(labels, get_n_classes(y_true,p_values))
#    if len(labels) != n_class:
#        raise ValueError('parameter labels must have the same length as the number of classes')
     #list(range(n_class))
#    elif len(labels) != n_class:
#        raise ValueError('parameter labels must have the same length as the number of classes')
    
    # For every observed class - t
    for t in range(n_class):
        
        # Get the predictions for this class
        t_filter = y_true == t
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
    
    row_labels = list(labels)
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
    
    return pd.DataFrame(result_matrix, columns=labels, index = row_labels)

########################################
### UNOBSERVED METRICS
########################################


def cp_credibility(p_values):
    """**Classification** - CP Credibility
    
    Mean of the largest p-values

    Parameters
    ----------
    p_values : array, 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..
    
    Returns
    ------- 
    credibility : float
    """
    p_values = to_numpy2D(p_values,'p_values')
    sorted_matrix = np.sort(p_values, axis=1)
    return np.mean(sorted_matrix[:,-1]) # last index is the largest

def cp_confidence(p_values):
    """**Classification** - CP Confidence 
    
    Mean of 1-'second largest p-value'

    Parameters
    ----------
    p_values : array, 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..
    
    Returns
    ------- 
    confidence : float
    """
    p_values = to_numpy2D(p_values,'p_values')
    sorted_matrix = np.sort(p_values, axis=1)
    return np.mean(1-sorted_matrix[:,-2])

def s_criterion(p_values):
    """**Classification** - S criterion
    
    Mean of the sum of all p-values

    Parameters
    ----------
    p_values : array, 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..
    
    Returns
    ------- 
    s_score : float
    """
    p_values = to_numpy2D(p_values,'p_values')
    return np.mean(np.sum(p_values, axis=1))

def n_criterion(p_values, sign=_default_significance):
    """**Classification** - N criterion
    
    "Number" criterion - the average number of predicted labels. Significance dependent metric

    Parameters
    ----------
    p_values : array, 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..
    
    Returns
    -------
    n_score : float
    """
    p_values = to_numpy2D(p_values,'p_values')
    validate_sign(sign)
    return np.mean(np.sum(p_values > sign, axis=1))

def u_criterion(p_values):
    """**Classification** - U criterion - "Unconfidence"

    Smaller values are preferable
    
    Parameters
    ----------
    p_values : array, 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..
    
    Returns
    ------- 
    u_score : float
    """
    p_values = to_numpy2D(p_values,'p_values')
    sorted_matrix = np.sort(p_values, axis=1)
    return np.mean(sorted_matrix[:,-2])

def f_criteria(p_values):
    """**Classification** - F criterion 
    
    Average fuzziness. Average of the sum of all p-values appart from the largest one.
    Smaller values are preferable.

    Parameters
    ----------
    p_values : array, 2D numpy array or DataFrame
        The predicted p-values, first column for the class 0, second for class 1, ..
    
    Returns
    ------- 
    f_score : float
    """
    p_values = to_numpy2D(p_values,'p_values')
    sorted_matrix = np.sort(p_values, axis=1)
    if sorted_matrix.shape[1] == 2:
        # Mean of only the smallest p-value
        return np.mean(sorted_matrix[:,0]) 
    else:
        # Here we must take the sum of the values appart from the first column
        return np.mean(np.sum(sorted_matrix[:,:-1], axis=1))
