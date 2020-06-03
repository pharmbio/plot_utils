import numpy as np
import pandas as pd
from collections import Counter

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
        raise ValueError('arguments true_labels and p_values must have the same length')
    
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
        raise ValueError('arguments true_labels and p_values must have the same length')
    
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
        raise ValueError('arguments true_labels and p_values must have the same length')
    
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
        raise ValueError('arguments true_labels and p_values must have the same length')
    
    of_sum = 0
    for i in range(0,p_values.shape[0]):
        # Mask the p-value of the true label
        p_vals_masked = np.ma.array(p_values[i,:], mask=False)
        p_vals_masked.mask[true_labels[i]] = True
        # Sum the remaining p-values
        of_sum += p_vals_masked.sum()
    
    return of_sum / len(true_labels)
    
def calc_confusion_matrix_old(true_labels, p_vals, sign, class_labels=['A','N']):
    ''' Calculates conformal confusion matrix with number of predictions for each class and number of both and none. 
    Only supports 2 classes at the moment.

    Arguments:
    true_labels -- A list or 1D numpy array, with values 0, 1, etc for each class
    p_vals -- A 2D numpy array with first column p-value for the 0-class, second column p-value for second class
    sign -- A value between 0 and 1 corresponding to the significance level
    class_labels -- An array with the class names

    returns -- A Pandas dataframe with conformal confusion matrix
    '''
    if len(class_labels) != 2 or p_vals.shape[1] != 2 :
        raise ValueError('Only two classes are supported at this point in time')

    predictions1 = [ class_labels[0] if p0>sign and p1<sign and t==class_labels[0] else
                            class_labels[1] if p1>sign and p0<sign and t==class_labels[0] else 
                            'none'          if p1<sign and p1<sign and t==class_labels[0] else
                            'both'          for p0,p1,t in zip(p_vals[:,0], p_vals[:,1], true_labels) if t==class_labels[0] ]
    predictions2 = [ class_labels[0] if p0>sign and p1<sign and t==class_labels[1] else
                            class_labels[1] if p1>sign and p0<sign and t==class_labels[1] else 
                            'none'          if p1<sign and p1<sign and t==class_labels[1] else
                            'both'          for p0,p1,t in zip(p_vals[:,0], p_vals[:,1], true_labels) if t==class_labels[1] ]
    possibleOutcomes = [class_labels[0], class_labels[1], "both", "none"]
    dict1 = Counter(predictions1)
    dict2 = Counter(predictions2)
    for outcome in possibleOutcomes :
        if not outcome in dict1:
            dict1[outcome] = 0
        if not outcome in dict2:
            dict2[outcome] = 0
    df1 = pd.DataFrame.from_dict(dict1, orient='index')
    df1.columns=[class_labels[0]]
    df2 = pd.DataFrame.from_dict(dict2, orient='index')
    df2.columns=[class_labels[1]]
    df = pd.concat([df1,df2], axis=1)
    return df

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



####################################
### REGRESSION - TODO
####################################

def calc_error_rate_regression(true_labels, prediction_ranges):
    # what is the best way to do this? prediction-ranges as 2D numpy array? [lower bound, upper bound] ?
    raise NotImplementedError('TODO')
