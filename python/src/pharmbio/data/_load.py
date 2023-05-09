"""Utility functions for loading and converting datasets
"""

import numpy as np
import pandas as pd
import re

def load_regression(f,
    y_true_col,
    sep = ',',
    lower_regex=r'^prediction.*interval.*lower.*\d+',
    upper_regex=r'^prediction.*interval.*upper.*\d+',
    specifies_significance=None):
    """Loads a CSV file with predictions and converts to the format used by Plot_utils

    The required format is that the csv has;
    - A header
    - Specifies significance or confidence in the header names of the 'lower' and 'upper' columns
    - Those headers must only contain a single number

    Note that there is no requirement for a true label to exist, the `y_true_col` can be set to None and no y-labels will be returned

    Parameters
    ----------
    f : str or buffer
        File path or buffer that `Pandas.read_csv` can read
    
    y_true_col : str or None
        The (case insensitive) column header of the true labels, or None if it should not be loaded
    
    sep : str, default ','
        Delimiter that is used in the CSV between columns
    
    lower_regex, upper_regex : str or re.Pattern
        Regex used for finding columns of the lower and upper interval limits. Must match the column headers
    
    specifies_significance : bool or None, default None
        If the numbers in the headers are significance level (True) or confidence (False). If None, the first column-header found by `lower_regex` will be used to check for occurrences of 'significance' or 'conf' to try to infer what is used

    Returns
    -------
    (y, pred_matrix, sign_values)
    """

    if not isinstance(lower_regex, re.Pattern):
        lower_regex = re.compile(lower_regex,re.IGNORECASE)
    if not isinstance(upper_regex,re.Pattern):
        upper_regex = re.compile(upper_regex,re.IGNORECASE)
    num_pattern = re.compile('\d*\.\d*')
    y_col_lc = None if y_true_col is None else y_true_col.lower()
    y_true_ind = None
    
    df = pd.read_csv(f,sep=sep)
    low_ind, upp_ind, sign_low, sign_upp = [], [], [], []
    for i, c in enumerate(df.columns):
        if lower_regex.match(c) is not None:
            low_ind.append(i)
            sign_low.append(float(num_pattern.findall(c)[0]))
        elif upper_regex.match(c) is not None:
            upp_ind.append(i)
            sign_upp.append(float(num_pattern.findall(c)[0]))
        elif y_col_lc is not None and c.lower() == y_col_lc:
            y_true_ind = i

    # Some validation
    assert sign_low == sign_upp
    assert len(low_ind) == len(upp_ind)
    if not isinstance(specifies_significance,bool):
        col_lc = df.columns[low_ind[0]].lower()
        contains_sign =col_lc.__contains__('significance')
        contains_conf = col_lc.__contains__('confidence')

        if (contains_sign and contains_conf) or (not contains_sign and not contains_conf):
            raise ValueError('Parameter \'specifies_significance\' not set, could not deduce if significance or confidence is used. Explicitly set this parameter and try again')
        
        specifies_significance = True if contains_sign else False

    sign_vals = np.array(sign_low) if specifies_significance else 1 - np.array(sign_low)
    
    y, p = convert_regression(df,y_true_ind,low_ind,upp_ind)
    return y, p, sign_vals


def convert_regression(data,
    y_true_index,
    min_index,
    max_index):
    """
    Converts a 2D input matrix to a 3D ndarray that
    is required by the metrics and plotting functions
    
    Parameters
    ----------
    data : 2d array like
        Data matrix that must be convertible to 2D ndarray
    
    y_true_index : int or None
        Column index that the ground truth values are, or None if no 
        y values should be generated. Output `y` will then be None
    
    min_index, max_index : list or array of int
        Column indices for min and max values for prediction intervals
    
    Returns
    -------
    (y, predictions)
        y : 1D ndarray
            The y_true values or None if `y_true_index` is None
        
        predictions : 3D ndarray
            matrix of shape (n_examples, 2, n_significance_levels), where the second
            dimension is [min, max] of the prediction intervals
    """
    if not isinstance(data,np.ndarray):
        data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError('Input must be a 2D array type')
    
    ys = data[:,y_true_index].astype(np.float64) if y_true_index is not None else None

    if len(min_index) != len(max_index):
        raise ValueError('min_index and max_index must be of same length')
    
    # Allocate matrix
    preds = np.zeros((len(data),2,len(min_index)),dtype=np.float64)

    for s, (min,max) in enumerate(zip(min_index,max_index)):
        preds[:,0,s] = data[:,min]
        preds[:,1,s] = data[:,max]

    # Return tuple
    return (ys, preds)