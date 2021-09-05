from numpy.lib.arraysetops import isin
from sklearn.preprocessing import OneHotEncoder
# from os import sendfile
import numpy as np
import pandas as pd

# def get_sign_vals(sign_vals, sign_min=0,sign_max=1,sign_step=0.01):
#     """Generate a numpy array of significance values
    
#     Returns
#     -------
#     list of float
#     """
#     # prefer an explict list of values
#     if sign_vals is not None:
#         if not isinstance(sign_vals, list):
#             raise TypeError('parameter sign_vals must be a list of floats')
#         if len(sign_vals) < 2:
#             raise ValueError('parameter sign_vals must be a list with more than one value')
#         # Validate the given significance values
#         sign_vals = sorted(sign_vals)
#         for sign in sign_vals:
#             if sign > 1 or sign < 0:
#                 raise ValueError('Significance value must be in the range [0,1]')
#     else:
#         # Do some validation
#         if sign_min<0:
#             raise ValueError('sign_min must be >= 0')
#         if sign_max > 1:
#             raise ValueError('sign_min must be <= 1')
#         if sign_max < sign_min:
#             raise ValueError('sign_max < sign_min not allowed')
#         if sign_step < 1e-4 or sign_step > .5:
#             raise ValueError('sign_step must be in the range [1e-4, 0.5]')
#         sign_vals = list(np.arange(sign_min,sign_max,sign_step))
#         if sign_vals[-1] < sign_max:
#             sign_vals.append(sign_max)
#     return np.array(sign_vals)

def get_n_classes(y_true, p_vals):
    """Helper method for finding the maximum number of classes

    The number could either be the number of columns in the p-value matrix. 
    Or the user could have only sent a slice of the p-values/added more labels
    in the `y_true` due to wanting to plot them in a different color. The value 
    of `n_class` is the maxium number of these, so trying to access the `n_class'th - 1`
    column the p-value matrix might be out of range!
    
    """
    if y_true is None and p_vals is None:
        raise ValueError('Neither y_true nor p_values were given')
    elif y_true is None:
        if isinstance(p_vals, np.ndarray):
            return p_vals.shape[1]
        return to_numpy2D(p_vals, None).shape[1]
    elif p_vals is None:
        if isinstance(y_true, (list,np.ndarray,pd.Series)):
            return int(np.max(y_true))
    return max(int(np.max(y_true)+1), p_vals.shape[1]) # +1 on max(y_true) as labels start at 0

def get_str_labels(labels, n_class):
    """Helper method for turning numerical labels to str labels

    Parameters
    ----------
    labels : list of str or None
        Labels given as parameter, or None if not given

    n_class : int
        The number of classes

    """
    if labels is not None:
        if not isinstance(labels, (np.ndarray, list, pd.Series)):
            raise TypeError('parameter labels must be either a list or 1D numpy array')
        if len(labels) < n_class:
            raise TypeError('parameter labels and number of classes does not match')
        return np.array(labels).astype(str)
    else:
        # No labels, generate n_classes labels
        return ['Label {}'.format(i) for i in range(0,n_class)]

def validate_sign(sign):
    """Validate that `sign` is within [0,1] or raise error

    Checks both the type and the range of the input `sign`

    Parameters
    ----------
    sign : int or float
        The significance level to check
    """
    if isinstance(sign, np.ndarray):
        if not np.any((0<=sign) | (sign <=0)):
            raise ValueError('All significance levels must be in the range [0..1]')
    elif isinstance(sign, list):
        for s in sign:
            if s < 0 or s >1:
                raise ValueError('All significance levels must be in the range [0..1], got: {}'.format(s))
    elif not isinstance(sign, (int,float)):
        raise TypeError('parameter sign must be a number')
    elif sign < 0 or sign >1:
        raise ValueError('parameter sign must be in the range [0,1]')

def to_numpy2D(input, param_name, min_num_cols=2, return_copy=True):
    """ Converts python list-based matrices and Pandas DataFrames into numpy 2D arrays

    If input is already a numpy array, it will be copied in case `return_copy` is True
    """

    if input is None:
        raise ValueError('Input {} cannot be None'.format(param_name))
    elif isinstance(input, list):
        # This should be a python list-matrix, convert to numpy matrix
        matrix = np.array(input)
    elif isinstance(input, pd.DataFrame):
        matrix = input.to_numpy()
    elif isinstance(input, np.ndarray):
        if input.ndim != 2:
            raise ValueError('parameter {} must be a 2D matrix, was a ndarray of shape {}'.format(param_name,input.shape))
        matrix = input.copy() if return_copy else input 
    else:
        raise ValueError('parameter {} in unsupported format: {}'.format(param_name,type(input)))
    # Validate at least min num columns present
    if len(matrix.shape) < 2 or matrix.shape[1] < min_num_cols:
        raise ValueError('parameter {} must be a matrix with at least {} columns'.format(param_name, min_num_cols))
    return matrix

def to_numpy1D(input,param_name,return_copy=True):
    """Convert lists and Panda Series to 1D numpy array. 
    
    If input is already a numpy array, it is copied if `return_copy` is True
    """
    if isinstance(input, (list, pd.Series)):
        return np.array(input)
    elif isinstance(input, np.ndarray):
        if len(input.shape) == 1:
            return input.copy() if return_copy else input
        elif input.shape[1]>1:
            raise ValueError('parameter {} must be a list, 1D numpy array or pandas Series'.format(param_name))
        else:
            if return_copy:
                cpy = input.copy()
                cpy.shape = (len(cpy), )
                return cpy
            else:
                input.shape = (len(input),)
                return input
    else:
        raise ValueError('parameter {} must be a list, 1D numpy array or pandas Series'.format(param_name))

def to_numpy1D_int(input, param_name):
    return to_numpy1D(input,param_name).astype(np.int16)

def to_numpy1D_onehot(input, param_name, return_encoder=False, dtype=bool, labels=None):
    """
    Returns
    -------
    (matrix, array):
        matrix : numpy 2D of bool
            The one-hot-encoded version of y_true
        array : numpy 1D
            The categories, corresponding to the indicies of `matrix`
    
    (matrix, array, sklearn.preprocessing.OneHotEncoder)
        When `return_encoder` is set to True.
    """
    one_dim = to_numpy1D(input,param_name,return_copy=False).reshape(-1,1)
    if labels is None:
        labels = np.unique(one_dim)
    enc = OneHotEncoder(sparse=False,dtype=dtype,categories=[labels])
    one_hot = enc.fit_transform(one_dim)

    if return_encoder:
        return one_hot, enc.categories_[0], enc
    else:
        return one_hot, enc.categories_[0]


def validate_regression_preds(input_matrix):
    """Checks if input is either 2D (one significance level) or 3D (multiple significance levels)

    The second dimension must always be 2 [lower, upper] of the prediction interval

    Returns
    -------
    (n_significance_lvls, 2D/3D ndarray)
    """
    if not isinstance(input_matrix, np.ndarray):
        raise ValueError('Regression predictions only supports numpy 2D or 3D arrays')
    
    if input_matrix.ndim == 2:
        # 2D matrix should be (N,2) shape
        if input_matrix.shape[1] != 2:
            raise ValueError('Regression predictions should be of the shape (N,2) or (N,2,S), where N is the number of predictions and S is the number of significance levels')
        return 1, input_matrix
    elif input_matrix.ndim == 3:
        # 3D matrix should be (N,2,S) shape
        if (input_matrix.shape[1]!= 2):
            raise ValueError('Regression predictions should be of the shape (N,2) or (N,2,S), where N is the number of predictions and S is the number of significance levels')
        if input_matrix.shape[2]==1:
            # "Fake 3D matrix"
            return 1, input_matrix[:,:,0]
        return input_matrix.shape[2], input_matrix
    else:
        raise ValueError('Regression predictions only supported as numpy 2D or 3D arrays')

def to_numpy1D_reg_y_true(y_true,expected_len):
    arr = to_numpy1D(y_true,'y_true')
    if len(arr)!=expected_len:
        raise ValueError('Input predictions and true labels not of the same length: {} != {}'.format(len(arr),expected_len))
    return arr