from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

__sklearn_1_2_0_or_later = False

try:
    from packaging.version import Version, parse as parse_version
    import sklearn
    __sklearn_1_2_0_or_later = parse_version(sklearn.__version__)>= Version('1.2.0')
except ImportError as e:
    pass 


def get_n_classes(y_true, p_vals):
    """Helper method for finding the maximum number of classes

    The number could either be the number of columns in the p-value matrix. 
    Or the user could have only sent a slice of the p-values/added more labels
    in the `y_true` due to wanting to plot them in a different color. The value 
    of `n_class` is the maximum number of these, so trying to access the `n_class'th - 1`
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

    Checks both the type and content are OK. If numpy.ndarray the array must be 1dim

    Parameters
    ----------
    sign : int, float, numpy.ndarray, pandas.Series
        The significance level to check
    """
    if isinstance(sign, np.ndarray) and sign.ndim == 0:
        # This is a single element, convert to float
        sign = float(sign)
    
    if isinstance(sign, (np.ndarray,pd.Series, list, tuple)):
        # Check that ndarray is 1dim
        if isinstance(sign, np.ndarray):
            # must be dim == 1  
            if sign.ndim != 1 :
                raise ValueError('Significance levels must be given as a single value or an array / 1dim ndarray')
        # validate each value
        for s in sign:
            if s < 0 or s >1:
                raise ValueError('All significance levels must be in the range [0..1], got: {}'.format(s))   
    elif isinstance(sign, (int,float)):
        # I.e. a single value
        if sign < 0 or sign >1:
            # Single value but which is outside 
            raise ValueError('parameter sign must be in the range [0,1]')
    else:
        raise TypeError('parameter sign must be a number or sequence of numbers')

def to_numpy2D(input, param_name, min_num_cols=2, return_copy=True, unravel=False):
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
            if input.ndim == 1 and unravel:
                # if we are allowed to unravel (i.e. add an additional dim to the ndarray) - we create one
                return input.reshape((len(input),1))
            raise ValueError('parameter {} must be a 2D matrix, was a ndarray of shape {}'.format(param_name,input.shape))
        matrix = input.copy() if return_copy else input 
    else:
        raise TypeError('parameter {} in unsupported format: {}'.format(param_name,type(input)))
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
            The categories, corresponding to the indices of `matrix`
    
    (matrix, array, sklearn.preprocessing.OneHotEncoder)
        When `return_encoder` is set to True.
    """
    one_dim = to_numpy1D(input,param_name,return_copy=False).reshape(-1,1)
    if labels is None:
        labels = np.unique(one_dim)
    
    if __sklearn_1_2_0_or_later:
        enc = OneHotEncoder(sparse_output=False,dtype=dtype,categories=[labels])
    else:
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