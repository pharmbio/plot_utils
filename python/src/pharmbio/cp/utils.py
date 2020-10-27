import numpy as np
import pandas as pd

def get_sign_vals(sign_vals, sign_min=0,sign_max=1,sign_step=0.01):
    '''Generate a list of significance values
    
    Returns
    -------
    list of float
    '''
    
    # prefer an explict list of values
    if sign_vals is not None:
        if not isinstance(sign_vals, list):
            raise TypeError('parameter sign_vals must be a list of floats')
        if len(sign_vals) < 2:
            raise ValueError('parameter sign_vals must be a list with more than one value')
        # Validate the given significance values
        sign_vals = sorted(sign_vals)
        for sign in sign_vals:
            if sign > 1 or sign < 0:
                raise ValueError('Significance value must be in the range [0,1]')
    else:
        # Do some validation
        if sign_min<0:
            raise ValueError('sign_min must be >= 0')
        if sign_max > 1:
            raise ValueError('sign_min must be <= 1')
        if sign_max < sign_min:
            raise ValueError('sign_max < sign_min not allowed')
        if sign_step < 1e-4 or sign_step > .5:
            raise ValueError('sign_step must be in the range [1e-4, 0.5]')
        sign_vals = list(np.arange(sign_min,sign_max,sign_step))
        if sign_vals[-1] < sign_max:
            sign_vals.append(sign_max)
    return sign_vals

def validate_sign(sign):
    """Validate that `sign` is within [0,1] or raise error

    Checks both the type and the range of the input `sign`

    Parameters
    ----------
    sign : int or float
        The significance level to check
    """
    if not isinstance(sign, (int,float)):
        raise TypeError('parameter sign must be a number')
    if sign < 0 or sign >1:
        raise ValueError('parameter sign must be in the range [0,1]')

def to_numpy2D(input, param_name, min_num_cols=2):
    if input is None:
        raise ValueError('Input {} cannot be None'.format(param_name))
    elif isinstance(input, list):
        # This should be a python list-matrix, convert to numpy matrix
        matrix = np.array(input)
    elif isinstance(input, pd.DataFrame):
        matrix = input.to_numpy()
    elif isinstance(input, np.ndarray):
        matrix = input
    else:
        raise ValueError('parameter {} in unsupported format: {}'.format(param_name,type(input)))
    # Validate at least min num columns present
    if len(matrix.shape) < min_num_cols or matrix.shape[1] < min_num_cols:
        raise ValueError('parameter {} must be a matrix with at least {} columns'.format(param_name, min_num_cols))
    return matrix

def to_numpy1D_int(input, param_name):
    if isinstance(input, (list, pd.Series)):
        arr = np.array(input)
    elif isinstance(input, np.ndarray):
        if len(input.shape) == 1:
            arr = input
        elif input.shape[1]>1:
            raise ValueError('parameter {} must be a list, 1D numpy array or pandas Series'.format(param_name))
        else:
            input.shape = (len(input), )
            arr = input
    else:
        raise ValueError('parameter {} must be a list, 1D numpy array or pandas Series'.format(param_name))

    return arr.astype(np.int16)