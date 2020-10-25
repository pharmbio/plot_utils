import numpy as np
import pandas as pd


def _as_numpy2D(input, param_name, min_num_cols=2):
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

def _as_numpy1D_int(input, param_name):
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