""""CP Regression metrics

"""

import numpy as np
import pandas as pd
from ..utils import *

def pred_width(predictions, median = True):
    """**Regression** - Calculates the median or mean interval width of the Confidence Intervals

    Parameters
    ----------
    predictions : 2D or 3D numpy array
        The input can either be a 2D array from a single significance level with shape (n_samples,2), or a 3D array with predictions for several significance levels with shape (n_samples,2,n_significance_levels). In the second dimension of the array, the first index should contain the lower/min value and the second the upper/max value of the prediction interval
    
    median : bool
        True - if the median interval should be calculated
        False - if mean should be calculated

    Returns
    -------
    widths : float or 1D numpy array
        A scalar value if the `predictions` input is 2D, or a 1D array if the `predictions` is 3D (one median/mean value for each significance level)
    """
    n_sign, pred_matrix = validate_regression_preds(predictions) 

    if n_sign > 1:
        # 3D matrix
        widths = pred_matrix[:,1,:] - pred_matrix[:,0,:]
    else:
        # 2D matrix
        widths = pred_matrix[:,1] - pred_matrix[:,0]
    
    if (np.any(widths < 0)):
        raise ValueError('Invalid input, prediction intervals cannot be negative')
    
    return (np.median(widths, axis=0) if median else np.mean(widths, axis=0))
    
def frac_error_reg(y_true, predictions):
    """**Regression** - Calculate the fraction of errors

    Parameters
    ----------
    y_true : 1d array like
        List or array with the true labels, must be convertable to numpy ndarray
    
    predictions : 2D or 3D ndarray
        A matrix with either shape (n_samples, 2, n_sign_levels) or (n_sampes,2). The shape of the preidctions will decide the output dimension of the error_rates
    
    Returns
    -------
    error_rates : float or 1D ndarray
        The either a single float in case input is 2D, or an array of error rates (one for each significance level)

    """
    # Validation and potential 
    n_sign, pred_matrix = validate_regression_preds(predictions)
    ys = to_numpy1D_reg_y_true(y_true, pred_matrix.shape[0])

    if n_sign > 1:
        # 3D matrix
        ys.shape = (ys.shape[0],1) # turn to matrix in order to broadcast
        truth_vals = (np.greater_equal(ys,pred_matrix[:,0])) & (np.greater_equal(pred_matrix[:,1],ys))
    else:
        # 2D matrix
        truth_vals = (pred_matrix[:,0] <= ys) & (ys <= pred_matrix[:,1])
        # True = 1, False = 0, mean will be fraction of true values
    return 1 - truth_vals.mean(0)
