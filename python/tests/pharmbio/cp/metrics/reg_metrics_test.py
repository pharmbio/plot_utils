import numpy as np
import pandas as pd
import unittest

from ....help_utils import get_resource
from pharmbio.cp.metrics import *
from statistics import mean 

class Test_pred_width(unittest.TestCase):

    def test_single_pred(self):
        pred = np.array([
            [0,1]
        ])
        m = pred_width(pred)
        self.assertEqual(1,m)

        pred_2 = np.array([
            [-1.5,5.2]
        ])
        self.assertAlmostEqual(1.5+5.2, pred_width(pred_2))

    def test_two_pred(self):
        pred = np.array([
            [0.5,1],
            [3.5,9]
        ])
        # median
        self.assertAlmostEqual(np.median([.5, 9-3.5]), pred_width(pred))
        # mean
        self.assertAlmostEqual(np.mean([.5, 9-3.5]), pred_width(pred,median=False))

    def test_3d_pred(self):
        # Create a (3,2,2) matrix (for two different significance levels)
        
        # First "significance" level
        pred_sig1 = np.array([
            [0.5,1],
            [3.5,9],
            [5.1,8.7]])
        
        # Second "significance" level
        pred_sig2= np.array([
            [0.25,1.5],
            [3,9.5],
            [5,9]])
        
        # Stack them to 3D
        pred_3d = np.stack((pred_sig1,pred_sig2), axis=-1)
        self.assertEqual((3,2,2), pred_3d.shape)
        # first level:  np.mean([.5,9-3.5,8.7-5.1]) = 3.2, median = 3.6
        # second level: np.mean([1.5-.25,9.5-3,9-5]) = 3.9166667, median = 4
        median = pred_width(pred_3d, median=True)
        mean = pred_width(pred_3d, median=False)
        self.assertTrue(equal_np_arrays([3.2, 3.9166667], mean))
        self.assertTrue(equal_np_arrays([3.6, 4], median))
    
    def test_3d_boston(self):
        boston_preds = np.load(get_resource('boston_pred_out_3D_169.npy'))
        self.assertEqual((169,2,99),boston_preds.shape)
        withs = pred_width(boston_preds)
        #print(withs)
        self.assertEqual(len(withs),99)

class Test_frac_error_reg(unittest.TestCase):

    def test_boston(self):
        boston_preds = np.load(get_resource('boston_pred_out_3D_169.npy'))
        boston_labels = np.load(get_resource('boston_labels.npy'))
        # Try 3D
        errs3d = frac_error_reg(boston_labels,boston_preds)
        #print(errs3d)
        # As 2D (a single significance level)
        for i in range(boston_preds.shape[2]):
            errs2d = frac_error_reg(boston_labels,boston_preds[:,:,i])
            #print(errs2d)
            self.assertAlmostEqual(errs2d,errs3d[i])
            


def equal_np_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return False
    return True if np.all((arr1 - arr2) < 0.000001) else False

if __name__ == '__main__':
    unittest.main()