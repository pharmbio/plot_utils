import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../src')
from pharmbio.cp import metrics,plotting

boston_preds = np.load('resources/boston_pred_out_3D_169.npy')
boston_labels = np.load('resources/boston_labels.npy')
# These are the ones nonconformist calculates
significance_lvls = np.arange(0.01,1,0.01)

class Test_calib_plot(unittest.TestCase):

    def test_boston(self):
        error_rates = metrics.frac_error_reg(boston_labels,boston_preds)
        fig = plotting.plot_calibration_curve_reg(error_rates,significance_lvls)
        plt.show()


class Test_pred_width(unittest.TestCase):

    def test_boston(self):
        pred_widths = metrics.pred_width(boston_preds)
        fig = plotting.plot_pred_widths(pred_widths,significance_lvls)
        plt.show()
    
if __name__ == '__main__':
    unittest.main()