import numpy as np
import pytest
import matplotlib.pyplot as plt

from pharmbio.cp import metrics,plotting
from ....help_utils import _save_reg, get_resource
from ....context import resource_dir

boston_preds = np.load(get_resource('boston_pred_out_3D_169.npy'))
boston_preds_norm = np.load(get_resource('boston_pred_out_3D_169_normalized.npy'))
boston_labels = np.load(get_resource('boston_labels.npy'))
# These are the ones nonconformist calculates
significance_lvls = np.arange(0.01,1,0.01)

class Test_calib_plot():

    def test_boston(self):
        error_rates = metrics.frac_error_reg(boston_labels,boston_preds)
        fig = plotting.plot_calibration(significance_lvls,error_rates)
        _save_reg(fig,"Test_calib_plot.test_boston")
    
    def test_boston_flipping(self):
        error_rates = metrics.frac_error_reg(boston_labels,boston_preds)
        # Plot all in one image
        fig, axes = plt.subplots(2,2,figsize=(10,10))
        # std
        plotting.plot_calibration(significance_lvls,error_rates, ax=axes[0,0], title='std')
        # flip x
        plotting.plot_calibration(significance_lvls,error_rates, ax=axes[0,1], flip_x=True, title='flip x')
        # flip y
        plotting.plot_calibration(significance_lvls,error_rates, ax=axes[1,0], flip_y=True, title='flip y')
        # flip both
        plotting.plot_calibration(significance_lvls,error_rates,ax=axes[1,1], flip_x=True, flip_y=True, title='both')

        _save_reg(fig,"Test_calib_plot.test_boston_flippin")


class Test_pred_width():

    def test_boston(self):
        pred_widths = metrics.pred_width(boston_preds)
        fig = plotting.plot_pred_widths(significance_lvls,pred_widths)
        _save_reg(fig,"Test_pred_width.test_boston")
    
    def test_boston_subset(self):
        pred_widths = metrics.pred_width(boston_preds)
        fig = plotting.plot_pred_widths(significance_lvls[10:],pred_widths[10:])
        ax = fig.axes[0]
        _save_reg(fig,"Test_pred_width.test_boston_subset")
    
    def test_boston_non_std(self):
        pred_widths = metrics.pred_width(boston_preds)
        fig, axes = plt.subplots(1,2,figsize=(10,10))
        # No flip
        plotting.plot_pred_widths(significance_lvls,pred_widths,ax=axes[0],title="standard")
        # Flip
        plotting.plot_pred_widths(significance_lvls,pred_widths,ax=axes[1],flip_x=True,title="flip_x")
        # Save
        _save_reg(fig,"Test_pred_width.test_boston_flippin'")
    

class Test_pred_intervals():

    def test_boston(self):
        fig = plotting.plot_pred_intervals(boston_labels,
            boston_preds[:,:,70]
            # ,incorrect_ci='red'
            # , line_cap = 2
            )
        fig.get_axes()[0].legend(loc='upper left')
        _save_reg(fig,"Test_pred_intervals.test_boston")
    
    def test_boston_norm(self):
        fig = plotting.plot_pred_intervals(boston_labels,
            boston_preds_norm[:,:,70]
            # ,incorrect_ci='red'
            , line_cap = True
            , incorrect_ci= 'k'
            )
        fig.get_axes()[0].legend(loc='upper left')
        _save_reg(fig,"Test_pred_width:test_boston_norm")
    
    def test_boston_gray(self):
        fig = plotting.plot_pred_intervals(boston_labels,
            boston_preds_norm[:,:,70],
            correct_color='gray',
            correct_marker='o',
            incorrect_color='k',
            incorrect_marker='X'
            # ,incorrect_ci='red'
            , line_cap = True
            , incorrect_ci= 'k'
            )
        fig.get_axes()[0].legend(loc='upper left')
        _save_reg(fig,"Test_pred_width:test_boston_gray")

