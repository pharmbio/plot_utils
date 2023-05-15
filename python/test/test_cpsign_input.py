import numpy as np
import pandas as pd
import unittest

import sys
sys.path.append('../src')
from pharmbio.cp import plotting
from pharmbio.cpsign import *
from statistics import mean 
import time
from test_utils import _save_clf, _save_reg

class TestClassification(unittest.TestCase):

    # update plot-settings
    plotting.update_plot_settings()
    
    def test_load_stats_file(self):
        result = load_calib_stats('resources/cpsign_clf_stats.csv', sep='\t')
        
        self.assertEqual(4,len(result))
        # plot and save it
        fig = plotting.plot_calibration(sign_vals=result[0], error_rates=result[1], error_rates_sd=result[2], labels=result[3])
        fig.axes[0].set_title('from precomputed values CPSign')
        _save_clf(fig,"CPSign_computed_clf.test_load_stats_file")

        # No labels and no error-SD
        fig_no_label = plotting.plot_calibration(sign_vals=result[0], error_rates=result[1])
        fig_no_label.axes[0].set_title('from CPSign - no labels')
        _save_clf(fig_no_label,"CPSign_computed_clf.test_load_stats_file_2")

        # Flip axes
        fig_flipped = plotting.plot_calibration(sign_vals=result[0], error_rates=result[1], error_rates_sd=result[2], labels=result[3], flip_x=True,flip_y=True,title='precomputed from CPSign - flipped')
        _save_clf(fig_flipped,"CPSign_computed_clf.test_load_stats_file_3")
    
    def test_load_stats_file_conf_acc(self):
        result = load_calib_stats('resources/cpsign_clf_stats.csv', sep='\t')
        (signs, errs, errs_sd, labels) = result
        confs = 1- signs
        accs = 1 - errs
        accs_sd = errs_sd
        self.assertEqual(4,len(result))
        # plot and save it
        fig = plotting.plot_calibration(conf_vals=confs,accuracy_vals=accs, accuracy_sd=accs_sd, labels=result[3])
        fig.axes[0].set_title('confs + accs from CPSign')
        _save_clf(fig,"CPSign_computed_clf.test_load_stats_file_conf_acc")

        fig_no_label = plotting.plot_calibration(conf_vals=confs,accuracy_vals=accs, accuracy_sd=accs_sd, sd_alpha=.1)
        fig_no_label.axes[0].set_title('from CPSign - no labels')
        _save_clf(fig_no_label,"CPSign_computed_clf.test_load_stats_file_conf_acc_2")
    
    def test_plot_single_calib_line(self):
        (signs,errs,errs_sd,labels) = load_calib_stats('resources/cpsign_clf_stats.csv', sep='\t')
        fig = plotting.plot_calibration(sign_vals=signs,error_rates=errs[:,0], error_rates_sd=errs_sd[:,0], labels=labels[0], title='cpsign only overall calib')
        _save_clf(fig,"TestCLF_CPSign.test_plot_single_calib_line")

    def test_load_stats_label_eff(self):
        (signs,single,multi,empty, _,_,_) = load_clf_efficiency_stats('resources/cpsign_clf_stats.csv', sep='\t')
        # print(empty)
        # Explicitly turn of reading of SD values
        (signs,single,multi,empty) = load_clf_efficiency_stats('resources/cpsign_clf_stats.csv', sep='\t', prop_e_sd_regex=None, prop_m_sd_regex=None, prop_s_sd_regex=None)
        # print(empty)
        fig = plotting.plot_label_distribution(prop_single=single, sign_vals=signs,prop_multi=multi, prop_empty=empty)
        _save_clf(fig, "TestCLF_CPSign.label_distr")

    def test_load_preds(self):
        (ys, pvals, labels) = load_clf_predictions('resources/cpsign_clf_predictions.csv','target',';')
        fig = plotting.plot_label_distribution(y_true=ys,p_values= pvals)
        _save_clf(fig, "TestCLF_CPSign.load_clf_pred")

class TestRegression(unittest.TestCase):

    def test_load_reg_calib(self):
        (signs,errs,errs_sd,labels) = load_calib_stats('resources/cpsign_reg_stats.csv', sep='\t')
        fig = plotting.plot_calibration(sign_vals=signs,error_rates=errs, error_rates_sd=errs_sd, labels='Error rate', title='cpsign only overall calib')
        _save_reg(fig,"TestREG_CPSign.test_plot_calib")

    def test_load_reg_eff(self):
        (sign_vals, median_widths, mean_widths, median_widths_sd, mean_widths_sd) = load_reg_efficiency_stats('resources/cpsign_reg_stats.csv', sep='\t')
        fig = plotting.plot_pred_widths(sign_vals,median_widths)
        _save_reg(fig, "TestREG_CPSign.test_plot_widths")
        # With std
        fig_std = plotting.plot_pred_widths(sign_vals,median_widths, median_widths_sd)
        _save_reg(fig_std, "TestREG_CPSign.test_plot_widths_std")


if __name__ == '__main__':
    unittest.main()