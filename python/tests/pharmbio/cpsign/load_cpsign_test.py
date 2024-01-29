import numpy as np

import pytest

from pharmbio.cp import plotting
from pharmbio import cpsign
from statistics import mean 
from ...help_utils import _save_clf, _save_reg, get_resource

# file names
clf_stats_file = 'cpsign_clf_stats.csv'
clf_predictions_file = 'cpsign_clf_predictions.csv'
reg_stats_file = 'cpsign_reg_stats.csv'
reg_stats_2_file = 'cpsign_reg_stats_2.csv'




class TestClassification():

    # update plot-settings
    plotting.update_plot_settings()
    
    @pytest.fixture(scope="class")
    def load_data(self):
        (self.signs, self.errs, self.errs_sd, self.labels) = cpsign.load_calib_stats(get_resource(clf_stats_file), sep='\t')
      
        
    @pytest.fixture
    def test_load_stats_file(self,load_data):

        # plot and save it
        fig = plotting.plot_calibration(sign_vals=self.signs, error_rates=self.errs, error_rates_sd=self.errs_sd, labels=self.labels)
        fig.axes[0].set_title('from calculated values CPSign')
        _save_clf(fig,"CPSign_computed_clf.test_load_stats_file")

        # No labels and no error-SD
        fig_no_label = plotting.plot_calibration(sign_vals=self.signs, error_rates=self.errs)
        fig_no_label.axes[0].set_title('from CPSign - no labels')
        _save_clf(fig_no_label,"CPSign_computed_clf.test_load_stats_file_2")

        # Flip axes
        fig_flipped = plotting.plot_calibration(sign_vals=self.signs, error_rates=self.errs, 
                                                error_rates_sd=self.errs_sd, labels=self.labels, 
                                                flip_x=True,flip_y=True,title='precomputed from CPSign - flipped')
        _save_clf(fig_flipped,"CPSign_computed_clf.test_load_stats_file_3")
    
    @pytest.fixture
    def test_load_stats_file_conf_acc(self, load_data):
        # convert to confidence and accuracy instead 
        confs = 1- self.signs
        accs = 1 - self.errs
        accs_sd = self.errs_sd
        # plot and save it
        fig = plotting.plot_calibration(conf_vals=confs,accuracy_vals=accs, accuracy_sd=accs_sd, labels=self.labels)
        fig.axes[0].set_title('confs + accs from CPSign')
        _save_clf(fig,"CPSign_computed_clf.test_load_stats_file_conf_acc")

        fig_no_label = plotting.plot_calibration(conf_vals=confs,accuracy_vals=accs, accuracy_sd=accs_sd, sd_alpha=.1)
        fig_no_label.axes[0].set_title('from CPSign - no labels')
        _save_clf(fig_no_label,"CPSign_computed_clf.test_load_stats_file_conf_acc_2")
    
    @pytest.fixture
    def test_plot_single_calib_line(self, load_data):
        assert self.labels[0].lower() == 'overall'
        fig = plotting.plot_calibration(sign_vals=self.signs,error_rates=self.errs[:,0], error_rates_sd=self.errs_sd[:,0], labels=self.labels[0], title='cpsign only overall calib')
        _save_clf(fig,"TestCLF_CPSign.test_plot_single_calib_line")

    @pytest.fixture
    def test_load_stats_label_eff(self, load_data):
        (signs,single,multi,empty, _,_,_) = cpsign.load_clf_efficiency_stats(get_resource(clf_stats_file), sep='\t')
        # Explicitly turn of reading of SD values
        (signs2,single2,multi2,empty2) = cpsign.load_clf_efficiency_stats(get_resource(clf_stats_file), sep='\t', 
                                                                      prop_e_sd_regex=None, prop_m_sd_regex=None, prop_s_sd_regex=None)
        # Output should be identical for both function calls
        assert np.array_equal(signs, signs2)
        assert np.array_equal(single, single2)
        assert np.array_equal(multi, multi2)
        assert np.array_equal(empty, empty2)
        fig = plotting.plot_label_distribution(prop_single=single, sign_vals=signs,prop_multi=multi, prop_empty=empty)
        _save_clf(fig, "TestCLF_CPSign.label_distr")

    def test_load_preds(self):
        (ys, pvals, labels) = cpsign.load_clf_predictions(get_resource(clf_predictions_file),'target',';')
        fig = plotting.plot_label_distribution(y_true=ys,p_values= pvals)
        _save_clf(fig, "TestCLF_CPSign.load_clf_pred")

class TestRegression():
    
    def assert_label_output(self,labels):
        assert len(labels) == 1
        assert labels[0].lower() == 'overall'

    def test_load_reg_calib(self):
        (signs,errs,errs_sd,labels) = cpsign.load_calib_stats(get_resource(reg_stats_file), sep='\t')
        self.assert_label_output(labels) 
        fig = plotting.plot_calibration(sign_vals=signs,error_rates=errs, error_rates_sd=errs_sd, labels='Error rate', title='cpsign only overall calib')
        _save_reg(fig,"TestREG_CPSign.test_plot_calib")
    
    def test_load_reg_calib_2(self):
        (signs,errs,errs_sd,labels) = cpsign.load_calib_stats(get_resource(reg_stats_2_file), sep='\t')
        self.assert_label_output(labels) 
        print("cpsign-reg2: ",signs,errs,errs_sd,labels)
        fig = plotting.plot_calibration(sign_vals=signs,error_rates=errs, error_rates_sd=errs_sd, labels='Error rate', title='cpsign only overall calib')
        _save_reg(fig,"TestREG_CPSign.test_plot_calib_2")

    def test_load_reg_eff(self):
        (sign_vals, median_widths, mean_widths, median_widths_sd, mean_widths_sd) = cpsign.load_reg_efficiency_stats(get_resource(reg_stats_file), sep='\t')
        fig = plotting.plot_pred_widths(sign_vals,median_widths)
        _save_reg(fig, "TestREG_CPSign.test_plot_widths")
        # With std
        fig_std = plotting.plot_pred_widths(sign_vals,median_widths, median_widths_sd)
        _save_reg(fig_std, "TestREG_CPSign.test_plot_widths_std")


# if __name__ == '__main__':
#     unittest.main()