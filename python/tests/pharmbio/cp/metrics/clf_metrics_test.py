import numpy as np
import pandas as pd
import pytest


from pharmbio.cp.metrics import (confusion_matrix,frac_errors,frac_multi_label_preds,
                                 frac_single_label_preds,n_criterion,u_criterion,
                                 f_criteria,s_criterion,cp_credibility,cp_confidence,frac_error)
from statistics import mean 
import time
from ....help_utils import get_resource

class TestConfusionMatrix():

    def test_only_0_class(self):
        p_vals_m = np.array([
            [0.05, 0.85], 
            [0.23, 0.1], 
            [.1, .1], 
            [.21, .21], 
            [.3, 0.15]
            ])
        tr = np.array([0,0,0,0,0])
        expected_CM = np.array([[2, 0], [1,0], [1,0], [1,0]])
        
        cm = confusion_matrix(tr,p_vals_m, sign=0.2)
        assert np.array_equal(cm.to_numpy(), expected_CM)
        # test input as normal arrays instead
        cm2 = confusion_matrix(tr.tolist(), p_vals_m.tolist(), sign=0.2)
        assert cm.equals( cm2 )
        # assert cm == cm2 
        # Test with Pandas as input
        pvals_pd = pd.DataFrame(p_vals_m)
        tr_pd = pd.Series(tr)
        cm3 = confusion_matrix(tr_pd, pvals_pd, sign=0.2)
        assert cm.equals( cm3 )
        # assert cm == cm3
    
    def test_small_binary_ex(self):
        p_vals_m = np.array([[0.05, 0.85], [0.23, 0.1], [.1, .1], [.21, .21], [.3, 0.15]])
        tr = np.array([0,1,0,1,0])
        expected_CM = np.array([
            [1,1], 
            [1,0], 
            [1,0], 
            [0,1]
            ])

        cm = confusion_matrix(tr,p_vals_m, sign=0.2)
        assert np.array_equal(cm.to_numpy(), expected_CM)
    
    def test_with_custom_labels(self):
        p_vals_m = np.array([[0.05, 0.85], [0.23, 0.1], [.1, .1], [.21, .21], [.3, 0.15]])
        tr = np.array([0,1,0,1,0])
        custom_labels = ['Mutagen', 'Nonmutagen']
        expected_CM = np.array([[1, 1], [1,0], [1,0], [0,1]])

        cm = confusion_matrix(tr,p_vals_m, sign=0.2, labels=custom_labels)
        assert cm.shape == expected_CM.shape
        assert np.array_equal(cm.to_numpy(), expected_CM)
        #assert(cm.shape, expected_CM.shape)
        assert np.array_equal(cm.to_numpy(), expected_CM)

        cm_names = list(cm.columns)
        assert custom_labels == cm_names
        assert custom_labels == list(cm.index.values)[:2]
        assert 4 == len(list(cm.index.values))
        assert len(tr) == cm.to_numpy().sum()

    def test_3_class(self):
        p_vals_m = np.array(
            [
                [0.05, 0.1, 0.5],
                [0.05, 0.1, 0.5],
                [0.05, 0.3, 0.5],
                [0.05, 0.3, 0.1],
            ]
        )
        true_l = np.array([0, 1, 2, 0])
        custom_labels = [4, 5, 6]
        cm = confusion_matrix(true_l,p_vals_m, sign=0.2, labels=custom_labels)

        expected_CM = np.array([
            [0,0,0],
            [1,0,0],
            [1,1,0],
            [0,0,0],
            [0,0,1],
            [0,0,0]
        ])
        assert np.array_equal(cm.to_numpy(), expected_CM)
    
    ## Ebba TODO write test
    
    def test_normalize_3(self):
        p_vals_m = np.array(
            [
                [0.05, 0.1, 0.5],
                [0.05, 0.1, 0.5],
                [0.05, 0.3, 0.5],
                [0.05, 0.3, 0.1],
            ]
        )
        true_l = np.array([0, 1, 2, 0])
        custom_labels = [4, 5, 6]
        cm = confusion_matrix(true_l,p_vals_m, sign=0.2, labels=custom_labels, normalize_per_class=True)
        #print(cm)
        expected_CM = np.array([
            [0,0,0],
            [.5,0,0],
            [.5,1,0],
            [0,0,0],
            [0,0,1],
            [0,0,0]
        ])
        assert np.array_equal(cm.to_numpy(), expected_CM)

class TestObservedMetrics():

    @pytest.fixture
    def set_up(self):
        raw_data = np.genfromtxt(get_resource('transporters.p-values.csv'), delimiter=';', skip_header=1)
        self.true_labels = np.array([1 if x == 1.0 else 0 for x in raw_data[:,1]])
        self.p_values = raw_data[:,[2,3]]

        multiclass_data = np.genfromtxt(get_resource('multiclass.csv'), delimiter=',')
        self.m_p_values = multiclass_data[:,1:]
        self.m_true_labels = multiclass_data[:,:1].astype(int)


    def test_3D_frac_err(self, set_up):
        sign_vals = [0.7,0.8,0.9]
        (overall,class_wise) = frac_errors(self.true_labels,self.p_values,sign_vals)
        assert 3 == overall.shape[0] # One for each sign-value
        assert len(overall.shape)==1 # 1D array
        assert (len(sign_vals),self.p_values.shape[1]) == class_wise.shape

        # Test using a single significance level
        (overall,class_wise) = frac_errors(self.true_labels,self.p_values,sign_vals=0.25)
        assert 1 == len(overall)
        assert (1, self.p_values.shape[1]) == class_wise.shape
    

    def test_3D_frac_err_multiclass(self, set_up):
        sign_vals = [0.7,0.8,0.9]
        (overall,class_wise) = frac_errors(self.m_true_labels,self.m_p_values,sign_vals)
        assert 3 == overall.shape[0] # One for each sign-value
        assert len(overall.shape) == 1 # 1D array
        assert (len(sign_vals),self.m_p_values.shape[1]) == class_wise.shape
    

    def test_3D_frac_err_multiclass_only_one_cls(self, set_up):
        sign_vals = [0.7,0.8,0.9]

        only_1_index = self.m_true_labels == 1
        ys = self.m_true_labels[only_1_index]
        pvals = self.m_p_values[only_1_index.reshape(-1),:]

        (overall,class_wise) = frac_errors(ys,pvals,sign_vals)
        # The 0 and 2 classes should have error-rate of 0 for all significance levels!
        assert np.all(np.zeros(3)==class_wise[:,0])
        assert np.all(np.zeros(3)==class_wise[:,2])
        assert 3 == overall.shape[0] # One for each sign-value
        assert len(overall.shape)==1 # 1D array
        assert (len(sign_vals),self.m_p_values.shape[1]) == class_wise.shape


    def test_3D_vs_2D(self, set_up):
        sign_vals = np.arange(0,1,0.01)
        # Here check consistent results
        (overall,cls_wise) = frac_errors(self.true_labels,self.p_values,sign_vals)
        joined_overall = []
        joined_cls_wise = np.zeros((len(sign_vals),self.p_values.shape[1]))
        for i,s in enumerate(sign_vals):
            err, cls_ = frac_error(self.true_labels, self.p_values, s)
            joined_overall.append(err)
            joined_cls_wise[i,:] = cls_
        assert np.allclose(overall,np.array(joined_overall))
        assert np.allclose(joined_cls_wise, cls_wise)

        # Small benchmark of the two versions, not even considering the 
        num_iter = 0
        tic = time.perf_counter()
        for _ in range(num_iter):
            for s in sign_vals:
                _ = frac_error(self.true_labels, self.p_values, s)
        toc = time.perf_counter()
        if num_iter >10:
            print(f"For loop in {toc - tic:0.4f} seconds")

        tic = time.perf_counter()
        for _ in range(num_iter):
            _ = frac_errors(self.true_labels,self.p_values,sign_vals)
        toc = time.perf_counter()
        if num_iter >10:
            print(f"All in one in {toc - tic:0.4f} seconds") 


    def test_fraction_errors(self, set_up):
        # Taken from the values Ulf gave for this dataset
        sign = .25
        overall, (e0, e1) = frac_error(self.true_labels, self.p_values, sign)
        assert pytest.approx(0.1845) == round(overall, 4)
        assert pytest.approx(.24) == e0
        assert pytest.approx(.12,abs=1e-3) == e1

        sign=.2
        overall, (e0, e1) = frac_error(self.true_labels, self.p_values, sign)
        assert pytest.approx(.1459,abs=1e-4) == overall
        assert pytest.approx(.2) == e0
        assert pytest.approx(0.083, abs=1e-3) == e1
        
        sign=.15
        overall, (e0, e1) = frac_error(self.true_labels, self.p_values, sign)
        assert pytest.approx(.12,abs=1e-3) == overall
        assert pytest.approx(.16) == e0
        assert pytest.approx(0.074, abs=1e-3) == e1
    

    def test_single_label_ext(self, set_up):
        for s in np.arange(0,1,0.1):
            overall, correct_s, incorrect_s = frac_single_label_preds(self.true_labels, self.p_values, s)
            all_single, = frac_single_label_preds(None, self.p_values, s)
            assert pytest.approx(all_single) == correct_s+incorrect_s
    

    def test_multilabel_ext(self, set_up):
        for s in np.arange(0,1,0.1):
            overall, correct_m, incorrect_m = frac_multi_label_preds(self.true_labels, self.p_values, s)
            all_m, = frac_multi_label_preds(None, self.p_values, s)
            assert pytest.approx(all_m) == correct_m+incorrect_m
            assert 0 == incorrect_m # For binary - all multi-label are always correct!
    

    def test_multilabel_ext_3class(self, set_up):
        for s in np.arange(0,.1,0.01):
            overall, correct_m, incorrect_m = frac_multi_label_preds(self.m_true_labels, self.m_p_values, s)
            all_m, = frac_multi_label_preds(None, self.m_p_values, s)
            assert pytest.approx(correct_m+incorrect_m) == all_m

    
    def test_multilabel_ext_synthetic(self):
        p = np.array([
            [0.1,0.2,0.3,0.5],
            [0.1,0.2,0.3,0.5],
            [0.1,0.2,0.3,0.5],
            [0.1,0.2,0.3,0.5],
        ])
        s = 0.09
        all_m, correct_m, incorrect_m = frac_multi_label_preds([3,3,3,3], p, s)
        assert 1 == correct_m # All predicted and all correct
        assert 0 == incorrect_m
        assert all_m == correct_m + incorrect_m

        s = 0.11
        all_m, correct_m, incorrect_m = frac_multi_label_preds([0,0,0,3], p, s)
        assert all_m == correct_m + incorrect_m
        all_m, = frac_multi_label_preds(None, p, s)
        assert .25 == correct_m
        assert .75 == incorrect_m
        assert all_m == incorrect_m+correct_m

class TestUnobMetrics():

    @pytest.fixture
    def set_up(self):
        self.pvals_2 = np.array([
            [0.05, 0.25],
            [0.45, 0.03],
            [0.65, 0.15],
            [0.03, 0.92]
        ])

        self.pvals_3 = np.array([
            [0.05, 0.25, 0.67],
            [0.45, 0.03, 0.8],
            [0.65, 0.15, 0.05],
            [0.03, 0.92, 0.76],
            [0.23, 0.02, 0.5]
        ])

        self.pvals_4 = np.array([
            [0.05, 0.25, 0.67, 0.2],
            [0.45, 0.03, 0.8, 0.1],
            [0.65, 0.15, 0.05, 0.5]
        ])
        # Ulfs real-life data
        raw_data = np.genfromtxt(get_resource('transporters.p-values.csv'), delimiter=';', skip_header=1)
        self.p_values = raw_data[:,[2,3]]


    def test_single_label_preds(self, set_up):
        sign = 0.25
        assert pytest.approx(.974) == round(frac_single_label_preds(None, self.p_values, sign)[0],3)
        sign = 0.2
        assert pytest.approx(.893) == round(frac_single_label_preds(None,self.p_values, sign)[0],3)
        sign = 0.15
        assert pytest.approx(.79) == round(frac_single_label_preds(None,self.p_values, sign)[0],3) 


    def test_f_criteria(self, set_up):
        assert pytest.approx(mean([.05, .03, .15,.03])) == f_criteria(self.pvals_2)
        assert pytest.approx(mean([.3, .48, .2, .79, .25])) == f_criteria(self.pvals_3)
        assert pytest.approx(mean([.5, .58, .7])) == f_criteria(self.pvals_4)


    def test_u_criterion(self, set_up):
        assert pytest.approx(mean([.05, .03, .15,.03])) == u_criterion(self.pvals_2)
        assert pytest.approx(mean([.25, .45, .15, .76, .23])) == u_criterion(self.pvals_3)
        assert pytest.approx(mean([.25, .45, .5])) == u_criterion(self.pvals_4)
    

    def test_n_criterion(self, set_up):
        # sig = 0.01 > all labels predicted!
        sig = 0.01
        assert 2 == n_criterion(self.pvals_2, sig)
        assert 3 == n_criterion(self.pvals_3, sig)
        assert 4 == n_criterion(self.pvals_4, sig)
        # sig = 0.1 - most labels predicted
        sig = 0.1
        assert pytest.approx(mean([1,1,2,1])) == n_criterion(self.pvals_2, sig)
        assert 2 == n_criterion(self.pvals_3, sig)
        assert mean([3,2,3]) == n_criterion(self.pvals_4, sig)
        # sig = 0.5 - few labels
        sig = 0.5
        assert pytest.approx(.5) == n_criterion(self.pvals_2, sig)
        assert pytest.approx(mean([1,1,1,2,0])) == n_criterion(self.pvals_3, sig)
        assert pytest.approx(1) == n_criterion(self.pvals_4, sig)
        # sig = 1.0 - no labels predicted
        sig = 1.0
        assert 0 == n_criterion(self.pvals_2, sig)
        assert 0 == n_criterion(self.pvals_3, sig)
        assert 0 == n_criterion(self.pvals_4, sig)


    def test_s_criterion(self, set_up):
        assert pytest.approx(mean([.3, .48, .8, .95])) == s_criterion(self.pvals_2)
        assert pytest.approx(mean([.97, 1.28, .85, 1.71, .75])) == s_criterion(self.pvals_3)
        assert pytest.approx(mean([1.17, 1.38, 1.35])) == s_criterion(self.pvals_4)


    def test_confidence(self, set_up):
        assert pytest.approx(mean([.95, .97, .85, .97])) == cp_confidence(self.pvals_2)
        assert pytest.approx(mean([.75, .55, .85, .24, .77])) == cp_confidence(self.pvals_3)
        assert pytest.approx(mean([.75, .55, .5])) == cp_confidence(self.pvals_4)
    

    def test_credibility(self, set_up):
        assert pytest.approx(mean([.25, .45,.65, .92])) == cp_credibility(self.pvals_2)
        assert pytest.approx(mean([.67, .8, .65, .92, .5])) == cp_credibility(self.pvals_3)
        assert pytest.approx(mean([.67, .8, .65])) == cp_credibility(self.pvals_4)
    


