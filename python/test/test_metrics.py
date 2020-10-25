import numpy as np
import unittest

import sys
sys.path.append('../src')
from pharmbio.cp.metrics import *
from statistics import mean 

class TestConfusionMatrix(unittest.TestCase):

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
        
        cm = calc_confusion_matrix(tr,p_vals_m, sign=0.2)
        self.assertTrue(np.array_equal(cm.to_numpy(), expected_CM))
    
    def test_small_binary_ex(self):
        p_vals_m = np.array([[0.05, 0.85], [0.23, 0.1], [.1, .1], [.21, .21], [.3, 0.15]])
        tr = np.array([0,1,0,1,0])
        expected_CM = np.array([
            [1,1], 
            [1,0], 
            [1,0], 
            [0,1]
            ])

        cm = calc_confusion_matrix(tr,p_vals_m, sign=0.2)
        self.assertTrue(np.array_equal(cm.to_numpy(), expected_CM))
    
    def test_with_custom_labels(self):
        p_vals_m = np.array([[0.05, 0.85], [0.23, 0.1], [.1, .1], [.21, .21], [.3, 0.15]])
        tr = np.array([0,1,0,1,0])
        custom_labels = ['Mutagen', 'Nonmutagen']
        expected_CM = np.array([[1, 1], [1,0], [1,0], [0,1]])

        cm = calc_confusion_matrix(tr,p_vals_m, sign=0.2, labels=custom_labels)
        self.assertEqual(cm.shape, expected_CM.shape)
        self.assertTrue(np.array_equal(cm.to_numpy(), expected_CM))

        cm_names = list(cm.columns)
        self.assertEqual(custom_labels, cm_names)
        self.assertEqual(custom_labels, list(cm.index.values)[:2])
        self.assertEqual(4, len(list(cm.index.values)))

        self.assertEqual(len(tr), cm.to_numpy().sum())

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
        cm = calc_confusion_matrix(true_l,p_vals_m, sign=0.2, labels=custom_labels)

        expected_CM = np.array([
            [0,0,0],
            [1,0,0],
            [1,1,0],
            [0,0,0],
            [0,0,1],
            [0,0,0]
        ])
        self.assertTrue(np.array_equal(cm.to_numpy(), expected_CM))
    
    
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
        cm = calc_confusion_matrix(true_l,p_vals_m, sign=0.2, labels=custom_labels, normalize_per_class=True)
        #print(cm)
        expected_CM = np.array([
            [0,0,0],
            [.5,0,0],
            [.5,1,0],
            [0,0,0],
            [0,0,1],
            [0,0,0]
        ])
        self.assertTrue(np.array_equal(cm.to_numpy(), expected_CM))

class TestObservedMetrics(unittest.TestCase):

    def setUp(self):
        raw_data = np.genfromtxt('resources/transporters.p-values.csv', delimiter=';', skip_header=1)
        self.true_labels = np.array([1 if x == 1.0 else 0 for x in raw_data[:,1]])
        self.p_values = raw_data[:,[2,3]]

        multiclass_data = np.genfromtxt('resources/multiclass.csv', delimiter=',')
        self.m_p_values = multiclass_data[:,1:]
        self.m_true_labels = multiclass_data[:,:1].astype(np.int)

    
    def test_error_rate(self):
        # Taken from the values Ulf gave for this dataset
        sign = .25
        overall, (e0, e1) = calc_error_rate(self.true_labels, self.p_values, sign)
        self.assertAlmostEqual(0.1845, round(overall, 4))
        self.assertAlmostEqual(.24, round(e0,3))
        self.assertAlmostEqual(.12, round(e1,3))

        sign=.2
        overall, (e0, e1) = calc_error_rate(self.true_labels, self.p_values, sign)
        self.assertAlmostEqual(.1459, round(overall, 4))
        self.assertAlmostEqual(.2, round(e0,3))
        self.assertAlmostEqual(1-0.917, round(e1,3))
        
        sign=.15
        overall, (e0, e1) = calc_error_rate(self.true_labels, self.p_values, sign)
        self.assertAlmostEqual(.12, round(overall, 3))
        self.assertAlmostEqual(1-.84, round(e0,3))
        self.assertAlmostEqual(1-.926, round(e1,3))
    
    def test_single_label_ext(self):
        for s in np.arange(0,1,0.1):
            correct_s, incorrect_s = calc_single_label_preds_ext(self.true_labels, self.p_values, s)
            all_single = calc_single_label_preds(self.p_values, s)
            self.assertAlmostEqual(all_single, correct_s+incorrect_s)
    
    def test_multilabel_ext(self):
        for s in np.arange(0,1,0.1):
            correct_m, incorrect_m = calc_multi_label_preds_ext(self.true_labels, self.p_values, s)
            all_m = calc_multi_label_preds(self.p_values, s)
            self.assertAlmostEqual(all_m, correct_m+incorrect_m)
            self.assertEqual(0, incorrect_m) # For binary - all multi-label are always correct!
    
    def test_multilabel_ext_3class(self):
        for s in np.arange(0,.1,0.01):
            correct_m, incorrect_m = calc_multi_label_preds_ext(self.m_true_labels, self.m_p_values, s)
            all_m = calc_multi_label_preds(self.m_p_values, s)
            self.assertAlmostEqual(all_m, correct_m+incorrect_m)
    
    def test_multilabel_ext_synthetic(self):
        p = np.array([
            [0.1,0.2,0.3,0.5],
            [0.1,0.2,0.3,0.5],
            [0.1,0.2,0.3,0.5],
            [0.1,0.2,0.3,0.5],
        ])
        s = 0.09
        correct_m, incorrect_m = calc_multi_label_preds_ext([3,3,3,3], p, s)
        self.assertEqual(1, correct_m) # All predicted and all correct
        self.assertEqual(0, incorrect_m)

        s = 0.11
        correct_m, incorrect_m = calc_multi_label_preds_ext([0,0,0,3], p, s)
        all_m = calc_multi_label_preds(p, s)
        self.assertEqual( .25, correct_m)
        self.assertEqual( .75, incorrect_m)
        self.assertEqual(all_m, incorrect_m+correct_m)

class TestUnobMetrics(unittest.TestCase):

    def setUp(self):
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
        raw_data = np.genfromtxt('resources/transporters.p-values.csv', delimiter=';', skip_header=1)
        self.p_values = raw_data[:,[2,3]]

    def test_single_label_preds(self):
        sign = 0.25
        self.assertAlmostEqual(.974, round(calc_single_label_preds(self.p_values, sign),3))
        sign = 0.2
        self.assertAlmostEqual(.893, round(calc_single_label_preds(self.p_values, sign),3))
        sign = 0.15
        self.assertAlmostEqual(.79, round(calc_single_label_preds(self.p_values, sign),3))

    def test_f_criteria(self):
        self.assertAlmostEqual(mean([.05, .03, .15,.03]),calc_f_criteria(self.pvals_2))
        self.assertAlmostEqual(mean([.3, .48, .2, .79, .25]),calc_f_criteria(self.pvals_3))
        self.assertAlmostEqual(mean([.5, .58, .7]),calc_f_criteria(self.pvals_4))

    def test_u_criterion(self):
        self.assertAlmostEqual(mean([.05, .03, .15,.03]),calc_u_criterion(self.pvals_2))
        self.assertAlmostEqual(mean([.25, .45, .15, .76, .23]),calc_u_criterion(self.pvals_3))
        self.assertAlmostEqual(mean([.25, .45, .5]),calc_u_criterion(self.pvals_4))
    
    def test_n_criterion(self):
        # sig = 0.01 > all labels predicted!
        sig = 0.01
        self.assertEqual(2,calc_n_criterion(self.pvals_2, sig))
        self.assertEqual(3,calc_n_criterion(self.pvals_3, sig))
        self.assertEqual(4,calc_n_criterion(self.pvals_4, sig))
        # sig = 0.1 - most labels predicted
        sig = 0.1
        self.assertAlmostEqual(mean([1,1,2,1]),calc_n_criterion(self.pvals_2, sig))
        self.assertEqual(2,calc_n_criterion(self.pvals_3, sig))
        self.assertEqual(mean([3,2,3]),calc_n_criterion(self.pvals_4, sig))
        # sig = 0.5 - few labels
        sig = 0.5
        self.assertAlmostEqual(.5,calc_n_criterion(self.pvals_2, sig))
        self.assertAlmostEqual(mean([1,1,1,2,0]),calc_n_criterion(self.pvals_3, sig))
        self.assertAlmostEqual(1,calc_n_criterion(self.pvals_4, sig))
        # sig = 1.0 - no labels predicted
        sig = 1.0
        self.assertEqual(0,calc_n_criterion(self.pvals_2, sig))
        self.assertEqual(0,calc_n_criterion(self.pvals_3, sig))
        self.assertEqual(0,calc_n_criterion(self.pvals_4, sig))

    def test_s_criterion(self):
        self.assertAlmostEqual(mean([.3, .48, .8, .95]),calc_s_criterion(self.pvals_2))
        self.assertAlmostEqual(mean([.97, 1.28, .85, 1.71, .75]),calc_s_criterion(self.pvals_3))
        self.assertAlmostEqual(mean([1.17, 1.38, 1.35]),calc_s_criterion(self.pvals_4))

    def test_confidence(self):
        self.assertAlmostEqual(mean([.95, .97, .85, .97]),calc_confidence(self.pvals_2))
        self.assertAlmostEqual(mean([.75, .55, .85, .24, .77]),calc_confidence(self.pvals_3))
        self.assertAlmostEqual(mean([.75, .55, .5]),calc_confidence(self.pvals_4))
    
    def test_credibility(self):
        self.assertAlmostEqual(mean([.25, .45,.65, .92]),calc_credibility(self.pvals_2))
        self.assertAlmostEqual(mean([.67, .8, .65, .92, .5]),calc_credibility(self.pvals_3))
        self.assertAlmostEqual(mean([.67, .8, .65]),calc_credibility(self.pvals_4))
    


if __name__ == '__main__':
    unittest.main()