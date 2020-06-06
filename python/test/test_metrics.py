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
        
        cm = calc_confusion_matrix(tr,p_vals_m, significance=0.2)
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

        cm = calc_confusion_matrix(tr,p_vals_m, significance=0.2)
        self.assertTrue(np.array_equal(cm.to_numpy(), expected_CM))
    
    def test_with_custom_labels(self):
        p_vals_m = np.array([[0.05, 0.85], [0.23, 0.1], [.1, .1], [.21, .21], [.3, 0.15]])
        tr = np.array([0,1,0,1,0])
        custom_labels = ['Mutagen', 'Nonmutagen']
        expected_CM = np.array([[1, 1], [1,0], [1,0], [0,1]])

        cm = calc_confusion_matrix(tr,p_vals_m, significance=0.2, class_labels=custom_labels)
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
        cm = calc_confusion_matrix(true_l,p_vals_m, significance=0.2, class_labels=custom_labels)

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
        cm = calc_confusion_matrix(true_l,p_vals_m, significance=0.2, class_labels=custom_labels, normalize_per_class=True)
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