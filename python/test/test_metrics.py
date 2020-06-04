import numpy as np
import unittest

import sys
sys.path.append('../src')
from pharmbio.cp import metrics

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
        
        cm = metrics.calc_confusion_matrix(tr,p_vals_m, significance=0.2)
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

        cm = metrics.calc_confusion_matrix(tr,p_vals_m, significance=0.2)
        self.assertTrue(np.array_equal(cm.to_numpy(), expected_CM))
    
    def test_with_custom_labels(self):
        p_vals_m = np.array([[0.05, 0.85], [0.23, 0.1], [.1, .1], [.21, .21], [.3, 0.15]])
        tr = np.array([0,1,0,1,0])
        custom_labels = ['Mutagen', 'Nonmutagen']
        expected_CM = np.array([[1, 1], [1,0], [1,0], [0,1]])

        cm = metrics.calc_confusion_matrix(tr,p_vals_m, significance=0.2, class_labels=custom_labels)
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
        cm = metrics.calc_confusion_matrix(true_l,p_vals_m, significance=0.2, class_labels=custom_labels)

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
        cm = metrics.calc_confusion_matrix(true_l,p_vals_m, significance=0.2, class_labels=custom_labels, normalize_per_class=True)
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

if __name__ == '__main__':
    unittest.main()