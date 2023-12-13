#import numpy as np
#import pandas as pd
import unittest

from pharmbio import data
from ...help_utils import get_resource

class Test_regression(unittest.TestCase):

    def test_load(self):
        (y,predictions,signs) = data.load_regression(get_resource('cpsign_reg_predictions.csv'),'y')
        self.assertEqual(len(y), len(predictions))
        self.assertEqual(predictions.shape[2],len(signs))
        self.assertAlmostEqual(signs[0],1)
        

if __name__ == '__main__':
    unittest.main()