import numpy as np
import pandas as pd
import unittest

import sys
sys.path.append('../src')
from pharmbio.data import *

class Test_regression(unittest.TestCase):

    def test_load(self):
        (y,preds,signs) = load_regression('resources/pred_file_reg.csv','y')
        self.assertEqual(len(y), len(preds))
        self.assertEqual(preds.shape[2],len(signs))
        self.assertAlmostEqual(signs[0],1)
        

if __name__ == '__main__':
    unittest.main()