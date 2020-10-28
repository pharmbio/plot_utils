import numpy as np
import pandas as pd
import unittest
import matplotlib.pyplot as plt

import pharmbio.cp.plotting._utils as plt_utils

class TestPlottingUtils(unittest.TestCase):
    
    def test_generate_figure(self):
        fig, ax = plt_utils.get_fig_and_axis()
        ax.set_title('Default fig')
        
        fig, ax = plt_utils.get_fig_and_axis(figsize=10)
        ax.set_title('fig 10')
        
        fig, ax = plt_utils.get_fig_and_axis(figsize=(10,5))
        ax.set_title('fig 10, 5')
        
        plt.show()

if __name__ == '__main__':
    unittest.main()
