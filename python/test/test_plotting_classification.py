import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys

from numpy.core.numeric import False_
sys.path.append('../src')
from pharmbio.cp import metrics,plotting
from test_utils import _save_clf

# Some testing data - 2 class
my_data = np.genfromtxt('resources/transporters.p-values.csv', delimiter=';', skip_header=1)
true_labels_2_class = (my_data[:,1] == 1).astype(np.int16)
p_vals_2_class = my_data[:,[2,3]]
cm_2_class_015 = metrics.confusion_matrix( true_labels_2_class, p_vals_2_class, sign=0.15 )
cm_2_class_075 = metrics.confusion_matrix( true_labels_2_class, p_vals_2_class, sign=0.75 )

# 3 class
data3class = np.genfromtxt('resources/multiclass.csv', delimiter=',', skip_header=0)
true_labels_3_class = data3class[:,0].astype(np.int16)
p_vals_3_class = data3class[:,1:]
cm_3_class_015 = metrics.confusion_matrix( true_labels_3_class, p_vals_3_class, sign=0.15 )
# print(type(true_labels_3class))


class TestPValuesPlot(unittest.TestCase):
    def test_2_class(self):
        fig = plotting.plot_pvalues(true_labels_2_class, p_values=p_vals_2_class)
        fig.axes[0].set_title('p0/p1 2-class')
        _save_clf(fig,"TestPValuesPlot.test_2_class")
    
    def test_3_class_01(self):
        fig = plotting.plot_pvalues(true_labels_3_class, p_values=p_vals_3_class)
        fig.axes[0].set_title('p0/p1 3-class')
        _save_clf(fig,"TestPValuesPlot.test_3_class01")
    
    def test_3_class_21(self):
        fig = plotting.plot_pvalues(true_labels_3_class, p_values=p_vals_3_class, cols=[2,1])
        fig.axes[0].set_title('p2/p1 3-class')
        _save_clf(fig,"TestPValuesPlot.test_3_class21")
    
    def test_3_class_only_send_2pvals(self):
        fig = plotting.plot_pvalues(true_labels_3_class, p_values=p_vals_3_class[:,[0,1]])
        fig.axes[0].set_title('p0/p1 3-class (2-vals sent)')
        _save_clf(fig,"TestPValuesPlot.test_3_class_only_send_2pvals")
    
    def test_cols_outside_range(self):
        with self.assertRaises(ValueError):
            plotting.plot_pvalues(true_labels_2_class, p_values=p_vals_2_class, cols=[2,1])

class TestLabelDistributionPlot(unittest.TestCase):

    def test_2_class(self):
        fig1 = plotting.plot_label_distribution(true_labels_2_class, p_values=p_vals_2_class)
        fig1.axes[0].set_title('LabelDistribution 2-class')
        _save_clf(fig1,"TestLabelDistPlot.test_2_class")
    
    def test_3_class(self):
        fig = plotting.plot_label_distribution(true_labels_3_class, p_values=p_vals_3_class)
        fig.axes[0].set_title('LabelDistribution 3-class')
        _save_clf(fig,"TestLabelDistPlot.test_3_class")

class TestCalibrationPlot(unittest.TestCase):
    def test_2_class(self):
        fig1 = plotting.plot_calibration_curve(true_labels_2_class, p_values=p_vals_2_class)
        fig1.axes[0].set_title('Calib plot 2-class')
        fig2 = plotting.plot_calibration_curve(true_labels_2_class, p_values=p_vals_2_class, labels = ['class 0', 'class 1'])
        fig2.axes[0].set_title('Calib plot 2-class with labels')
        _save_clf(fig2,"TestCalibPlot.test_2_class")
    
    def test_3_class(self):
        fig = plotting.plot_calibration_curve(true_labels_3_class, p_values=p_vals_3_class, labels = ['A', 'B', 'C'])
        fig.axes[0].set_title('Calib plot 3-class, labels={A,B,C}')
        _save_clf(fig,"TestCalibPlot.test_3_class")
    
    def test_3_class_conf_acc(self):
        # Plot all in one image
        fig, axes = plt.subplots(2,2,figsize=(10,10))
        # std
        plotting.plot_calibration_curve(true_labels_3_class, p_values=p_vals_3_class, labels = ['A', 'B', 'C'], ax=axes[0,0], title='std')
        # flip x
        plotting.plot_calibration_curve(true_labels_3_class, p_values=p_vals_3_class, labels = ['A', 'B', 'C'], ax=axes[0,1], flip_x=True, title='flip x')
        # flip y
        plotting.plot_calibration_curve(true_labels_3_class, p_values=p_vals_3_class, labels = ['A', 'B', 'C'], ax=axes[1,0], flip_y=True, title='flip y')
        # flip both
        plotting.plot_calibration_curve(true_labels_3_class, p_values=p_vals_3_class, labels = ['A', 'B', 'C'], ax=axes[1,1], flip_x=True, flip_y=True, title='both')
        _save_clf(fig,"TestCalibPlot.test_3_class_flip")


class TestBubblePlot(unittest.TestCase):
    
    def test_3_class(self):
        fig1 = plotting.plot_confusion_matrix_bubbles(cm_3_class_015)
        fig1.axes[0].set_title('Bubbles 3-class 0.15')
        _save_clf(fig1,"TestBubblebPlot.test_3_class")
    
    def test_2_class(self):
        fig2 = plotting.plot_confusion_matrix_bubbles(cm_2_class_015, annotate=False, scale_factor=5.5, figsize=(6,7))
        fig2.axes[0].set_title('Bubbles 2-class 0.15 - no annotation - scale 5.5')
        _save_clf(fig2,"TestBubblebPlot.test_2_class_1")

        fig3 = plotting.plot_confusion_matrix_bubbles(cm_2_class_075)
        fig3.axes[0].set_title('Bubbles 2-class 0.75')
        _save_clf(fig3,"TestBubblebPlot.test_2_class_2")

class TestConfusionMatrixHeatmap(unittest.TestCase):

    def test_3_class(self):
        fig1 = plotting.plot_confusion_matrix_heatmap(cm_3_class_015)
        fig1.axes[0].set_title('Heatmap 3-class 0.15')
        _save_clf(fig1,"TestConfMatrixHeatMap.test_3_class")
    
    def test_2_class(self):
        fig2 = plotting.plot_confusion_matrix_heatmap(cm_2_class_015, cmap="YlGnBu")
        fig2.axes[0].set_title('Heatmap 2-class 0.15 (YllGnBu colormap)')
        _save_clf(fig2,"TestConfMatrixHeatMap.test_2_class_1")
        fig3 = plotting.plot_confusion_matrix_heatmap(cm_2_class_075)
        fig3.axes[0].set_title('Heatmap 2-class 0.75')
        _save_clf(fig3,"TestConfMatrixHeatMap.test_2_class_2")

class FinalTest(unittest.TestCase):

    def display_plots(self):
        plt.show()


if __name__ == '__main__':
    unittest.main()