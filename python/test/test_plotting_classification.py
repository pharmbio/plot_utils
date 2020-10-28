import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../src')
from pharmbio.cp import metrics,plotting

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
        plt.show()
    
    def test_3_class_01(self):
        fig = plotting.plot_pvalues(true_labels_3_class, p_values=p_vals_3_class)
        fig.axes[0].set_title('p0/p1 3-class')
        plt.show()
    
    def test_3_class_21(self):
        fig = plotting.plot_pvalues(true_labels_3_class, p_values=p_vals_3_class, cols=[2,1])
        fig.axes[0].set_title('p2/p1 3-class')
        plt.show()
    
    def test_3_class_only_send_2pvals(self):
        fig = plotting.plot_pvalues(true_labels_3_class, p_values=p_vals_3_class[:,[0,1]])
        fig.axes[0].set_title('p0/p1 3-class (2-vals sent)')
        plt.show()
    
    # @unittest.expectedFailure
    def test_cols_outside_range(self):
        with self.assertRaises(ValueError):
            plotting.plot_pvalues(true_labels_2_class, p_values=p_vals_2_class, cols=[2,1])
        # try:
            
        # except (ValueError)
        # plt.show()

class TestLabelDistributionPlot(unittest.TestCase):

    def test_2_class(self):
        fig1 = plotting.plot_label_distribution(true_labels_2_class, p_values=p_vals_2_class)
        fig1.axes[0].set_title('LabelDistribution 2-class')
        plt.show()
    
    def test_3_class(self):
        fig = plotting.plot_label_distribution(true_labels_3_class, p_values=p_vals_3_class)
        fig.axes[0].set_title('LabelDistribution 3-class')
        plt.show()

class TestCalibrationPlot(unittest.TestCase):
    def test_2_class(self):
        fig1 = plotting.plot_calibration_curve(true_labels_2_class, p_values=p_vals_2_class)
        fig1.axes[0].set_title('Calib plot 2-class')
        fig2 = plotting.plot_calibration_curve(true_labels_2_class, p_values=p_vals_2_class, labels = ['class 0', 'class 1'])
        fig2.axes[0].set_title('Calib plot 2-class with labels')
        plt.show()
    
    def test_3_class(self):
        fig = plotting.plot_calibration_curve(true_labels_3_class, p_values=p_vals_3_class, labels = ['A', 'B', 'C'])
        fig.axes[0].set_title('Calib plot 3-class')
        plt.show()


class TestBubblePlot(unittest.TestCase):
    
    def test_3_class(self):
        fig1 = plotting.plot_confusion_matrix_bubbles(cm_3_class_015)
        fig1.axes[0].set_title('Bubbles 3-class 0.15')
        plt.show()
    
    def test_2_class(self):
        fig2 = plotting.plot_confusion_matrix_bubbles(cm_2_class_015)
        fig2.axes[0].set_title('Bubbles 2-class 0.15')
        fig3 = plotting.plot_confusion_matrix_bubbles(cm_2_class_075)
        fig3.axes[0].set_title('Bubbles 2-class 0.75')
        plt.show()

class TestConfusionMatrixHeatmap(unittest.TestCase):

    def test_3_class(self):
        fig1 = plotting.plot_confusion_matrix_heatmap(cm_3_class_015)
        fig1.axes[0].set_title('Heatmap 3-class 0.15')
        plt.show()
    
    def test_2_class(self):
        fig2 = plotting.plot_confusion_matrix_heatmap(cm_2_class_015, cmap="YlGnBu")
        fig2.axes[0].set_title('Heatmap 2-class 0.15')
        fig3 = plotting.plot_confusion_matrix_heatmap(cm_2_class_075)
        fig3.axes[0].set_title('Heatmap 2-class 0.75')
        plt.show()

class FinalTest(unittest.TestCase):

    def display_plots(self):
        plt.show()


if __name__ == '__main__':
    unittest.main()