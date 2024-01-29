import numpy as np
import pytest
import matplotlib.pyplot as plt

from pharmbio.cp import metrics,plotting
from ....help_utils import _save_clf, get_resource

# Some testing data - 2 class
my_data = np.genfromtxt(get_resource('transporters.p-values.csv'), delimiter=';', skip_header=1)
true_labels_2_class = (my_data[:,1] == 1).astype(np.int16)
p_vals_2_class = my_data[:,[2,3]]
cm_2_class_015 = metrics.confusion_matrix( true_labels_2_class, p_vals_2_class, sign=0.15 )
cm_2_class_015_normalized = metrics.confusion_matrix( true_labels_2_class, p_vals_2_class, sign=0.15, normalize_per_class=True)
cm_2_class_075 = metrics.confusion_matrix( true_labels_2_class, p_vals_2_class, sign=0.75 )

# 3 class
data3class = np.genfromtxt(get_resource('multiclass.csv'), delimiter=',', skip_header=0)
true_labels_3_class = data3class[:,0].astype(np.int16)
p_vals_3_class = data3class[:,1:]
cm_3_class_015 = metrics.confusion_matrix( true_labels_3_class, p_vals_3_class, sign=0.15 )

# hER predictions
er_data = np.genfromtxt(get_resource('er.p-values.csv'), delimiter=',', skip_header=1)
er_labels = er_data[:,0].astype(np.int16)
er_pvals = er_data[:,1:]


class TestPValuesPlot():
    def test_2_class(self):
        fig = plotting.plot_pvalues(true_labels_2_class, p_values=p_vals_2_class)
        fig.axes[0].set_title('p0/p1 2-class')
        _save_clf(fig,"TestPValuesPlot.test_2_class")
    
    def test_3_class_01(self):
        fig = plotting.plot_pvalues(true_labels_3_class, p_values=p_vals_3_class,split_chart=False)
        fig.axes[0].set_title('p0/p1 3-class')
        _save_clf(fig,"TestPValuesPlot.test_3_class01")
    
    def test_3_class_21(self):
        fig = plotting.plot_pvalues(true_labels_3_class, p_values=p_vals_3_class, cols=[2,1])
        fig.axes[0].set_title('p2/p1 3-class')
        _save_clf(fig,"TestPValuesPlot.test_3_class21")
    
    def test_3_class_only_send21(self):
        excl_filter = true_labels_3_class == 0

        fig = plotting.plot_pvalues(true_labels_3_class[~excl_filter], p_values=p_vals_3_class[~excl_filter], cols=[2,1])
        fig.axes[0].set_title('p2/p1 3-class only single class 1 and 2')
        _save_clf(fig,"TestPValuesPlot.test_3_class21_rm0")
    
    def test_3_class_only_send_2pvals(self):
        fig = plotting.plot_pvalues(true_labels_3_class, p_values=p_vals_3_class[:,[0,1]],cm=['r','b','k'],title='p0/p1 3-class (2-vals sent)')
        fig.axes[0].set_title('p0/p1 3-class (2-vals sent)')
        _save_clf(fig,"TestPValuesPlot.test_3_class_only_send_2pvals")
    
    def test_cols_outside_range(self):
        with pytest.raises(ValueError):
            plotting.plot_pvalues(true_labels_2_class, p_values=p_vals_2_class, cols=[2,1])

    def test_her(self):
        import matplotlib as mpl
        from matplotlib.markers import MarkerStyle
        import seaborn as sns
        sns.set_style('ticks')

        non_filled_o = MarkerStyle(marker='o', fillstyle='none')

        markers_ls = [None,'*',['*','o'],[non_filled_o,'*']]
        colors = [None, ['r','b'],'y']
        
        # FREQ order
        # print("FREQ ORDER")
        freq_fig, axes = plt.subplots(3,4,figsize = (5*4,5*3))
        for row, c in enumerate(colors):
            for col, m in enumerate(markers_ls):
                plotting.plot_pvalues(er_labels, er_pvals,
                    ax=axes[row,col], 
                    order=None,
                    cm = c,
                    alphas= [.9,.5],
                    markers=m,
                    linewidths=.5,
                    fontargs={'fontsize':'medium'})
        freq_fig.tight_layout()
        _save_clf(freq_fig,"TestPValuesPlot.hER_all_freq")

        # print("CLS ORDER")
        label_fig, axes = plt.subplots(3,4,figsize = (5*4,5*3))
        # class order
        for row, c in enumerate(colors):
            for col, m in enumerate(markers_ls):
                plotting.plot_pvalues(er_labels, er_pvals,
                    ax=axes[row,col], 
                    order='class',
                    alpha=0.8, # Same alpha for both
                    # alphas = [.8,.75], # default
                    cm = c,
                    markers=m,
                    sizes = mpl.rcParams['lines.markersize']**2,
                    lw=1.5,
                    fontargs={'fontsize':'large'})
        label_fig.tight_layout()
        _save_clf(label_fig,"TestPValuesPlot.hER_all_class")

        # print("REV ORDER")
        # reverse class order
        rev_label_fig, axes = plt.subplots(3,4,figsize = (5*4,5*3))
        for row, c in enumerate(colors):
            for col, m in enumerate(markers_ls):
                plotting.plot_pvalues(er_labels, er_pvals,
                    ax=axes[row,col], 
                    order='rev class',
                    cm = c,
                    alphas= [.3,.75],
                    markers=m,fontargs={'fontsize':'x-large'})
        rev_label_fig.tight_layout()
        _save_clf(rev_label_fig,"TestPValuesPlot.hER_all_rev_class")


class TestLabelDistributionPlot():

    def test_2_class(self):
        fig1 = plotting.plot_label_distribution(y_true = true_labels_2_class, p_values=p_vals_2_class)
        fig1.axes[0].set_title('LabelDistribution 2-class')
        _save_clf(fig1,"TestLabelDistPlot.test_2_class")
    
    def test_3_class(self):
        fig = plotting.plot_label_distribution(y_true = true_labels_3_class, p_values=p_vals_3_class)
        fig.axes[0].set_title('LabelDistribution 3-class')
        _save_clf(fig,"TestLabelDistPlot.test_3_class")

class TestCalibrationPlot():
    def test_2_class(self):
        fig1 = plotting.plot_calibration_clf(true_labels_2_class, p_vals_2_class)
        fig1.axes[0].set_title('Calib plot 2-class')
        fig2 = plotting.plot_calibration_clf(true_labels_2_class, p_vals_2_class, labels = ['class 0', 'class 1'])
        fig2.axes[0].set_title('Calib plot 2-class with labels')
        _save_clf(fig2,"TestCalibPlot.test_2_class")
    
    def test_3_class(self):
        fig = plotting.plot_calibration_clf(true_labels_3_class, p_vals_3_class, labels = ['A', 'B', 'C'])
        fig.axes[0].set_title('Calib plot 3-class, labels={A,B,C}')
        _save_clf(fig,"TestCalibPlot.test_3_class")
    
    def test_3_class_conf_acc(self):
        # Plot all in one image
        fig, axes = plt.subplots(2,2,figsize=(10,10))
        # std
        plotting.plot_calibration_clf(true_labels_3_class, p_values=p_vals_3_class, labels = ['A', 'B', 'C'], ax=axes[0,0], title='std')
        # flip x
        plotting.plot_calibration_clf(true_labels_3_class, p_values=p_vals_3_class, labels = ['A', 'B', 'C'], ax=axes[0,1], flip_x=True, title='flip x')
        # flip y
        plotting.plot_calibration_clf(true_labels_3_class, p_values=p_vals_3_class, labels = ['A', 'B', 'C'], ax=axes[1,0], flip_y=True, title='flip y')
        # flip both
        plotting.plot_calibration_clf(true_labels_3_class, p_values=p_vals_3_class, labels = ['A', 'B', 'C'], ax=axes[1,1], flip_x=True, flip_y=True, title='both')
        _save_clf(fig,"TestCalibPlot.test_3_class_flip")


class TestBubblePlot():
    
    def test_3_class(self):
        fig1 = plotting.plot_confusion_matrix_bubbles(cm_3_class_015,color_scheme=None)
        fig1.axes[0].set_title('Bubbles 3-class 0.15')
        _save_clf(fig1,"TestBubblebPlot.test_3_class")
    
    def test_2_class_percentage(self):
        fig2 = plotting.plot_confusion_matrix_bubbles(cm_2_class_015_normalized, annotate=True, annotate_as_percentage=True, figsize=(6,7))
        fig2.axes[0].set_title('Bubbles 2-class 0.15 - percentage - scale 5.5')
        _save_clf(fig2,"TestBubblebPlot.test_2_class_1_percentage")

        # Test without normalized CM
        with pytest.raises(ValueError):
            _ = plotting.plot_confusion_matrix_bubbles(cm_2_class_015, annotate=True, annotate_as_percentage=True, figsize=(6,7))
    
    def test_2_class(self):
        fig2 = plotting.plot_confusion_matrix_bubbles(cm_2_class_015, annotate=False, scale_factor=5.5, figsize=(6,7))
        fig2.axes[0].set_title('Bubbles 2-class 0.15 - no annotation - scale 5.5')
        _save_clf(fig2,"TestBubblebPlot.test_2_class_1")

        fig3 = plotting.plot_confusion_matrix_bubbles(cm_2_class_075)
        fig3.axes[0].set_title('Bubbles 2-class 0.75')
        _save_clf(fig3,"TestBubblebPlot.test_2_class_2")
    
    def test_illegal_color_scheme(self):
        with pytest.warns(UserWarning):
            fig_ = plotting.plot_confusion_matrix_bubbles(cm_2_class_015, color_scheme='bad_arg', annotate=False, scale_factor=5.5, figsize=(6,7))

class TestConfusionMatrixHeatmap():

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

class FinalTest():

    def display_plots(self):
        plt.show()

