import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure 
import matplotlib.pyplot as plt
from pharmbio.cp import metrics
import os

import pharmbio.cp.plotting._utils as plt_utils

from .context import output_dir, resource_dir
# Utility functions

def get_resource(file_name:str)->str:
    return os.path.join(resource_dir,file_name)

def _save_reg(fig, test_func, close_after=True):
    test_func = test_func if test_func is not None and isinstance(test_func,str) else str(test_func)
    ax1 = fig.axes[0]
    ax1_orig_title = ax1.get_title()
    ax1.set_title(ax1_orig_title+":"+test_func)
    fig.savefig(output_dir+'/reg/'+test_func+".pdf", bbox_inches='tight')
    ax1.set_title(ax1_orig_title)
    if close_after:
        plt.close(fig)

def _save_clf(fig, test_func, close_after=True):
    test_func = test_func if test_func is not None and isinstance(test_func,str) else str(test_func)
    ax1 = fig.axes[0]
    ax1_orig_title = ax1.get_title()
    ax1.set_title(ax1_orig_title+":"+test_func)
    fig.savefig(output_dir+'/clf/'+test_func+".pdf", bbox_inches='tight')
    ax1.set_title(ax1_orig_title)
    if close_after:
        plt.close(fig)

def assert_fig_wh(fig : Figure, w : int, h : int, ax):
    """
    fig: matplotlib.Figure
    w: float, expected width of fig
    h: float, expected height of fig
    ax: Figure.Axes that should be part of the `fig` object

    """
    
    assert w == fig.get_figwidth()
    assert h == fig.get_figheight()
    
    ax_found = False
    assert ax is not None, "Axes should not be None"
    for a in fig.axes:
        if a == ax:
            ax_found = True
    assert ax_found, "Should found axes"

class TestMetricUtils():

    def test_get_onehot_bool(self):
        y_true = [1, 0, 1, 2, 0]
        one_hot, cats = metrics.to_numpy1D_onehot(y_true,'test')
        assert (5,3) == one_hot.shape
        assert np.all(np.ones(5) == one_hot.sum(axis=1))
        assert np.all(cats == np.unique(y_true))

class TestPlottingUtils():

    def test_generate_figure(self):
        fig, ax = plt_utils.get_fig_and_axis()
        ax.set_title('Default fig')
        assert_fig_wh(fig,10,8,ax)
        
        fig, ax = plt_utils.get_fig_and_axis(figsize=10)
        assert_fig_wh(fig,10,10,ax)

        fig, ax = plt_utils.get_fig_and_axis(figsize=(10,5))
        assert_fig_wh(fig,10,5,ax)
    
    def test_existing_ax_object(self):
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(2,3))
        assert_fig_wh(fig,2,3,ax1)
        assert_fig_wh(fig,2,3,ax2)
    
    def test_faulty_figsize_param(self):
        with pytest.raises(TypeError):
            plt_utils.get_fig_and_axis(figsize="")
    
    def test_faulty_ax_param(self):
        with pytest.raises(TypeError):
            plt_utils.get_fig_and_axis(ax=1)
    
    def test_utility_save_func(self):
        fig, _ = plt_utils.get_fig_and_axis(figsize=(10,5))
        _save_reg(fig,'test_utility_save_func')
        _save_clf(fig,'test_utility_save_func_clf')


