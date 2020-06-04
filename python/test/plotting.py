import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../src')
from pharmbio.cp import metrics,plotting

from numpy import genfromtxt
my_data = genfromtxt('resources/transporters.p-values.csv', delimiter=';', skip_header=1)
true_labels = np.array([1 if x == 1.0 else 0 for x in my_data[:,1]])
p_values=my_data[:,[2,3]]



fig1 = plotting.plot_label_distribution(true_labels, p_values=p_values)
m1 = metrics.calc_confusion_matrix( true_labels=true_labels, p_values=p_values, significance=0.15 )
m2 = metrics.calc_confusion_matrix( true_labels=true_labels, p_values=p_values, significance=0.7 )

fig2 = plotting.plot_confusion_matrix_bubbles(m1)
fig3 = plotting.plot_confusion_matrix_bubbles(m2)
plt.show()


