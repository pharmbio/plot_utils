from sklearn.datasets import load_iris
import numpy as np
from sklearn.svm import SVC

import sys
sys.path.append('/Users/staffan/git/peptid_studie/experiments/src') # Nonconformist

from nonconformist.cp import TcpClassifier
from nonconformist.nc import NcFactory


iris = load_iris()

idx = np.random.permutation(iris.target.size)

# Divide the data into training set and test set
idx_train, idx_test = idx[:100], idx[100:]

model = SVC(probability=True)	# Create the underlying model
nc = NcFactory.create_nc(model)	# Create a default nonconformity function
tcp = TcpClassifier(nc)			# Create a transductive conformal classifier

# Fit the TCP using the proper training set
tcp.fit(iris.data[idx_train, :], iris.target[idx_train])

# Produce predictions for the test set
predictions = tcp.predict(iris.data[idx_test, :])

# 
targets = np.array(iris.target[idx_test], copy=True)
targets.shape = (len(targets),1)
output = np.hstack((targets, predictions))

np.savetxt('resources/multiclass.csv', output, delimiter=',')