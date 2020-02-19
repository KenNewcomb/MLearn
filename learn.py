'''learn.py: The driver for the MLearn project.'''
from algorithms.supervised.k_nearest_neighbors import regressor, classifier
from algorithms.supervised.logistic_regression import logistic_regression
from algorithms.supervised.linear_regression import linear_regression
from datasets.MNIST import MNIST
from algorithms.supervised.naive_bayes import naive_bayes
import matplotlib.pyplot as plt
from tools.visualization.plot import plot2D
from algorithms.supervised.svm import linear

#X, y = MNIST.load_data()
#knn = classifier(4)
#X = ((0, 1), (4, 1), (0, 2), (0, 3), (0, 4), (3, 0), (9, 0), (1, 2), (9,7))
#y = (0, 1, 0, 0, 0, 1, 1, 0, 1)
#knn.fit(X, y)
#knn.predict((4, 1))

linear_reg = linear_regression()
X = ((0, 1), (4, 0.5), (0, 2), (0, 3), (0, 4), (3, 0), (9, 0), (2, 0), (1, 0))
y = (-1, 1, -1, -1, -1, 1, 1, -1, -1)
linear_reg.fit(X, y, optimizer='normal', alpha=0.001, epochs=30000)
linear_reg.predict((0, 2))
#plot2D(X, y)
#import numpy as np
#X = np.asarray(X)
#y = np.asarray(y)
#print(X.shape)
#print(y.shape)
#log.fit(X, y, epochs=1000, alpha=0.01)
#log.predict((3,  0))


#s = linear()
#s.fit(X, y, epochs=10000, C=10000, alpha=0.00005)
#lin = linear_regression()
#X = ((0, 1, 1), (4, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (3, 0, 1), (9, 0, 1), (1, 2, 1), (9,7, 1))
#y = (0, 1, 0, 0, 0, 1, 1, 0, 1)
#lin.fit(X, y, optimizer='sgd')
#lin.predict((0, 1, 1))
