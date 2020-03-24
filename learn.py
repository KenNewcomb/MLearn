'''learn.py: The driver for the MLearn project.'''
from algorithms.supervised.k_nearest_neighbors import regressor, classifier
from algorithms.supervised.logistic_regression import logistic_regression
from algorithms.supervised.linear_regression import linear_regression
from datasets.MNIST import MNIST
from algorithms.supervised.naive_bayes import naive_bayes
import matplotlib.pyplot as plt
from tools.visualization.plot import plot2D
from algorithms.supervised.svm import linear

linear_reg = linear_regression()
X = [(0, 1), (4, 0.5), (0, 2), (0, 3), (0, 4), (3, 0), (9, 0), (2, 0), (1, 0)]
y = (-1.2, 0.4, -1.2, -1.5, -4, 1, 1, -1, -1)
linear_reg.fit(X, y, optimizer='sgd', alpha=0.1, epochs=10000)
linear_reg.predict((0, 2))
