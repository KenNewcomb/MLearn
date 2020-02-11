'''learn.py: The driver for the MLearn project.'''
from algorithms.supervised.k_nearest_neighbors import regressor, classifier
from algorithms.supervised.logistic_regression import logistic_regression
from algorithms.supervised.linear_regression import linear_regression
from datasets.MNIST import MNIST

X, y = MNIST.load_data()
#knn = classifier(4)
#X = ((0, 1), (4, 1), (0, 2), (0, 3), (0, 4), (3, 0), (9, 0), (1, 2), (9,7))
#y = (0, 1, 0, 0, 0, 1, 1, 0, 1)
#knn.fit(X, y)
#knn.predict((4, 1))

log = logistic_regression()
#X = ((0, 1), (4, 1), (0, 2), (0, 3), (0, 4), (3, 0), (9, 0), (2, 0), (1, 0))
#y = (0, 1, 0, 0, 0, 1, 1, 0, 0)
log.fit(X, y, epochs=1000, alpha=0.01)
#log.predict((3,  0))

lin = linear_regression()
X = ((0, 1, 1), (4, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (3, 0, 1), (9, 0, 1), (1, 2, 1), (9,7, 1))
y = (0, 1, 0, 0, 0, 1, 1, 0, 1)
X = ((0,), (1,), (2,), (3,))
y = (0, 1, 2, 4)
lin.fit(X, y, epochs=1000, alpha=0.1)
lin.predict(1.5)
