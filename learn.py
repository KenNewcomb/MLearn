'''learn.py: The driver for the KenLearn project.'''
from algorithms.supervised.k_nearest_neighbors import regressor, classifier
from algorithms.supervised.logistic_regression import logistic_regression

knn = classifier(4)
x = ((0, 1), (4, 1), (0, 2), (0, 3), (0, 4), (3, 0), (9, 0), (1, 2), (9,7))
y = (0, 1, 0, 0, 0, 1, 1, 0, 1)
knn.fit(x, y)
knn.predict((4, 1))

    log = logistic_regression()
x = ((0, 1), (4, 1), (0, 2), (0, 3), (0, 4), (3, 0), (9, 0))
y = (0, 1, 0, 0, 0, 1, 1)
log.fit(x, y, epochs=10000, alpha=1)
log.predict((4,  1))

