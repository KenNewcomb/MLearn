'''learn.py: The driver for the KenLearn project.'''
from algorithms.supervised.k_nearest_neighbors import regressor

knn = regressor(1)
x = ((0, 1), (1,1), (0, 2))
y = (0, 1, 5)
knn.fit(x, y)
knn.predict((0, 5))
