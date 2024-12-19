import unittest
import numpy as np
from linear_regression import linear_regression

class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        self.model = linear_regression()
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([5, 3, 4, 5])

    def test_fit_normal(self):
        self.model.fit(self.X, self.y, optimizer='normal', regularizer=None)
        self.assertIsNotNone(self.model.theta)
        self.assertEqual(len(self.model.theta), self.X.shape[1] + 1)

    def test_fit_sgd(self):
        self.model.fit(self.X, self.y, optimizer='sgd', epochs=80000, alpha=0.0005, regularizer=None)
        self.assertIsNotNone(self.model.theta)
        self.assertEqual(len(self.model.theta), self.X.shape[1] + 1)

if __name__ == '__main__':
    unittest.main()