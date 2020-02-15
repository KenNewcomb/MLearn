'''svm.py: An implementation of the Support Vector Machine model.'''
import numpy as np
from time import sleep

class linear():

    def __init__(self):
        pass

    def fit(self, X, y, optimizer='sgd', epochs=10000, alpha=0.05):
        # Initialize theta, X, y numpy arrays.
        self.theta = np.ones(len(X[0]) + 1)
        X = np.asarray(X)
        X = np.insert(X, 0, 1, axis=1)
        y = np.asarray(y)

        if optimizer == 'sgd':
            for epoch in range(0, epochs):
                dthetas, loss = self.sgd(X, y, alpha)
                print("Epoch: {}, Loss: {}".format(epoch, loss))
                print(self.theta)
                for theta in range(len(self.theta)):
                    self.theta[theta] += dthetas[theta]

    def predict(self, X):
        X = np.asarray(X)
        X = np.insert(X, 0, 1)
        prediction = self.h(X)
        print("Prediction", prediction)

    ## auxillary f(X) ##

    def h(self, X):
        """Produces linear regression hypothesis function, h(X) = theta0*X0 + theta1*X1 + theta2*X2..."""
        return np.dot(X, self.theta)

    def sgd(self, X, y, alpha, C=0.1):
        '''Gradient descent algorithm.'''
        dthetas = [0 for i in range(len(X[0]))]
        m = len(X) # Number of training examples
        loss  = 0
        for t in range(0, len(dthetas)):
            grad_t = np.dot(self.h(X)-y, X[:, t])/m
            distance = 1 - (y * np.dot(X, self.theta))
            for ind, d in enumerate(distance):
                if max(0, d) == 0:
                    di = self.theta
                else:
                    di = self.theta - (C * y[ind] * X[ind])
                dthetas += di
            dthetas = dthetas/len(y)
            dthetas = dthetas*(-grad_t*alpha)
        # calculate hinge loss
        N = X.shape[0]
        distances = 1 - y * (np.dot(X, self.theta))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = C * (np.sum(distances) / N)
        loss = (1/2) * np.dot(self.theta, self.theta) + hinge_loss
        return (dthetas, loss)
