'''svm.py: An implementation of the Support Vector Machine model.'''
import numpy as np
from time import sleep
import sys

class linear():

    def __init__(self):
        pass

    def fit(self, X, y, optimizer='sgd', epochs=1000, alpha=0.001, C=1):
        # Initialize theta, X, y numpy arrays.
        self.theta = np.ones(len(X[0]) + 1)
        X = np.asarray(X)
        X = np.insert(X, 0, 1, axis=1)
        y = np.asarray(y)

        if optimizer == 'sgd':
            loss = 0
            for epoch in range(0, epochs):
                if epoch % 2000 == 0:
                    sys.stdout.write('Epoch: {0}, Loss: {1:.2f}, Theta: {2}\r'.format(epoch, loss, self.theta))
                    sys.stdout.flush()
                dthetas, loss = self.sgd(X, y, alpha, C)
                self.theta += dthetas
        print("Training complete.")

    def predict(self, X):
        X = np.asarray(X)
        X = np.insert(X, 0, 1)
        prediction = np.sign(np.dot(self.theta, X))
        print("Prediction", prediction)

    ## auxillary f(X) ##

    def sgd(self, X, y, alpha, C=100):
        '''Gradient descent algorithm.'''
        dthetas = [0 for i in range(len(X[0]))]
        m = len(X) # Number of training examples
        distance = 1 - (y * np.dot(X, self.theta))
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = self.theta
            else:
                di = self.theta - (C * y[ind] * X[ind])
            dthetas += di
        dthetas = (-alpha*dthetas)/m
        # calculate hinge loss
        N = X.shape[0]
        distance[distance < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = C * (np.sum(distance) / N)
        loss = (1/2) * np.dot(self.theta, self.theta) + hinge_loss
        return (dthetas, loss)
