'''logistic_regression.py: An implementation of regularized logistic regression.'''
from math import exp, log
import numpy as np
from tqdm import tqdm
from time import sleep
import sys

class logistic_regression:

    def __init__(self):
        pass

    def fit(self, X, y, epochs=100, alpha=0.1):
        # Initialize theta, X, y numpy arrays.
        self.theta = np.zeros(len(X[0])+1)
        X = np.asarray(X)
        X = np.insert(X, 0,  1, axis=1)
        y = np.asarray(y)

        for epoch in tqdm(range(0, epochs)):
            dthetas, loss = self.sgd(X, y)
            sys.stdout.write("\rEpoch: {}, Loss: {}, Thetas: {}".format(epoch, loss, self.theta))
            sys.stdout.flush()
            self.theta += alpha*dthetas

    def predict(self, X):
        X = np.asarray(X)
        X = np.insert(X, 0, 1, axis=1)  # Add bias term to X
        prob = self.h(X)
        print("Probability:", prob)
        if prob > 0.5:
            print("Predicted Class: 1")
        else:
            print("Predicted Class: 0")

    ## auxillary f(x) ##

    def h(self, X):
        """Produces logistic regression hypothesis function, h(X) = 1/(1+exp^(-(theta0*X0 + theta1*X1+ theta2*X2...)))"""
        activation = np.dot(X, self.theta)
        return self.logistic(activation)

    def logistic(self, x):
        return 1/(1+np.exp(-x))

    def sgd(self, X, y, regularizer=None, lamb=0):
        '''Gradient descent algorithm.'''
        dthetas = [0 for i in range(len(X[0]))]
        m = len(X)
        loss = 0
        grad = np.dot(self.h(X)-y, X)/m
        loss  += (np.dot(-y,np.log(self.h(X))) + np.dot(-(1-y), np.log(1-self.h(X))))/m
        dthetas = -grad
        if regularizer in ['l2', 'ridge'] and lamb != 0:
            dthetas += (lamb/m)*self.theta
            loss += (lamb/m)*sum([i**2 for i in self.theta[1:]])
        elif regularizer in ['l1', 'lasso'] and lamb != 0:
            dthetas += (lamb/m)*(self.theta/abs(self.theta))
            loss += (lamb/m)*sum([abs(i) for i in self.theta[1:]])
        dthetas[0] = -grad[0] # Don't regularize bias (theta[0])
        return (dthetas, loss)
