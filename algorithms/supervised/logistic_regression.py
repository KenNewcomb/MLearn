'''logistic_regression.py: An implementation of regularized logistic regression.'''
from math import exp, log
import numpy as np
from tqdm import tqdm
from time import sleep

class logistic_regression:

    def __init__(self):
        pass

    ## fit/predict f(x) ##

    def fit(self, X, y, epochs=100, alpha=0.1):
        # Initialize theta, X, y numpy arrays.
        self.theta = np.zeros(len(X[0])+1)
        X = np.asarray(X)
        X = np.insert(X, 0,  1, axis=1)
        y = np.asarray(y)

        for epoch in tqdm(range(0, epochs)):
            dthetas, loss = self.sgd(X, y, alpha)
            print("Epoch: {}, Loss: {}".format(epoch, loss))
            for theta in range(len(self.theta)):
                self.theta[theta] += dthetas[theta]
        

    def predict(self, X):
        X = np.asarray(X)
        X = np.insert(X, 0, 1)
        prob = self.h(X)
        print("Probability:", prob)
        if prob > 0.5:
            print("Predicted Class: 1")
        elif prob < 0.5:
            print("Predicted Class: 0")

    ## auxillary f(x) ##

    def h(self, X):
        """Produces logistic regression hypothesis function, h(X) = 1/(1+exp^(-(theta0*X0 + theta1*X1+ theta2*X2...)))"""
        activation = np.dot(X, self.theta)
        return self.logistic(activation)
            
    def logistic(self, x):
        return 1/(1+np.exp(-x))

    def sgd(self, X, y, alpha, regularizer=None, lamb=0):
        '''Gradient descent algorithm.'''
        dthetas = [0 for i in range(len(X[0]))]
        m = len(X)
        loss = 0
        for t in range(0, len(dthetas)):
            grad_t = np.dot(self.h(X)-y, X[:, t])/m
            loss  += (np.dot(-y,np.log(self.h(X))) + np.dot(-(1-y), np.log(1-self.h(X))))/m
            if not regularizer or t == 0 or lamb == 0:
                dthetas[t] = -grad_t*alpha
            elif regularizer in ['l2', 'ridge']:
                dthetas[t] = -1*alpha*(grad_t+lamb*self.theta[t])
                loss = loss+lamb*sum([i**2 for i in self.theta])
            elif regularizer in ['l1', 'lasso']:
                dthetas[t] = -alpha*(grad_t+lamb*(self.theta[t]/abs(self.theta[t])))
                loss = loss+lamb*sum([abs(i) for i in self.theta])
        return (dthetas, loss) 
