'''linear_regression.py: An implementation of regularized linear regression.'''
# TODO Write a better description of the class
import numpy as np
from time import sleep
import sys

class linear_regression:

    def fit(self, X, y, optimizer='normal', epochs=10000, alpha=0.05):
        # Initialize theta, X, y numpy arrays.
        self.theta = np.ones(len(X[0]) + 1)
        X = np.asarray(X)
        # Put a 1 as the first feature of each data point.
        X = np.insert(X, 0, 1, axis=1)
        y = np.asarray(y)

        # Two optimizers available.
        if optimizer == 'sgd':
            for epoch in range(0, epochs):
                #sleep(0.3)
                dthetas, loss = self.sgd(X, y)
                sys.stdout.write("\rEpoch: {}, Loss: {}, Thetas: {}".format(epoch, loss, self.theta))
                sys.stdout.flush()
                self.theta += alpha*dthetas
        elif optimizer == 'normal':
            self.theta = self.normal(X, y)

        print(self.theta)

    def predict(self, X):
        X = np.asarray(X)
        X = np.insert(X, 0, 1)
        prediction = self.h(X)
        print("Prediction", prediction)

    ## auxillary f(X) ##

    def h(self, X):
        """Produces linear regression hypothesis function, h(X) = theta0*X0 + theta1*X1 + theta2*X2..."""
        return np.dot(X, self.theta)

    def sgd(self, X, y, regularizer='l2', lamb=1):
        '''Gradient descent algorithm.'''
        dthetas = [0 for i in range(len(X[0]))]
        m = len(X) # Number of training eXamples
        loss  = 0
        error = self.h(X) - y
        grad = np.dot(error, X)/m
        dthetas = -grad
        loss += sum((error**2)/m)
        if regularizer in ['l2', 'ridge']:
            dthetas += (lamb/m)*self.theta
            loss += (lamb/m)*sum([i**2 for i in self.theta[1:]])
        elif regularizer in ['l1', 'lasso']:
            dthetas += (lamb/m)*(self.theta/abs(self.theta))
            loss += (lamb/m)*sum([abs(i) for i in self.theta[1:]])
        dthetas[0] = -grad[0] # Don't regularize bias (theta[0])
        return (dthetas, loss)

    def normal(self, X, y, regularizer='l2', lamb=1):
        """Solves the linear ordinary least squares problem by the normal equation."""
        # best fitting thetas = [Xt*X]^-1*Xt*y
        Xt = np.transpose(X)
        if not regularizer or lamb == 0:
            inverse_xtx = np.linalg.pinv(np.dot(Xt, X))
            return np.dot(np.dot(inverse_xtx, Xt), y)
        elif regularizer in ['l2', 'ridge']:
            # TODO Make this cleaner
            XtX = np.dot(np.transpose(X), X)
            zero_identity = np.identity(len(X[0]))
            zero_identity[0, 0] =  0
            return np.dot(np.dot(np.linalg.pinv(XtX + lamb*zero_identity), Xt), y)
