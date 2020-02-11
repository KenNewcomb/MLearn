'''linear_regression.py: An implementation of regularized linear regression.'''
import numpy as np
from time import sleep

class linear_regression:

    def __init__(self):
        pass

    ## fit/predict f(X) ##

    def fit(self, X, y, optimizer='normal', epochs=100, alpha=0.1):
        # Initialize theta, X, y numpy arrays.
        self.theta = np.ones(len(X[0]) + 1)
        X = np.asarray(X)
        X = np.insert(X, 0, 1, axis=1)
        y = np.asarray(y)

        if optimizer == 'sgd':
            for epoch in range(0, epochs):
                sleep(1)
                dthetas, loss = self.sgd(X, y, alpha)
                print("Epoch: {}, Loss: {}".format(epoch, loss))
                print(self.theta)
                for theta in range(len(self.theta)):
                    self.theta[theta] += dthetas[theta]
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

    def sgd(self, X, y, alpha, regularizer='l2', lamb=0):
        '''Gradient descent algorithm.'''
        dthetas = [0 for i in range(len(X[0]))]
        m = len(X) # Number of training eXamples
        loss  = 0
        for t in range(0, len(dthetas)):
            grad_t = np.dot(self.h(X)-y, X[:, t])/m
            loss += sum(((self.h(X)-y)**2)/m)

            # Don't regularize the bias (theta0, when t==0), or when there is no regularizer selected or lambda=0.
            if not regularizer or t == 0 or lamb == 0:
                dthetas[t] = -grad_t*alpha
            elif regularizer in ['l2', 'ridge']:
                dthetas[t] = -alpha*(grad_t+lamb*self.theta[t])
                loss = loss+lamb*sum([i**2 for i in self.theta])
            elif regularizer in ['l1', 'lasso']:
                dthetas[t] = -alpha*(grad_t+lamb*(self.theta[t]/abs(self.theta[t])))
                loss = loss+lamb*sum([abs(i) for i in self.theta])
        return (dthetas, loss)

    def normal(self, X, y, regularizer='l2', lamb=0):
        """Solves the linear ordinary least squares problem by the "normal equation" method."""
        if not regularizer or lamb == 0:
            return np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
        elif regularizer in ['l2', 'ridge']:
            XtX = np.dot(np.transpose(X), X)
            zero_identity = np.identity(len(X[0]))
            zero_identity[0, 0] =  0
            return np.dot(np.dot(np.linalg.pinv(XtX + lamb*zero_identity), np.transpose(X)), y)
