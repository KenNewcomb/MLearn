# linear_regression.py: An implementation of regularized linear regression.
import numpy as np
from time import sleep
import sys

class linear_regression:
    """
    A class implementing regularized linear regression.

    Parameters:
        None

    Attributes:
        theta (numpy.ndarray): Coefficients of the linear regression model.

    Methods:
        fit(X, y, optimizer='normal', epochs=10000, alpha=0.05):
            Train the linear regression model using the provided data.

        predict(X):
            Make predictions using the trained model.

    """
    def fit(self, X, y, optimizer='normal', epochs=10000, alpha=0.05):
        """
        Train the linear regression model.

        Parameters:
            X (numpy.ndarray): Input features (training data).
            y (numpy.ndarray): Target values (labels).
            optimizer (str, optional): Optimization algorithm (default is 'normal').
            epochs (int, optional): Number of training epochs (default is 10000).
            alpha (float, optional): Learning rate (default is 0.05).

        Returns:
            None
        """

        self.theta = np.ones(len(X[0]) + 1)
        X = np.asarray(X)
        X = np.insert(X, 0, 1, axis=1)
        y = np.asarray(y)

        if optimizer == 'sgd':
            for epoch in range(0, epochs):
                dthetas, loss = self.sgd(X, y)
                sys.stdout.write("\rEpoch: {}, Loss: {}, Thetas: {}".format(epoch, loss, self.theta))
                sys.stdout.flush()
                self.theta += alpha * dthetas
        elif optimizer == 'normal':
            self.theta = self.normal(X, y)

        print("Parameters: ", self.theta)

    def predict(self, X):
        """
        Make predictions using the trained linear regression model.

        Parameters:
            X (numpy.ndarray): Input features for which predictions are made.

        Returns:
            None
        """
        X = np.asarray(X)
        X = np.insert(X, 0, 1)
        prediction = self.h(X)
        print("Prediction: ", prediction)

    def sgd(self, X, y, regularizer='l2', lamb=1):
        """
        Perform stochastic gradient descent optimization.

        Parameters:
            X (numpy.ndarray): Input features (training data).
            y (numpy.ndarray): Target values (labels).
            regularizer (str, optional): Regularization type ('l2' or None, default is 'l2').
            lamb (float, optional): Regularization parameter (default is 0).

        Returns:
            tuple: Gradient and loss.
        """
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
        """
        Solve the linear ordinary least squares problem using the normal equation.

        Parameters:
            X (numpy.ndarray): Input features (training data).
            y (numpy.ndarray): Target values (labels).
            regularizer (str, optional): Regularization type ('l2', 'ridge', or None, default is 'l2').
            lamb (float, optional): Regularization parameter (default is 1).

        Returns:
            numpy.ndarray: Coefficients of the linear regression model.
        """
        # Calculate the best-fitting thetas using the normal equation:
        # thetas = (Xt*X)^(-1) * Xt * y
        Xt = np.transpose(X)

        if not regularizer or lamb == 0:
            # If no regularization or lambda is 0, use the pseudo-inverse of Xt*X
            inverse_xtx = np.linalg.pinv(np.dot(Xt, X))
            return np.dot(np.dot(inverse_xtx, Xt), y)
        elif regularizer in ['l2', 'ridge']:
            # For L2 regularization, add the regularization term to Xt*X
            XtX = np.dot(np.transpose(X), X)
            zero_identity = np.identity(len(X[0]))
            zero_identity[0, 0] =  0
            # Use the pseudo-inverse of the modified XtX matrix
            return np.dot(np.dot(np.linalg.pinv(XtX + lamb*zero_identity), Xt), y)

    def h(self, X):
        """Produces linear regression hypothesis function, h(X) = theta0*X0 + theta1*X1 + theta2*X2..."""
        return np.dot(X, self.theta)
