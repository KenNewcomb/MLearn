# linear_regression.py: An implementation of regularized linear regression.
import numpy as np

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
    def fit(self, X, y, optimizer='normal', epochs=10000, alpha=0.05, regularizer=None, lamb=1):
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
        # Ensure X and y are numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Initialize theta
        self.theta = np.zeros(X.shape[1] + 1)

        # Add bias term to X
        X = np.insert(X, 0, 1, axis=1)

        if optimizer == 'sgd':
            for epoch in range(epochs):
                dthetas, loss = self.sgd(X, y, regularizer, lamb)
                self.theta += alpha * dthetas
        elif optimizer == 'normal':
            self.theta = self.normal(X, y, regularizer, lamb)
        else:
            raise ValueError("Invalid optimizer. Choose 'sgd' or 'normal'.")

        print("Parameters: ", self.theta)

    def predict(self, X):
        """
        Make predictions using the trained linear regression model.

        Parameters:
            X (numpy.ndarray): Input features for which predictions are made.

        Returns:
            numpy.ndarray: Predictions.
        """
        X = np.asarray(X)
        X = np.insert(X, 0, 1, axis=1)  # Add bias term to X
        prediction = self.h(X)
        return prediction

    def sgd(self, X, y, regularizer='None', lamb=0):
        """
        Perform stochastic gradient descent optimization.

        Parameters:
            X (numpy.ndarray): Input features (training data).
            y (numpy.ndarray): Target values (labels).
            regularizer (str, optional): Regularization type ('l2', 'ridge', 'l1', 'lasso' or None, default is 'l2').
            lamb (float, optional): Regularization parameter (default is 1).

        Returns:
            tuple: Gradient and loss.
        """
        m = len(X)  # Number of training examples
        error = self.h(X) - y  # Calculate error
        grad = np.dot(error, X) / m  # Calculate gradient
        dthetas = -grad  # Gradient descent step
        loss = np.sum((error**2) / m)  # Calculate loss

        # Apply regularization if specified
        if regularizer in ['l2', 'ridge'] and lamb != 0:
            dthetas += (lamb / m) * self.theta  # L2 regularization
            loss += (lamb / m) * np.sum(self.theta[1:]**2)  # Add regularization to loss
        elif regularizer in ['l1', 'lasso'] and lamb != 0:
            dthetas += (lamb / m) * np.sign(self.theta)  # L1 regularization
            loss += (lamb / m) * np.sum(np.abs(self.theta[1:]))  # Add regularization to loss

        dthetas[0] = -grad[0]  # Don't regularize bias (theta[0])
        return dthetas, loss

    def normal(self, X, y, regularizer='None', lamb=0):
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
