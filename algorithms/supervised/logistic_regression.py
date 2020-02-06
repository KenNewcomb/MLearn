'''logistic_regression.py: An implementation of logistic regression with L2-regularization and stochastic gradient descent optimization.'''

class logistic_regression:

    def __init__(self):
        pass

    ## fit/predict f(x) ##

    def fit(self, x, y, epochs=10):
        # Initialize theta vector
        self.theta = [0 for i in range(len(x[0]+1))]

        # Stochastic gradient descent
        
        #for point in zip(x, y):
        pass

    def predict(self, x):
        pass

    ## auxillary f(x) ##

    def h(self, x):
        activation = theta[0]
        for p in range(len(x)):
            activation += x[p]*theta[p+1]
        return logistic(activation)
            
    def logistic(self, x):
        return 1/(1+exp(-x))
