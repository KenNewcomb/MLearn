'''logistic_regression.py: An implementation of logistic regression with L2-regularization and stochastic gradient descent optimization.'''
from math import exp

class logistic_regression:

    def __init__(self):
        pass

    ## fit/predict f(x) ##

    def fit(self, x, y, epochs=10, alpha=0.01):
        # Initialize theta vector
        self.theta = [0 for i in range(len(x[0])+1)]

        for epoch in range(0, epochs):
            dthetas, loss = self.sgd(x, y, alpha)
            print("Epoch: {}, Loss: {}".format(epoch, loss))
        
        #for point in zip(x, y):
        pass

    def predict(self, x):
        pass

    ## auxillary f(x) ##

    def h(self, x):
        activation = self.theta[0]
        for p in range(len(x)):
            activation += x[p]*self.theta[p+1]
        return self.logistic(activation)
            
    def logistic(self, x):
        return 1/(1+exp(-x))

    def sgd(self, x, y, alpha):
        '''Stochastic gradient descent.'''
        dthetas = [0 for i in range(len(x[0])+1)]
        for t in range(len(dthetas)):
            error = 0
            for data in zip(x, y):
                print(data)
                error += (self.h(data[0])-data[1])*data[0][t-1]
            dthetas[t] = -1*error*alpha*(1/len(x))
        return (dthetas, 0)
             
