'''logistic_regression.py: An implementation of logistic regression with L2-regularization and gradient descent optimization.'''
from math import exp, log

class logistic_regression:

    def __init__(self):
        pass

    ## fit/predict f(x) ##

    def fit(self, x, y, epochs=100, alpha=0.1):
        # Initialize theta vector
        self.theta = [1 for i in range(len(x[0])+1)]

        for epoch in range(0, epochs):
            dthetas, loss = self.sgd(x, y, alpha)
            print("Epoch: {}, Loss: {}".format(epoch, loss))
            print(self.theta)
            for theta in range(len(self.theta)):
                self.theta[theta] += dthetas[theta]
        

    def predict(self, x):
        prob = self.h(x)
        print("Probability:", prob)
        if prob > 0.5:
            print("Predicted Class: 1")
        elif prob < 0.5:
            print("Predicted Class: 0")

    ## auxillary f(x) ##

    def h(self, x):
        activation = self.theta[0]
        for p in range(len(x)):
            activation += x[p]*self.theta[p+1]
        return self.logistic(activation)
            
    def logistic(self, x):
        return 1/(1+exp(-x))

    def sgd(self, x, y, alpha):
        '''Gradient descent algorithm.'''
        dthetas = [0 for i in range(len(x[0])+1)]
        for t in range(0, len(dthetas)):
            error = 0
            loss  = 0
            for d in range(len(x)):
                datax = x[d]
                datay = y[d]
                if t == 0:
                    error +=  (self.h(datax)-datay)
                elif t > 0:
                    error += (self.h(datax)-datay)*datax[t-1]
                try:
                    loss += datay*log(self.h(datax)) + (1-datay)*log(1-self.h(datax))
                except ValueError:
                    loss = 0
            dthetas[t] = -1*error*alpha*(1/len(x))
            loss /= -1*len(x)
        return (dthetas, loss)
             
