'''logistic_regression.py: An implementation of logistic regression with L2-regularization and gradient descent optimization.'''
from math import exp, log
from tqdm import tqdm

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

    def sgd(self, x, y, alpha, regularizer=None, lamb=0.1):
        '''Gradient descent algorithm.'''
        dthetas = [0 for i in range(len(x[0])+1)]
        m = len(x)
        for t in tqdm(range(0, len(dthetas))):
            grad_t = 0
            loss  = 0
            for d in range(m):
                datax = x[d]
                datay = y[d]
                if t == 0:
                    grad_t +=  (self.h(datax)-datay)
                elif t > 0:
                    grad_t += (self.h(datax)-datay)*datax[t-1]
                try:
                    loss += datay*log(self.h(datax)) + (1-datay)*log(1-self.h(datax))
                except ValueError:
                    loss = 0
            if not regularizer or t == 0 or lamb == 0:
                dthetas[t] = grad_t*alpha*(-1/m)
                loss /= m
            elif regularizer in ['l2', 'ridge']:
                dthetas[t] = -1*alpha*((1/m)*grad_t+(lamb/m)*self.theta[t])
                loss = (1/m)*(loss+lamb*sum([i**2 for i in self.theta]))
            elif regularizer in ['l1', 'lasso']:
                dthetas[t] = -1*alpha*((1/m)*grad_t+(lamb/m)*(self.theta[t]/abs(self.theta[t])))
                loss = (1/m)*(loss+lamb*sum([abs(i) for i in self.theta]))
        return (dthetas, loss)
             
