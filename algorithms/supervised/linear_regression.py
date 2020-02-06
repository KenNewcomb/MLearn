'''linear_regression.py: An implementation of linear regression with L2-regularization and stochastic gradient descent optimization.'''

class linear_regression:

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
        prediction = self.h(x)
        print("Prediction", prediction)

    ## auxillary f(x) ##

    def h(self, x):
        activation = self.theta[0]
        for p in range(len(x)):
            activation += x[p]*self.theta[p+1]
        return activation
            
    def sgd(self, x, y, alpha):
        '''Batch stochastic gradient descent algorithm.'''
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
                    loss += (self.h(datax) - datay)**2
                except ValueError:
                    loss = 0
            dthetas[t] = -1*error*alpha*(1/len(x))
            loss /= len(x)
        return (dthetas, loss)
             
