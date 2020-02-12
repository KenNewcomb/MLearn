'''naive_bayes.py: A class representing a naive-bayes classifier.'''
from scipy.stats import norm
import numpy as np

class classifier():

    def __init__(self):
        pass

    def fit(self, x, y, variant='gaussian'):
        # Instantiate numpy arrays.
        X = np.asarray(x)
        y = np.asarray(y)

        # Compute number of classes (Classes MUST be integers starting with 0!)
        num_classes = len(set(y[0]))
        num_features = len(X[0])
        
        # Compute priors for each class.
        priors = []
        m = len(y)
        for c in range(num_classes):
            priors.append(y.count(c)/m)
            # Compute products of PDFs for each class.
            product_of_pdfs = 1
            for i in num_features:
                if variant == 'gaussian':
                    xs = []
                    for m in range(len(x)):
                        if y[m] == c:
                            xs.append(x[i])
                        pdf = self.gaussian(xs)
                    product_of_pdfs *= pdf


        pass

    def predict(self, x):
        pass

    ## Auxillary f(x) ##
    def gaussian(self, x):
        mu = mean(x)
        sigma = std(x)
        dist = norm(mu, sigma)
        return dist
