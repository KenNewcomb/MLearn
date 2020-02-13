'''naive_bayes.py: A class representing a naive-bayes classifier.'''
from scipy.stats import norm
import numpy as np

class naive_bayes():

    def __init__(self):
        self.priors = []
        self.pdfs   = []
        self.num_classes = 0

    def fit(self, X, y, variant='gaussian'):
        # Compute number of classes (Classes MUST be integers starting with 0!)
        self.num_classes = len(set(y))
        self.num_features = len(X[0])
        
        # Compute priors for each class.
        m = len(y)
        for c in range(self.num_classes):
            self.priors.append(y.count(c)/m)
            # Compute products of PDFs for each class.
            some_pdfs = []
            for i in range(self.num_features):
                if variant == 'gaussian':
                    xs = []
                    for m in range(len(X)):
                        if y[m] == c:
                            xs.append(X[i])
                    pdf = self.gaussian(xs)
                some_pdfs.append(pdf)
            self.pdfs.append(some_pdfs)
        
    def predict(self, x):
        for c in range(self.num_classes):
            # Predict each class probability
            total = 1
            class_pdfs = self.pdfs[c]
            for feature in range(len(class_pdfs)):
                total *= class_pdfs[feature].pdf((x[feature]))
            total *= self.priors[c]
            print(total)

    ## Auxillary f(x) ##
    def gaussian(self, x):
        mu = np.mean(x)
        sigma = np.std(x)
        dist = norm(mu, sigma)
        return dist
