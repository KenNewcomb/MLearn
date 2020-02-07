'''decision_trees.py: A class representing a basic decision tree, trained with the CART algorithm.'''

class classifier():

    def __init__(self):
        pass

    def fit(self, x, y, maxdepth=4, impurity='gini'):
        numfeatures = len(x[0])
        depth = 0
        while depth <= maxdepth:
        # Search (feature, splitval) space for split that minimizes impurity.
            for feature in range(numfeatures):
                


        pass

    def predict(self, x):
        pass

    ## Helper f(x) ##

    def gini(self):
        pass
