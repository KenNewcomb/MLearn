'''decision_trees.py: A class representing a basic decision tree, trained with the CART algorithm.'''

class classifier():

    def __init__(self):
        pass

    def fit(self, x, y, maxdepth=4, impurity='gini'):
        numfeatures = len(x[0])
        depth = 0
        while depth <= maxdepth:
            # 1) Generate splits (feature, splitval)


        pass

    def predict(self, x):
        pass

    ## Auxillary f(x) ##

    def gini(self):
        pass
