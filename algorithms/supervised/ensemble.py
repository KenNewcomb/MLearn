from math import mode
class Voting:
    # TODO Describe what this class does

    def __init__(self, models):
        self.models = models

    def majority_voting(self, X):
        predictions = []
        for model in models:
            predictions.append(model.predict(X))
        majority = mode(predictions)
        return majority
