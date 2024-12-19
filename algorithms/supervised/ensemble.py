from statistics import mode
class Voting:
    """
    A class that implements a voting ensemble method for combining predictions from multiple models.
    """

    def __init__(self, models):
        """
        Initialize the Voting class with a list of models.

        Parameters:
        models (list): A list of models that have a predict method.
        """
        self.models = models

    def majority_voting(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        if not predictions:
            raise ValueError("No predictions were made by the models.")
        majority = mode(predictions)
        return majority
