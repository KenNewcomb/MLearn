'''k_nearest_neighbors.py: A class representing the K-Nearest Neighbors (KNN) algorithm.'''
import math
import statistics
from statistics import mode, mean

class classifier():

    def __init__(self, k):
        self.k = k
        self.data = []

    def fit(self, x, y):
        for point in zip(x, y):
            self.data.append((point[0], point[1]))

    def predict(self, x):
        # Compute euclidean distance between input, x, and all data.
        distances = []
        for d in self.data:
            features = d[0]
            label   = d[1]
            distance = 0
            # d = (dx**2) + (dy**2) + (dz**2) + ...)**0.5
            for feature in range(len(features)):
                distance += (features[feature] - x[feature])**2
            distance = math.sqrt(distance)
            distances.append((distance, label))

        # Sort them by distance, get first k items.
        distances = sorted(distances, key=lambda x: x[0])[:self.k]
        print("Data points considered:")
        print("Distance\tClass")
        for point in distances:
            print("{0:.2f}      \t{1}".format(point[0], point[1]))

        # KNN --> prediction is the majority vote of k nearest neighbors.
        try:
            prediction = mode([i[1] for i in distances])
            print("Predicted class: {}".format(prediction))
            return prediction

        # If there is more than one mode, rerun with 1 less neighbor
        except statistics.StatisticsError:
            if self.k > 1:
                print("Tie. Rerunning with k-1...")
                self.k -= 1
                return self.predict(x)
            else:
                print("Tie. Unable to resolve with k=1.")
                return None
            

class regressor():

    def __init__(self, k):
        self.k = k
        self.data = []

    def fit(self, x, y):
        for point in zip(x, y):
            self.data.append((point[0], point[1]))

    def predict(self, x):
        # Compute distances from all data in dataset.
        distances = []
        for d in self.data:
            features = d[0]
            label   = d[1]
            distance = 0
            for feature in range(len(features)):
                distance += (features[feature] - x[feature])**2
            distance = math.sqrt(distance)
            distances.append((distance, label))

        # Sort them by distance, get first k
        distances = sorted(distances, key=lambda x: x[0])[:self.k]
        print("Data points considered:")
        print("Distance\tClass")
        for point in distances:
            print("{0:.2f}      \t{1}".format(point[0], point[1]))

        prediction = mean([i[1] for i in distances])
        print("Prediction: {}".format(prediction))
        return prediction
