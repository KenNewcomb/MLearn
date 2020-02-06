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

        try:
            prediction = mode([i[1] for i in distances])
            print("Prediced class: {}".format(prediction))
            return prediction

        # If there is more than one mode
        except statistics.StatisticsError:
            print("Tie. Rerunning with k-1")
            self.k -= 1
            self.predict(x)

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
        print("Predicted class: {}".format(prediction))
        return prediction
