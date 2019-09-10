from __future__ import division, print_function
from collections import Counter
from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:
    features = None
    labels = None

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.features = list(features)
        self.labels = list(labels)
        
        
    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        predicted_labels = []
        for point in features:
            k_labels = self.get_k_neighbors(point)
            most_common, num_most_common = Counter(k_labels).most_common(1)[0]
            predicted_labels.append(most_common)
            #predicted_labels.append(self.majority_element(k_labels))
        return predicted_labels
        

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        distance_list = []
        for i in range(len(self.features)):
            distance_list.append(( self.labels[i], self.distance_function(point, self.features[i]) ))
            distance_list.sort(key= lambda y:y[1])
        k_labels = []
        for i in range(self.k):
            k_labels.append(distance_list[i][0])
            
        return k_labels
        


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
