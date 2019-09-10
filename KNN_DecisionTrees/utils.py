from __future__ import division
import numpy as np
from typing import List
from hw1_knn import KNN


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    child_sum_list = []
    child_entropy_list = []
    for child in branches:
        child_sum = sum(child)
        child_entropy = 0
        child_sum_list.append(child_sum)
        if child_sum != 0:
            for val in child:
                if val != 0 :
                    child_entropy = child_entropy - ((val/child_sum)*np.log2(val/child_sum))
        child_entropy_list.append(child_entropy)
    entropy = 0
    total_sum = sum(child_sum_list)
    if total_sum == 0:
        return 0
    for child_sum, child_ent  in zip(child_sum_list, child_entropy_list):
        entropy = entropy - ((child_sum/total_sum) * child_ent)
    IG = S + entropy
    return IG


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    # raise NotImplementedError
    pass
    


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    num = sum_real = sum_predict = 0
    for x,y in zip(real_labels, predicted_labels):
        num = num + x*y
        sum_real = sum_real + x
        sum_predict = sum_predict + y
    score = 2*(float(num)/float(sum_real+sum_predict))
    return score
    
#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    sq = 0
    for x, y in zip(point1, point2):
        sq = sq + (x-y)**2
    distance = np.sqrt(sq)
    return float(distance)

#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    distance = 0
    for x, y in zip(point1, point2):
        distance = distance + x*y
    return float(distance)
    

#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    sq= 0
    for x,y in zip(point1, point2):
        sq = sq + (x-y)**2
    distance = -np.exp(-sq/2)
    return float(distance)


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    num= mod_x= mod_y = 0
    for x,y in zip(point1, point2):
        mod_x = mod_x + x**2
        mod_y = mod_y + y**2
        num = num + x*y
    distance = 1 - float(num)/float(np.sqrt(mod_x)*np.sqrt(mod_y))
    return float(distance)


g_score = 0
# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    global g_score
    N = len(Xtrain)
    if N < 30:
        upper = N-1
    else:
        upper = 30

    k = 1
    max_score = 0
    best_k = 0
    function_list = ['euclidean', 'gaussian', 'inner_prod', 'cosine_dist']

    for key, func in distance_funcs.items():
        for k in range(1,upper, 2):
        
            obj = KNN(k, func)
            obj.train(Xtrain, ytrain) # Training the system with training data
            predicted_labels = obj.predict(Xval) # Predicting labels for validation data
            score = f1_score(yval, predicted_labels)
            if score > max_score:
                max_score = score
                best_k = k
                best_func = key
                best_model = obj
            elif score == max_score:
                # Choose distance function on priority keys
                if max_score == 0 or function_list.index(key)<function_list.index(best_func):
                    max_score = score
                    best_k = k
                    best_func = key  
    g_score = max_score
    return best_model, best_k, best_func



# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    # raise NotImplementedError
    #print ("model_selection_with_transformation")
    global g_score
    max_score = 0
    for key, values in scaling_classes.items():
        scaler = values()
        train_features_scaled = scaler(Xtrain)
        val_features_scaled = scaler(Xval)
        model, k, function = model_selection_without_normalization(distance_funcs, train_features_scaled, ytrain, val_features_scaled, yval)
        score = g_score
        if score > max_score:
            max_score = score
            best_model = model
            best_k = k
            best_function = function
            best_scaler = key
    return best_model, best_k, best_function, best_scaler
   


class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        #raise NotImplementedError
        N = len(features)
        M = len(features[0])
        new_features = [[0] * M for _ in range(N)]
        for i in range(N):
            sq = 0
            for j in range(M):
                sq = sq + (features[i][j] ** 2)
            denominator = np.sqrt(sq)
            if denominator == 0:
                new_features[i] = features[i]
                continue
            for j in range(M):
                if features[i][j] == 0:
                    new_features[i][j] = 0
                else:
                    new_features[i][j] = features[i][j] / denominator
        print ("Features for Normalization ", features)
        print ("New features for Normalization ", new_features)
        return new_features


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    
    
    def __init__(self):
        self.callCounter = 0
        self.min_list = []
        self.max_list = []

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        # raise NotImplementedError
        N = len(features)
        M = len(features[0])
        new_features = [[0] * M for _ in range(N)]
        if self.callCounter == 0:
            print ("First call, find min and max values")
            for j in range(M):
                min = 99999
                max = -99999
                for i in range(N):
                    element = features[i][j]
                    if element > max:
                        max = element
                    if element < min:
                        min = element
                self.max_list.append(max)
                self.min_list.append(min)
            print ("max list is ", self.max_list)
            print ("min list is ", self.min_list)
            self.callCounter = 1
        else:
            print ("Min Max call for test data")
        for j in range(M):
            max = self.max_list[j]
            min = self.min_list[j]
            diff = max-min
            for i in range(N):
                if diff == 0:
                    new_features[i][j] = 0
                else:
                    new_features[i][j] = (features[i][j] - min)/diff
        return new_features

