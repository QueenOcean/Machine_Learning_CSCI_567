import numpy as np
import utils as Util
from collections import Counter


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    
    def modify_features(self):
        # convert into columns
        #print "Modifying features once"
        feat = np.array(self.features)
        col_features = []
        #print "Features-", self.features
        for i in range(len(self.features[0])):
            col_features.append(list(feat[:, i]))
        self.updated_features = list(col_features)

    def parent_entropy(self, counts):
        #print "Calculating parent entropy"
        s = sum(counts)
        parent_entropy = 0
        for i in list(counts):
            d = np.float(i/s)
            if d ==0 :
                break
            parent_entropy = parent_entropy - np.float(d * np.log2(d))
        return parent_entropy

    def create_branches(self):
        #print "Creating branches"
        branches = []
        # making branch structure
        for col in self.updated_features:
            if '#' not in col:
                unique_feature_value, counts = np.unique(col, return_counts=True)
                # print "feature_values are ", unique_feature_value, "counts - ", counts
                l = [[0] * self.num_cls for _ in range(len(unique_feature_value))]
                for val in range(len(col)):
                    ind1 = list(unique_feature_value).index(col[val])
                    ind2 = self.unique_labels.index(self.labels[val])
                    # print "ind1-",ind1," ind2-",ind2
                    l[ind1][ind2] = l[ind1][ind2] + 1
                # print "l is ",l
                branches.append(l)
            else:
                branches.append('#')
        # print " branches-", branches
        return branches

    def find_max_gain_ind(self, S, branches):
        #print "Calling IG to find_max_gain_ind"
        # Find index of feature with max gain
        max_gain = -999
        ind = 0
        max_ind = 0
        max_unique = 0
        for branch in branches:
            if -999 not in branch:
                gain = Util.Information_Gain(S, branch)
                # find feature with max gain
                if gain > max_gain:
                    #print "gain > max_gain for gain-",gain," ind -",ind," previous max_gain ",max_gain,"for ind-",max_ind
                    max_gain = gain
                    max_ind = ind
                    max_unique = len(branch)
                # find feature with more unique values and same gain -TIE BREAKER
                elif gain == max_gain:
                    unique = len(branch)
                    if unique > max_unique:
                        #print "Testing suzan", unique, "max_unique-",max_unique
                        #print "gain > max_gain for gain-", gain, " ind -", ind, " previous max_gain ", max_gain, "for ind-", max_ind
                        max_gain = gain
                        max_ind = ind
                        max_unique = unique
            ind = ind + 1
        #print "3. index of feature to split max_ind-",max_ind," max_gain-",max_gain," max_unique-",max_unique
        return max_ind

    def done_check(self):
        if all(all(item == -999 for item in items) for items in self.features):
            # self.labels = [max(self.labels)]
            self.splittable = False
            #print "Testing ", self.labels
            return True
        else:
            return False

    #TODO: try to split current node
    def split(self):
        if self.done_check():
            return

        value, counts = np.unique(self.labels, return_counts=True)
        self.unique_labels = list(value)
        #print "unique_labels - ", self.unique_labels, "counts -",counts
        #print

        # 0. modify features to col_features
        self.modify_features()
        #print "0. updated_features-", self.updated_features
        #print

        # 1. calculate parent_entropy S
        S = self.parent_entropy(counts)
        #print "1. Parent entropy is ", S
        #print

        # 2. create branches for IG
        branches = self.create_branches()
        #print "2. Branches in split- ", branches
        #print

        # 3. call IG function for all features
        max_ind = self.find_max_gain_ind(S, branches)
        # print "3. index of feature to split is- ", max_ind
        #print

        # 4. set dim_split, feature_uniq_split
        self.dim_split = max_ind
        self.feature_uniq_split = np.unique(self.updated_features[max_ind])
        #print "4. -"
        #print "Column to split - ", self.updated_features[max_ind]
        #print "dim_split index - ",self.dim_split
        #print "unique features of split - ", self.feature_uniq_split
        #print

        # 5. check if splittable - branch values does not have 0 for dim_split index
        # pure, mixed, self.splittable = self.is_splittable(branches)
        # print "5.-"
        # print "pure-", pure
        # print "mixed-", mixed
        # print "To split-", self.splittable
        # print

        for val in self.feature_uniq_split: # a,b,c
            new_features = []
            new_labels = []
            for x in range(len(self.features)):
                if self.features[x][self.dim_split] == val:
                    new_features.append(self.features[x][:])
                    new_labels.append(self.labels[x])
                    # print "features-", self.features
                    # print "labels-", self.labels
                    # print "new_feature-", new_features, " for ", val
                    # print "new_labels-", new_labels, " for ", val
            # Add '#' to split feature
            for x in range(len(new_features)):
                new_features[x][self.dim_split] = -999
            # print "hash_features with # - ", new_features
            #print
            #print "creating child_node and calling split for ", val
            num_cls = np.unique(new_labels).size
            child_node = TreeNode(new_features, new_labels, num_cls)
            self.children.append(child_node)
            #print "children list-", self.children
            # if self.all_done:
            #     break
            if child_node.splittable:
                #print val, "Splittable true calling split"
                child_node.split()
            else:
                continue
                #print val," is not splittable"
                #print "No more splittable-return"
                #print


    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if not self.splittable:
            most_common, num_most_common = Counter(self.labels).most_common(1)[0]  
            return most_common
        if feature[self.dim_split] in self.feature_uniq_split:
            branch_to_call = list(self.feature_uniq_split).index(feature[self.dim_split])
        else:
            most_common, num_most_common = Counter(self.labels).most_common(1)[0]  
            return most_common
        child = self.children[branch_to_call]
        predicted_label = child.predict(feature)
        return predicted_label
