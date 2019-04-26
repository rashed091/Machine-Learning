from __future__ import print_function
import sys
import numpy as np
import pandas as pd


class TreeNode:
    def __init__(self, split_criterion, data, depth):
        self.split_criterion = split_criterion
        self.data = data
        self.depth = depth

        self.children = dict()
        self.split_feature = "feature"
        self.class_counts = data.ix[:,-1].value_counts()
        self.majority_class = self.data.ix[:,-1].value_counts().argmax()

        if ~(len(self.data.columns) == 1):
            self.split()

    def classify(self, data_point):
        # If the class is a leaf or if there is no child corresponding to
        # data_point's category within the split feature, return the classRu
        # with the highest count.
        # Otherwise, recursively call classify on the appropriate child
        if self.is_leaf():
            return self.majority_class

        split_feature_value = data_point[self.split_feature]

        if split_feature_value not in self.children:
            return self.majority_class

        return self.children[split_feature_value].classify(data_point)

    def is_leaf(self):
        return len(self.children) == 0

    def get_child(self, category):
        return self.children[category]

    # I use this function to print spaces corresponding to the depth of the
    # node before printing any info about the node
    def print_prefix(self):
        print(self.depth * '    ', end=' ')

    def print_node(self):
        # Print the split criterion and the counts for each class, then call
        # each child's print_node() method.
        self.print_prefix()
        print(self.split_criterion)

        for class_name in self.class_counts.index:
            count = self.class_counts.ix[class_name]
            self.print_prefix()
            print(class_name + " : " + str(count))

        for child in self.children.values():
            child.print_node()

    def split(self):
        # Calculate the information gain for each feature and keep track of the
        # feature with the highest gain. In case of ties, just pick one.
        max_IG = 0
        feature_data = self.data.drop(self.data.columns[-1], axis = 1)

        for feature in feature_data:
            information_gain = self.calculate_information_gain(feature, self.calculate_entropy(self.data.ix[:,-1]))
            if information_gain >= max_IG:
                max_IG = information_gain
                self.split_feature = feature
        ## if leaf, return
        if (self.calculate_entropy(self.data.ix[:,-1]) == 0 or len(self.data.columns) == 1):
            return

        # For each category within the feature with the highest information
        # gain, get the subset of the data that corresponds to that category,
        # drop the feature column, and create a child with the subset of data
        for category in self.data[self.split_feature].unique():
            category_data = self.data[self.data[self.split_feature] == category]
            child_data = category_data.drop(self.split_feature, axis = 1)
            child = TreeNode(self.split_feature + " = " + category, child_data, self.depth + 1)
            #add child to dictionary
            self.children[category] = child

    def calculate_entropy(self, class_column):
        class_counts = class_column.value_counts()
        entropy = 0
        total_count = len(class_column)

        for c in class_counts.index:
            count = class_counts.ix[c]
            ratio = float(count) / total_count
            entropy += ratio * np.log2(ratio)
        return -entropy

    def calculate_target_entropy(self, feature_name, class_column):
        col = self.data[feature_name]
        category_counts = col.value_counts()
        target_entropies = []

        for category in col.unique():
            count = 0
            entropy = 0
            class_counts = self.data[col == category][class_column].value_counts()
            for c in class_counts.index:
                ratio = float(class_counts[c]) / category_counts[count]
                entropy += (ratio * np.log2(ratio))
            count += 1
            target_entropies.append(-entropy)
        return target_entropies

    def calculate_information_gain(self, feature_name, overall_entropy):
        # Calculate the information gain for a feature. Takes the
        # overall_entropy as an additional argument to avoid re-calculating
        col = self.data[feature_name]
        category_counts = col.value_counts()
        total = len(col)

        target_entropies = []
        target_entropies = self.calculate_target_entropy(feature_name, self.data.columns[-1])

        weighted_sum = 0
        for t in range(len(target_entropies)):
            weighted_sum += category_counts[t] / total * target_entropies[t]

        return overall_entropy - weighted_sum



# Returns the data as a pandas data frame and the train ratio as a float.
# The program will exit on the following conditions:
# - There are too few command line arguments
# - The supplied CSV filename cannot be read by pandas
# - The supplied train ratio is not a float
# - The supplied train ratio is not in the interval (0.0, 1.0]
def get_data_and_train_ratio_from_args():
    if len(sys.argv) < 2:
        sys.exit('Usage: {0} <CSV file> [<train ratio>]'.format(sys.argv[0]))

    csv_filename = sys.argv[1]

    if len(sys.argv) >= 3:
        ratio_string = sys.argv[2]
    else:
         ratio_string = '0.75'

    try:
        data = pd.read_csv(csv_filename, dtype=str)
    except:
        sys.exit('pandas could not read {0} as a CSV file'.format(csv_filename))

    try:
        train_ratio = float(ratio_string)
    except ValueError:
        sys.exit('{0} is not a float'.format(ratio_string))

    if train_ratio <= 0.0 or train_ratio > 1.0:
        sys.exit('training ratio of {0} is out of bounds'.format(train_ratio))

    return data, train_ratio

# Split the data into train and test data frames.
def train_test_split_data_frame(data, train_ratio):
    # Generate an array of random floats on the interval [0.0, 1.0) with the
    # same number of values as there are rows of data
    rand = np.random.random(len(data))

    # Create a boolean array which will be used to select rows for the train
    # and test sets. mask[i] will be True when rand[i] < train_ratio and False
    # when rand[i] >= train_ratio
    mask = rand < train_ratio

    # data_train gets the rows from data which correspond to True values in mask
    data_train = data[mask]

    # ~mask inverts the mask so we can get the test data
    data_test = data[~mask]

    return data_train, data_test

# Return a data frame of zeros where the columns and indexes are the classes
# from data
def get_zeroed_confusion_matrix(data):
    classes = data.ix[:,-1].unique()
    return pd.DataFrame(0, index=classes, columns=classes)

data, train_ratio = get_data_and_train_ratio_from_args()
data_train, data_test = train_test_split_data_frame(data, train_ratio)

tree_node = TreeNode("root", data_train, 0)
tree_node.print_node()

con_matrix = get_zeroed_confusion_matrix(data)

for index, row in data.iterrows():
    target = row[-1]
    predicted = tree_node.classify(row)
    con_matrix.loc[target,predicted] += 1

print("Confusion Matrix: ")
print(con_matrix)
