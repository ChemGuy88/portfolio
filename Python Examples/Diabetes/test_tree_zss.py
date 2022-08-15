"""
Taken from https://github.com/timtadh/zhang-shasha
"""

from zss import simple_distance, Node
import pickle
import numpy as np
import pandas as pd


########################################################################
### Load data ##########################################################
########################################################################

fpath = f"diabetes_data_upload.csv"
data = pd.read_csv(fpath)

pos_label = ['Yes',
             'Positive',
             'Male']
neg_label = ['No',
             'Negative',
             'Female']
data = data.replace(to_replace=pos_label + neg_label, value=[1 for _ in pos_label] + [0 for _ in neg_label])

ageThreshold = 48  # Categorize age using `48` as the threshold, based on our exploratory data analysis
data["Age"] = (data["Age"] > ageThreshold) * 1

xcolumns = data.columns[data.columns != 'class']
featureNameLength = max([len(x) for x in xcolumns])
ycolumns = 'class'
xx = data[xcolumns].to_numpy()
yy = data[ycolumns].to_numpy()

########################################################################
###  ###################################################################
########################################################################

mpath = f"models.pickle"
modelObjsResults = pickle.load(open(mpath, 'rb'))
classifier = modelObjsResults["Decision Tree"][0]
tree = classifier.tree_

########################################################################
###  ###################################################################
########################################################################

class NoTreeMadeException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class TreeGraph(object):
    def __init__(self, classifier):
        self.classifier = classifier
        self.n_nodes = classifier.tree_.node_count
        self.children_left = classifier.tree_.children_left
        self.children_right = classifier.tree_.children_right
        self.feature = classifier.tree_.feature
        self.threshold = classifier.tree_.threshold

    def makeTree(self):
        n_nodes = self.n_nodes
        children_left = self.children_left
        children_right = self.children_right

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        self.node_depth = node_depth
        self.is_leaves = is_leaves
        return node_depth, is_leaves

    def print(self):
        n_nodes = self.n_nodes
        children_left = self.children_left
        children_right = self.children_right
        feature = self.feature
        threshold = self.threshold
        node_depth = self.node_depth
        is_leaves = self.is_leaves

        if not isinstance(node_depth, type(None)) and not isinstance(is_leaves, type(None)):
            pass
        else:
            raise(NoTreeMadeException("You tried to print a tree that has not been made. Try `TreeGraph.makeTree()` first."))
        for i in range(n_nodes):
            if is_leaves[i]:
                print(
                    "{space}node={node} is a leaf node.".format(
                        space=node_depth[i] * "\t", node=i
                    )
                )
            else:
                print(
                    "{space}node={node} is a split node: "
                    "go to node {left} if X[:, {feature}] <= {threshold} "
                    "else to node {right}.".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        left=children_left[i],
                        feature=feature[i],
                        threshold=threshold[i],
                        right=children_right[i],
                    )
                )

treeGraph = TreeGraph(classifier)

node_depth, is_leaves = treeGraph.makeTree()
treeGraph.print()