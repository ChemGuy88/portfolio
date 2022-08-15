"""
https://biopython.org/wiki/Phylo
"""

from Bio import Phylo
from PIL import Image
from sklearn.tree import export_graphviz
from subprocess import call
from zss import simple_distance, Node
import dicttoxml
import json
import pickle
import numpy as np
import pandas as pd


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

mpath = f"models.pickle"
modelObjsResults = pickle.load(open(mpath, 'rb'))

trees = []
for idx in [0, 1]:
    estimator = modelObjsResults["Decision Tree"][idx]
    fpath1 = f"tree-{idx}.dot"
    fpath2 = f"tree-{idx}.JSON"
    export_graphviz(estimator, out_file=fpath1,
                    feature_names=xcolumns,
                    class_names=np.unique(yy).astype(str),
                    rounded=True, proportion=False,
                    precision=2, filled=True)
    dot_command = "C:\\Users\\autoreh\\Documents\\Graphviz\\bin\\dot"
    call([dot_command, '-Tjson', fpath1, '-o', fpath2])

    di = json.load(open(fpath2, "r"))
    xml = dicttoxml.dicttoxml(di).decode()
    fpath3 = f"tree-{idx}.xml"
    with open(fpath3, "w") as file:
        file.write(xml)
    tree = Phylo.parse(fpath3, "phyloxml")
    trees.append(tree)
    print(tree.name)  # TODO Doesn't work

# strict_tree = strict_consensus(trees)
# majority_tree = majority_consensus(trees, 0.5)
# adam_tree = adam_consensus(trees)
