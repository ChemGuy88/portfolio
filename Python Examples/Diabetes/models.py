#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from IPython import get_ipython
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call

########################################################################
### Script settings ####################################################
########################################################################

# IDE settings (Not necessary in Jupyter Notebooks)
plt.close('all')
get_ipython().run_line_magic('matplotlib', "")

########################################################################
### Functions ##########################################################
########################################################################


def findBestThreshold(fprList, tprList, thresholds):
    '''
    '''
    bestDistance = 0
    for fpr, tpr, threshold in zip(fprList, tprList, thresholds):
        vec1 = [fpr, tpr]  # point on ROC curve
        vec2 = [fpr, fpr]  # point on diagonal line
        distance = np.sqrt(np.sum([(a - b) ** 2 for (a, b) in zip(vec1, vec2)]))
        if distance > bestDistance:
            bestDistance = distance
            bestThreshold = {'threshold': threshold,
                             'fpr': fpr,
                             'tpr': tpr}

    return bestThreshold


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

########################################################################
### Pre-processing #####################################################
########################################################################


fpath = f"diabetes_data_upload.csv"
# cols: Age	Gender	Polyuria	Polydipsia	sudden weight loss	weakness	Polyphagia	Genital thrush	visual blurring	Itching	Irritability	delayed healing	partial paresis	muscle stiffness	Alopecia	Obesity	class
data = pd.read_csv(fpath)

# Replace string categories to ordinal categories
pos_label = ['Yes',
             'Positive',
             'Male']
neg_label = ['No',
             'Negative',
             'Female']
data = data.replace(to_replace=pos_label + neg_label, value=[1 for _ in pos_label] + [0 for _ in neg_label])

# Categorize age using `48` as the threshold, based on our exploratory data analysis
ageThreshold = 48
data["Age"] = (data["Age"] > ageThreshold) * 1

xcolumns = data.columns[data.columns != 'class']
ycolumns = 'class'
xx = data[xcolumns].to_numpy()
yy = data[ycolumns].to_numpy()

########################################################################
### Analysis ###########################################################
########################################################################

# Parameters
train_size = .50
test_size = 1 - train_size
numSims = 1000
numSimsMod = numSims / 10
random_state = None

models = {'Logistic Regression': LogisticRegression().__class__,
          'Categorical NB': CategoricalNB().__class__,
          'BernoulliNB': BernoulliNB().__class__,
          'Decision Tree': DecisionTreeClassifier().__class__,
          'Random Forest': RandomForestClassifier().__class__}
modelObjs = {'Logistic Regression': LogisticRegression(max_iter=1000),
             'Categorical NB': CategoricalNB(),
             'BernoulliNB': BernoulliNB(),
             'Decision Tree': DecisionTreeClassifier(),
             'Random Forest': RandomForestClassifier()}
assert np.sum([1 for el in modelObjs if el in models.keys()]) == len(modelObjs)
modelsInv = {value: key for key, value in models.items()}
modelObjsResults = {name: [] for name in models.keys()}
categoricalModels = ["Decision Tree", "Random Forest"]
nonCategoricalModels = set(models.keys()).difference(categoricalModels)
assert np.sum([1 for el in categoricalModels if el in models.keys()]) == len(categoricalModels)


# Analysis
if True:
    results1 = {modelName: [] for modelName in models.keys()}
    results2 = {modelName: [] for modelName in models.keys()}
    results3 = {modelName: [] for modelName in models.keys() if modelName in categoricalModels}
    results4 = np.zeros((numSims, len(categoricalModels), xx.shape[1]))
    i = -1
    resultsCols = ['Train', 'Test']
    arr = np.zeros((len(models), len(resultsCols)))
    summary1 = pd.DataFrame(arr, columns=resultsCols,
                            index=models.keys())
    print(f"Starting simulations at {dt.datetime.now()}")
    for iterSim in range(1, numSims+1):
        i += 1
        j = -1
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=random_state)
        for trainindices, testindices in sss.split(xx, yy):
            xtrain = xx[trainindices]
            ytrain = yy[trainindices]
            xtest = xx[testindices]
            ytest = yy[testindices]

        # Evaluate models
        for modelName, clf in modelObjs.items():
            # Train
            clf.fit(xtrain, ytrain)

            # Accuracy results
            trainacc = clf.score(xtrain, ytrain)
            testacc = clf.score(xtest, ytest)
            results1[modelName].append([trainacc, testacc])

            # ROC results
            ytrainhat = clf.predict(xtrain)
            ytesthat = clf.predict(xtest)
            fprList1, tprList1, thresholds1 = roc_curve(ytrain, ytrainhat)
            fprList2, tprList2, thresholds2 = roc_curve(ytest, ytesthat)
            roc_auc1 = auc(fprList1, tprList1)
            roc_auc2 = auc(fprList2, tprList2)
            results2[modelName].append([roc_auc1, roc_auc2])

            # Feature Importances results
            if modelName in categoricalModels:
                j += 1
                results3[modelName].append(clf.feature_importances_)
                results4[i, j, :] = clf.feature_importances_
            elif modelName in nonCategoricalModels:
                pass

            # Store model objs
            modelObjsResults[modelName].append(clf)

        if iterSim % numSimsMod == 0:
            timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = f"Running simulation {iterSim} at {timestamp}"
            print(text)

    # Save results
    rpath1 = f"results1.pickle"
    rpath2 = f"results2.pickle"
    mpath = f"models.pickle"
    pickle.dump(results1, open(rpath1, 'wb'))
    pickle.dump(results2, open(rpath2, 'wb'))
    pickle.dump(modelObjsResults, open(mpath, 'wb'))

########################################################################
### Compute and print results from simulations #########################
########################################################################

rpath1 = f"results1.pickle"
rpath2 = f"results2.pickle"
mpath = f"models.pickle"
results1 = pickle.load(open(rpath1, 'rb'))
results2 = pickle.load(open(rpath2, 'rb'))
modelObjsResults = pickle.load(open(mpath, 'rb'))

columns = ['Train lb',
           'Train mean',
           'Train ub',
           'Test lb',
           'Test mean',
           'Test ub']
confIntCols = ['LB', 'Mean', 'UB']
summary2 = pd.DataFrame(columns=columns, index=models.keys(), dtype=float)
summary3 = pd.DataFrame(columns=columns, index=models.keys(), dtype=float)
summary4 = np.zeros((len(categoricalModels), len(confIntCols), xx.shape[1]))

if True:
    j = -1
    for modelName, clf in modelObjs.items():

        # Analyze accuracy results
        accs = np.array(results1[modelName])
        mean1, lb1, ub1 = mean_confidence_interval(accs[:, 0])
        mean2, lb2, ub2 = mean_confidence_interval(accs[:, 1])
        summary2.loc[modelName, :] = [lb1, mean1, ub1, lb2, mean2, ub2]

        # Analyze ROC results
        rocs = np.array(results2[modelName])
        mean1, lb1, ub1 = mean_confidence_interval(rocs[:, 0])
        mean2, lb2, ub2 = mean_confidence_interval(rocs[:, 1])
        summary3.loc[modelName, :] = [lb1, mean1, ub1, lb2, mean2, ub2]

        # Analyze feature importances/coefficients
        if modelName in categoricalModels:
            j += 1
            means = results4[:, j, :].mean(axis=0)
            moe = st.sem(results4[:, j, :])
            lb = means - moe
            ub = means + moe
            summary4[j, :, :] = pd.DataFrame([lb, means, ub],
                                             columns=xcolumns,
                                             index=confIntCols)
            # summary4.sort_values(by='Mean', ascending=False, inplace=True)

print(summary2.round(3))  # Accuracy
print(summary3.round(3))  # ROC
# Importances  # TODO Sort by mean
for k in range(xx.shape[1]):
    print(f"Importance Confidence Interval for {xcolumns[k]}:")
    print(summary4[:, :, k].round(3))

########################################################################
### Visualize Decision Tree ############################################
########################################################################

estimator = clf.estimators_[0]
fpath1 = f"tree.dot"
fpath2 = f"tree.png"
export_graphviz(estimator, out_file=fpath1,
                feature_names=xcolumns,
                class_names=np.unique(yy).astype(str),
                rounded=True, proportion=False,
                precision=2, filled=True)
call(['dot', '-Tpng', fpath1, '-o', fpath2, '-Gdpi=600'])
im = Image.open(fpath2)
im.show()
