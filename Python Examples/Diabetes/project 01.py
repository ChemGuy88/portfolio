#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from IPython import get_ipython
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import export_graphviz
from subprocess import call

########################################################################
### Code housekeeping ##################################################
########################################################################

plt.close('all')
userDir = str(Path.home())
workDir = f"{userDir}/Documents/instacon/project"
dataDir = f"{userDir}/Documents/instacon/project"
ipython = get_ipython()
ipython.magic('matplotlib')

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


fpath = f"{dataDir}/diabetes_data_upload.csv"
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

xcolumns = data.columns[data.columns != 'class']
ycolumns = 'class'
xx = data[xcolumns].to_numpy()
yy = data[ycolumns].to_numpy()

########################################################################
### Analysis ###########################################################
########################################################################

# Parameters
train_size = .30
test_size = 1 - train_size
numSims = 10000
numSimsMod = numSims / 10
random_state = None

models = {'Logistic Regression': LogisticRegression().__class__,
          'BernoulliNB': BernoulliNB().__class__,
          'Random Forest': RandomForestClassifier().__class__}
modelsInv = {value: key for key, value in models.items()}

# Analysis
if False:
    results1 = {modelName: [] for modelName in models.keys()}
    results2 = {modelName: [] for modelName in models.keys()}
    arr = np.zeros((3, 2))
    summary1 = pd.DataFrame(arr, columns=['Train', 'Test'], index=models.keys())
    print(f"Starting simulations at...\n{dt.datetime.now()}")
    for iterSim in range(1, numSims+1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for trainindices, testindices in sss.split(xx, yy):
            xtrain = xx[trainindices]
            ytrain = yy[trainindices]
            xtest = xx[testindices]
            ytest = yy[testindices]

        # Logistic Regression
        clf1 = LogisticRegression(max_iter=1000)
        clf1.fit(xtrain, ytrain)

        # Bernoulli Naive Bayes
        # Bernoulli NB is desgined for binary features. Get rid of age or classify it.
        clf3 = BernoulliNB()
        clf3.fit(xtrain, ytrain)

        # Random Forest Classification
        clf4 = RandomForestClassifier(max_depth=2)
        clf4.fit(xtrain, ytrain)

        # Evaluate models
        for clf in [clf1, clf3, clf4]:
            # Accuracy results
            trainacc = clf.score(xtrain, ytrain)
            testacc = clf.score(xtest, ytest)
            modelName = modelsInv[clf.__class__]
            results1[modelName].append([trainacc, testacc])

            # ROC results
            ytrainhat = clf.predict(xtrain)
            ytesthat = clf.predict(xtest)
            fprList1, tprList1, thresholds1 = roc_curve(ytrain, ytrainhat)
            fprList2, tprList2, thresholds2 = roc_curve(ytest, ytesthat)
            roc_auc1 = auc(fprList1, tprList1)
            roc_auc2 = auc(fprList2, tprList2)
            results2[modelName].append([roc_auc1, roc_auc2])

        if iterSim % numSimsMod == 0:
            text = f"Running simulation {iterSim}..."
            print(dt.datetime.now())
            print(text)

    # Save results
    rpath1 = f"{workDir}/pickles 01/results1.pickle"
    rpath2 = f"{workDir}/pickles 01/results2.pickle"
    mpath = f"{workDir}/pickles 01/models.pickle"
    pickle.dump(results1, open(rpath1, 'wb'))
    pickle.dump(results2, open(rpath2, 'wb'))
    pickle.dump([clf1, clf3, clf4], open(mpath, 'wb'))

########################################################################
### Compute and print results from simulations #########################
########################################################################

rpath1 = f"{workDir}/pickles 01/results1.pickle"
rpath2 = f"{workDir}/pickles 01/results2.pickle"
mpath = f"{workDir}/pickles 01/models.pickle"
results1 = pickle.load(open(rpath1, 'rb'))
results2 = pickle.load(open(rpath2, 'rb'))
clf1, clf3, clf4 = pickle.load(open(mpath, 'rb'))

columns = ['Train lb',
           'Train mean',
           'Train ub',
           'Test lb',
           'Test mean',
           'Test ub']
summary2 = pd.DataFrame(columns=columns, index=models.keys(), dtype=float)
summary3 = pd.DataFrame(columns=columns, index=models.keys(), dtype=float)
for clf in [clf1, clf3, clf4]:
    modelName = modelsInv[clf.__class__]

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

print(summary2.round(3))  # Accuracy
print(summary3.round(3))  # ROC

########################################################################
### Visualize Decision Tree ############################################
########################################################################

if False:
    for it, estimator in enumerate(clf4.estimators_[:5]):
        fpath1 = f"{workDir}/tree_{it}.dot"
        fpath2 = f"{workDir}/tree_{it}.png"
        export_graphviz(estimator, out_file=fpath1,
                        feature_names=xcolumns,
                        class_names=np.unique(yy).astype(str),
                        rounded=True, proportion=False,
                        precision=2, filled=True)
        call(['dot', '-Tpng', fpath1, '-o', fpath2, '-Gdpi=600'])
        if False:
            im = Image.open(fpath2)
            im.show()

    importances = pd.DataFrame(clf4.feature_importances, index=xcolumns)
    importances.sort_values(by=0, ascending=False, inplace=True)
