#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import auc, roc_curve
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
from IPython import get_ipython

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


########################################################################
### Pre-processing #####################################################
########################################################################

fpath = f"{dataDir}/diabetes_data_upload.csv"
# cols: Age	Gender	Polyuria	Polydipsia	sudden weight loss	weakness	Polyphagia	Genital thrush	visual blurring	Itching	Irritability	delayed healing	partial paresis	muscle stiffness	Alopecia	Obesity	class
data = pd.read_csv(fpath)

## Replace string categories to ordinal categories
pos_label = ['Yes',
             'Positive',
             'Male']
neg_label = ['No',
             'Negative',
             'Female']
data = data.replace(to_replace=pos_label + neg_label, value=[1 for _ in pos_label] + [0 for _ in neg_label])

columns = data.columns[data.columns != 'class']
xx = data[columns].to_numpy()
yy = data['class'].to_numpy()

########################################################################
### Train/test split ###################################################
########################################################################

# Parameters
train_size = .30
test_size = 1 - train_size
random_state = 0

# Split
# xtrain, xtest, ytrain, ytest = train_test_split(xx, yy, test_size=test_size, random_state=0)
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
for trainindices, testindices in sss.split(xx, yy):
    xtrain = xx[trainindices]
    ytrain = yy[trainindices]
    xtest = xx[testindices]
    ytest = yy[testindices]

print("""
########################################################################
### Logistic Regression ################################################
########################################################################
""")

clf = LogisticRegression(max_iter=1000).fit(xtrain, ytrain)

## Evaluate model
ytrainhat = clf.predict(xtrain)  # Done by clf.score()
ytesthat = clf.predict(xtest)
yscore = clf.predict_proba(xtest)
trainacc = clf.score(xtrain, ytrain)
testacc = clf.score(xtest, ytest)

# ROC
fprList1, tprList1, thresholds1 = roc_curve(ytrain, ytrainhat)
fprList2, tprList2, thresholds2 = roc_curve(ytest, ytesthat)
roc_auc1 = auc(fprList1, tprList1)
roc_auc2 = auc(fprList2, tprList2)

## Report
results = f"""
{clf.__class__}
Using only {train_size:0.0%} of our data to train our model we were able to achieve the following training and test accuracies:
"""

print(results)
columns1 = ['Train', 'Test']
index = ['Accuracy',
         'auc']
df = pd.DataFrame([[trainacc, testacc],
                   [roc_auc1, roc_auc2]], index=index, columns=columns1)
print(df.round(4))

coef = clf.coef_.ravel()
results = pd.DataFrame(coef, index=columns)
print(results)

print("""
########################################################################
### Naive Bayes (Categorical) ##########################################
########################################################################
""")

# Works for categorical data, like ours.

clf = CategoricalNB()
clf.fit(xtrain, ytrain)

## Evaluate model
ytrainhat = clf.predict(xtrain)  # Done by clf.score()
ytesthat = clf.predict(xtest)
yscore = clf.predict_proba(xtest)
trainacc = clf.score(xtrain, ytrain)
testacc = clf.score(xtest, ytest)

# ROC
fprList1, tprList1, thresholds1 = roc_curve(ytrain, ytrainhat)
fprList2, tprList2, thresholds2 = roc_curve(ytest, ytesthat)
roc_auc1 = auc(fprList1, tprList1)
roc_auc2 = auc(fprList2, tprList2)

## Report
results = f"""
{clf.__class__}
Using only {train_size:0.0%} of our data to train our model we were able to achieve the following training and test accuracies:
"""

print(results)
columns1 = ['Train', 'Test']
index = ['Accuracy',
         'auc']
df = pd.DataFrame([[trainacc, testacc],
                   [roc_auc1, roc_auc2]], index=index, columns=columns1)
print(df.round(4))

coef = clf.coef_
results = pd.DataFrame([x.ravel() for x in coef], index=range(1, len(columns)),
                       columns=range(1, 5))
print(results.round(4))

print("""
########################################################################
### Bernoulli Naive Bayes ##############################################
########################################################################
""")

# Bernoulli NB is desgined for binary features. Get rid of age or classify it.

clf = BernoulliNB()
clf.fit(xtrain, ytrain)

## Evaluate model
ytrainhat = clf.predict(xtrain)  # Done by clf.score()
ytesthat = clf.predict(xtest)
yscore = clf.predict_proba(xtest)
trainacc = clf.score(xtrain, ytrain)
testacc = clf.score(xtest, ytest)

# ROC
fprList1, tprList1, thresholds1 = roc_curve(ytrain, ytrainhat)
fprList2, tprList2, thresholds2 = roc_curve(ytest, ytesthat)
roc_auc1 = auc(fprList1, tprList1)
roc_auc2 = auc(fprList2, tprList2)

## Report
results = f"""
{clf.__class__}
Using only {train_size:0.0%} of our data to train our model we were able to achieve the following training and test accuracies:
"""

print(results)
columns1 = ['Train', 'Test']
index = ['Accuracy',
         'auc']
df = pd.DataFrame([[trainacc, testacc],
                   [roc_auc1, roc_auc2]], index=index, columns=columns1)
print(df.round(4))

coef = clf.coef_.ravel()
results = pd.DataFrame(coef, index=columns)
print(results)

print("""
########################################################################
### Random Forest Classification #######################################
########################################################################
""")

clf = RandomForestClassifier(max_depth=2).fit(xtrain, ytrain)

# Evaluate model
ytrainhat = clf.predict(xtrain)  # Done by clf.score()
ytesthat = clf.predict(xtest)
yscore = clf.predict_proba(xtest)
trainacc = clf.score(xtrain, ytrain)
testacc = clf.score(xtest, ytest)

# ROC
fprList1, tprList1, thresholds1 = roc_curve(ytrain, ytrainhat)
fprList2, tprList2, thresholds2 = roc_curve(ytest, ytesthat)
roc_auc1 = auc(fprList1, tprList1)
roc_auc2 = auc(fprList2, tprList2)

# Report
results = f"""
{clf.__class__}
Using only {train_size:0.0%} of our data to train our model we were able to achieve the following training and test accuracies:
"""
print(results)

columns1 = ['Train', 'Test']
index = ['Accuracy',
         'auc']
df = pd.DataFrame([[trainacc, testacc],
                   [roc_auc1, roc_auc2]], index=index, columns=columns1)
print(df.round(4))

importances = clf.feature_importances_  # This is the RF equivalent of model coefficients
results = pd.DataFrame(importances, index=columns)
print(results)

# Average train/test scores for 1000 simulations. Do another file with leaner code.
# Paint decision boundaries for best models. Decision boundaries will help us create a tool for patient self-evaluation
