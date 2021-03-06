#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kaggle's Titanic Competition

https://www.kaggle.com/competitions/titanic

I actually did most of this as homework for the Machine Learning class from FSU's Engineering in 2021
"""

import os
import regex
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from pprint import pprint
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime as dt

# Script settings
get_ipython().run_line_magic('matplotlib', "")  # Not necessary in Jupyter Notebooks

# Script-wide variables
datasetsPath = "data"

# Load training Data
# SibSp = Siblings and spouse. It gives the number of people accompanying a passenger.
trainfname = "train.csv"
trainpath = os.path.join(datasetsPath, trainfname)
traindata0 = pd.read_csv(trainpath, index_col=0)

#
# Convert ALL categorical variables to quantitative values.
#
# OHE can't handle NaN. Only the `Age`, `Cabin`, and `Embarked` variables have `NaN`. Note their frequencies
print(traindata0.isna().sum())

#
# Missing value: `Cabin`
#
# Because Cabin is specific to each passenger, it does not have much predictive power for new cases, unless we were to extract the level from each value. E.g., C level for cabins C23, C25, and C27.


def extractCabinLevels(dataframe):
    cabins = dataframe["Cabin"].dropna()
    pattern = r"(\w)\d*"
    index = []
    levels = []
    for it, string in cabins.iteritems():
        cabins = regex.findall(pattern, string)
        index.append(it)
        # levels.append(', '.join(cabins))
        levels.append(set(cabins))
    levels = pd.Series(levels, index=index)
    return levels


alphabet2numbers = {alpha: numeric for (numeric, alpha) in enumerate(string.ascii_uppercase[:], start=1)}


def convertCabinLevels(cabinAsSet):
    """
    Uses `alphabet2numbers` defined outside function
    """
    nums = [alphabet2numbers[char] for char in cabinAsSet]
    return np.mean(nums)


traindata1 = traindata0.copy()
levels_train = extractCabinLevels(traindata1)
print(levels_train)

# We further investigate to see if there are instances of multi-level tickets.
pprint(sorted(levels_train.apply(lambda x: ', '.join(x)).unique()))

# After extracting the cabin levels we see that only three cases have multiple levels, these are passengers 76, 700, and 716. For this reason, and because the train and test set have different cabin levels, we will convert this variable into a quantitative one, assuming that the levels represent and ordinal variable.
levels_quant_train = levels_train.apply(convertCabinLevels)

# We replace the values thus:
traindata1.loc[levels_quant_train.index, "Cabin"] = levels_quant_train.values

# As for the missing values, note that the correlation of converted values is with `Fare` is negative. So A (and thus lower numeric values), are worth more. So we assume those not assigned Cabins are because they had lower value tickets. With this logic in mind we assign missing values a numeric value one integer above the maximum ordinal value.
max_ord_val = len(alphabet2numbers)+1  # 27
traindata1["Cabin"] = traindata1["Cabin"].fillna(max_ord_val)

#
# Missing value: `Embarked`
#
# Since there are only three values for `Embarked`, we can simply replace the missing values with a dummy value, the string `None`, to create a fourth category.
traindata1["Embarked"] = traindata1["Embarked"].fillna("Embarked_None")

#
# Missing value: `Age`
#
# Since the `Age` variable is quantitative, creating a dummy value would not be as useful in this case. Instead, we must find another substitute value. We would prefer to bootstrap the missing values, but will instead use the median age.
medianAge_train = traindata1["Age"].median()
traindata1["Age"] = traindata1["Age"].fillna(medianAge_train)

#
# Missing value: `Fare`
#
# The training set has no missing values for this variable, but the test set does.

# Next we encode the nominal values using the one-hot encoder from `sklearn`.
ohe = OneHotEncoder(sparse=False)
varsToEncode = ["Sex", "Embarked"]
encs_train = ohe.fit_transform(traindata1[varsToEncode])
columns1_train = [f"{var}_{category}" for var, array in zip(varsToEncode, ohe.categories_) for category in array]
encdf_train = pd.DataFrame(encs_train, index=traindata1.index, columns=columns1_train)

# Exclude the encoded columns
columns2_train = traindata1.columns.difference(columns1_train)
traindata = pd.concat([traindata1[columns2_train], encdf_train], axis=1)

#
# Select regressors
#
# The ticket values are alphanumeric and almost exactly unique to each passenger. We also drop `Name` for this reason.
xcolumns = traindata.columns.difference({"Survived",
                                         "Name",
                                         "Ticket"}.union(varsToEncode))

X_train0 = traindata[xcolumns]
y_train = traindata["Survived"]

# Standardize
scaler = StandardScaler().fit(X_train0)
X_train = pd.DataFrame(scaler.transform(X_train0), index=X_train0.index, columns=X_train0.columns)

# Train
model1 = LogisticRegression().fit(X_train, y_train)

# Predict
accuracy1a = model1.score(X_train, y_train)

y_hat_train1 = model1.predict(X_train)
accuracy1b = np.mean(y_hat_train1 == y_train)

# print(accuracy1a == accuracy1b)  # True

xcolsa = ", ".join(xcolumns[:-1])
xcolsb = xcolumns[-1]
print(f"We used sklearn's package to perform logistic regression on the variables {xcolsa}, and {xcolsb} to achieve a TRAINING accuracy of {accuracy1a:0.4f}")

########################################################################
### Test Data ##########################################################
########################################################################

testfname = "test.csv"
testpath = os.path.join(datasetsPath, testfname)
testdata0 = pd.read_csv(testpath, index_col=0)

# Replace missing values, convert ordinal values, and encode categorical values
testdata1 = testdata0.copy()
levels_test = extractCabinLevels(testdata1)
levels_quant_test = levels_test.apply(convertCabinLevels)

# Compare unique Cabin level values from train and test sets
pprint(sorted(pd.Series([",".join(s) for s in levels_test]).unique()))
pprint(sorted(pd.Series([",".join(s) for s in levels_train]).unique()))

# We replace the values thus:
testdata1.loc[levels_quant_test.index, "Cabin"] = levels_quant_test.values
testdata1["Cabin"] = testdata1["Cabin"].fillna(len(alphabet2numbers)+1)
testdata1["Embarked"] = testdata1["Embarked"].fillna("Embarked_None")

medianAge_test = testdata1["Age"].median()
testdata1["Age"] = testdata1["Age"].fillna(medianAge_test)

medianFare_test = testdata1["Fare"].median()
testdata1["Fare"] = testdata1["Fare"].fillna(medianFare_test)

# Encode nominal values
encs_test = ohe.transform(testdata1[varsToEncode])
columns1_test = [f"{var}_{category}" for var, array in zip(varsToEncode, ohe.categories_) for category in array]
encdf_test = pd.DataFrame(encs_test, index=testdata1.index, columns=columns1_test)

columns2_test = testdata1.columns.difference(varsToEncode)
testdata = pd.concat([testdata1[columns2_test], encdf_test], axis=1)

# Select regressors
X_test0 = testdata[xcolumns]

# Standardize
X_test = pd.DataFrame(scaler.transform(X_test0), index=X_test0.index, columns=X_test0.columns)

########################################################################
### Confusion Matrix ###################################################
########################################################################

conf_mat1 = pd.DataFrame(metrics.confusion_matrix(y_train, y_hat_train1), columns=["Pred. Positive", "Pred. Negative"], index=["True Positive", "True Negative"])
# TP | FN
# -- | --
# FP | TN

print(conf_mat1)

########################################################################
### ROC Curve ##########################################################
########################################################################

# The ROC curve is made from training set, because we need the true y values, which are not available for the test set.

y_score1 = model1.decision_function(X_train)
fpr1, tpr1, thresholds1 = metrics.roc_curve(y_train, y_score1)

roc_auc = metrics.auc(fpr1, tpr1)

fig1 = plt.figure()
fignum = fig1.number
lw = 2
plt.plot(fpr1, tpr1, color='darkorange',
         lw=lw, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic on training data')
plt.legend(loc="lower right")
plt.show()

########################################################################
### Random forest classification #######################################
########################################################################

model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
y_hat_train2 = model2.predict(X_train)

accuracy2 = np.mean(y_train == y_hat_train2)

## Confusion Matrix
conf_mat2 = pd.DataFrame(metrics.confusion_matrix(y_train, y_hat_train2), columns=["Pred. Positive", "Pred. Negative"], index=["True Positive", "True Negative"])
print(conf_mat2)

## ROC Curve
y_score2 = model2.predict_proba(X_train)[:,1]
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_train, y_score2)

roc_auc2 = metrics.auc(fpr2, tpr2)

fig2 = plt.figure()
fignum2 = fig2.number
lw = 2
plt.plot(fpr2, tpr2, color='darkorange',
         lw=lw, label=f'ROC curve (area = {roc_auc2:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic on training data (Random Forest)')
plt.legend(loc="lower right")
plt.show()

# Model1 has a AUC of 0.8599
# Model2 has a AUC of 0.9976

########################################################################
### Make Prediction ####################################################
########################################################################

# Because the Random Forest model has the highest ROC AUC, we choose it to make our predictions

# Predict response
y_hat_test2 = model2.predict(X_test)
submission = pd.Series(y_hat_test2, index=X_test.index, name=y_train.name)

timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
submissionPath = os.path.join(datasetsPath, f"submission_{timestamp}.csv")
submission.to_csv(submissionPath)

# Kaggle submission results:
# Rank: 11440
# Score (accuracy): 0.75837
# See `metaAnalysis.py` for discussion