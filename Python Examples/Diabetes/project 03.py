#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from IPython import get_ipython
from pathlib import Path

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
for el in xcolumns:
    print(el)

########################################################################
### Analysis ###########################################################
########################################################################

model = sm.Logit(yy, pd.DataFrame(xx, columns=xcolumns))
model1 = model.fit()
print(model1.summary())

#                            Logit Regression Results
# ==============================================================================
# Dep. Variable:                      y   No. Observations:                  520
# Model:                          Logit   Df Residuals:                      504
# Method:                           MLE   Df Model:                           15
# Date:                Wed, 09 Dec 2020   Pseudo R-squ.:                  0.7416
# Time:                        18:55:57   Log-Likelihood:                -89.520
# converged:                       True   LL-Null:                       -346.46
# Covariance Type:            nonrobust   LLR p-value:                6.502e-100
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Age                    0.0077      0.011      0.684      0.494      -0.014       0.030
# Gender                -3.9342      0.580     -6.778      0.000      -5.072      -2.797
# Polyuria               4.1756      0.685      6.093      0.000       2.833       5.519
# Polydipsia             4.7945      0.791      6.061      0.000       3.244       6.345
# sudden weight loss     0.2796      0.534      0.523      0.601      -0.767       1.327
# weakness               1.0201      0.522      1.955      0.051      -0.003       2.043
# Polyphagia             0.9188      0.488      1.884      0.060      -0.037       1.874
# Genital thrush         1.9635      0.559      3.510      0.000       0.867       3.060
# visual blurring        0.7881      0.605      1.303      0.193      -0.398       1.974
# Itching               -2.6443      0.635     -4.163      0.000      -3.889      -1.399
# Irritability           2.0675      0.577      3.582      0.000       0.936       3.199
# delayed healing       -0.5594      0.559     -1.001      0.317      -1.654       0.535
# partial paresis        1.1224      0.486      2.308      0.021       0.169       2.076
# muscle stiffness      -0.9027      0.537     -1.681      0.093      -1.955       0.150
# Alopecia              -0.3914      0.562     -0.697      0.486      -1.492       0.710
# Obesity               -0.2002      0.547     -0.366      0.714      -1.272       0.871
# ======================================================================================

# Remove insignificant variables
xcolumns = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'Genital thrush',
            'Itching', 'Irritability', 'partial paresis']
ycolumns = 'class'
xx = data[xcolumns].to_numpy()
yy = data[ycolumns].to_numpy()
model = sm.Logit(yy, pd.DataFrame(xx, columns=xcolumns))
model2 = model.fit()
print(model2.summary())

# Remove 'Age'
xcolumns = ['Gender', 'Polyuria', 'Polydipsia', 'Genital thrush',
            'Itching', 'Irritability', 'partial paresis']
ycolumns = 'class'
xx = data[xcolumns].to_numpy()
yy = data[ycolumns].to_numpy()
model = sm.Logit(yy, pd.DataFrame(xx, columns=xcolumns))
model3 = model.fit()
print(model3.summary())

#                            Logit Regression Results
# ==============================================================================
# Dep. Variable:                      y   No. Observations:                  520
# Model:                          Logit   Df Residuals:                      513
# Method:                           MLE   Df Model:                            6
# Date:                Wed, 09 Dec 2020   Pseudo R-squ.:                  0.7132
# Time:                        19:05:58   Log-Likelihood:                -99.360
# converged:                       True   LL-Null:                       -346.46
# Covariance Type:            nonrobust   LLR p-value:                1.486e-103
# ===================================================================================
#                       coef    std err          z      P>|z|      [0.025      0.975]
# -----------------------------------------------------------------------------------
# Gender             -3.3991      0.430     -7.903      0.000      -4.242      -2.556
# Polyuria            3.5924      0.503      7.144      0.000       2.607       4.578
# Polydipsia          4.6265      0.647      7.155      0.000       3.359       5.894
# Genital thrush      1.7880      0.488      3.666      0.000       0.832       2.744
# Itching            -1.9727      0.415     -4.751      0.000      -2.787      -1.159
# Irritability        2.1138      0.503      4.203      0.000       1.128       3.100
# partial paresis     1.5345      0.398      3.852      0.000       0.754       2.315
# ===================================================================================
