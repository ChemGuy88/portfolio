#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
from IPython import get_ipython
from matplotlib import patches
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call

# Import work directory, user modules

from plots import plot_evaluation_metrics

########################################################################
### Script settings ####################################################
########################################################################

# IDE settings (Not necessary in Jupyter Notebooks)
plt.close('all')
get_ipython().run_line_magic('matplotlib', "")

figure_width = 18
figure_height = 12

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
featureNameLength = max([len(x) for x in xcolumns])
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
nameLength = max([len(modelName) for modelName in models.keys()])
assert np.sum([1 for el in modelObjs if el in models.keys()]) == len(modelObjs)
modelsInv = {value: key for key, value in models.items()}
modelObjsResults = {name: [] for name in models.keys()}
categoricalModels = ["Decision Tree", "Random Forest"]
nonCategoricalModels = set(models.keys()).difference(categoricalModels)
assert np.sum([1 for el in categoricalModels if el in models.keys()]) == len(categoricalModels)


# Analysis
if False:
    results1 = {modelName: [] for modelName in models.keys()}  # Accuracy results
    results2 = {modelName: [] for modelName in models.keys()}  # ROC results
    results3 = np.zeros((numSims, len(categoricalModels), xx.shape[1]))  # Feature importances
    i = -1
    resultsCols = ['Train', 'Test']
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting simulations at {timestamp}")
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
                results3[i, j, :] = clf.feature_importances_
            elif modelName in nonCategoricalModels:
                pass

            # Store model objs
            modelObjsResults[modelName].append(clf)

        if iterSim % numSimsMod == 0:
            timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = f"Running simulation {iterSim} at {timestamp}"
            print(text)

    # Save results
    rpath1 = "results1.pickle"
    rpath2 = "results2.pickle"
    rpath3 = "results3.pickle"
    mpath = f"models.pickle"
    li = [(rpath1, results1),
          (rpath2, results2),
          (rpath3, results3),
          (mpath, modelObjsResults)]
    for path, content in li:
        with open(path, "wb") as file:
            pickle.dump(content, file)

########################################################################
### Compute and print results from simulations #########################
########################################################################

rpath1 = f"results1.pickle"
rpath2 = f"results2.pickle"
rpath3 = "results3.pickle"
mpath = f"models.pickle"
results1 = pickle.load(open(rpath1, 'rb'))  # TODO Not sure if this usage of `open` closes the file object afterwards. If so, could change the save block above.
results2 = pickle.load(open(rpath2, 'rb'))
results3 = pickle.load(open(rpath3, 'rb'))
modelObjsResults = pickle.load(open(mpath, 'rb'))

columns = ['Train lb',
           'Train mean',
           'Train ub',
           'Test lb',
           'Test mean',
           'Test ub']
confIntCols = ['LB', 'Mean', 'UB']
summary1 = pd.DataFrame(columns=columns, index=models.keys(), dtype=float)
summary2 = pd.DataFrame(columns=columns, index=models.keys(), dtype=float)
dtypes = [("Lower Bound", float), ("Mean", float), ("Upper Bound", float), ("Feature", np.dtype(f"U{featureNameLength}")), ("Model Name", np.dtype(f"U{nameLength}"))]
summary3 = np.zeros((len(categoricalModels), xx.shape[1]), dtype=dtypes)

if True:
    j = -1
    for modelName, clf in modelObjs.items():

        # Analyze accuracy results
        accs = np.array(results1[modelName])
        mean1, lb1, ub1 = mean_confidence_interval(accs[:, 0])
        mean2, lb2, ub2 = mean_confidence_interval(accs[:, 1])
        summary1.loc[modelName, :] = [lb1, mean1, ub1, lb2, mean2, ub2]

        # Analyze ROC results
        rocs = np.array(results2[modelName])
        mean1, lb1, ub1 = mean_confidence_interval(rocs[:, 0])
        mean2, lb2, ub2 = mean_confidence_interval(rocs[:, 1])
        summary2.loc[modelName, :] = [lb1, mean1, ub1, lb2, mean2, ub2]

        # Analyze feature importances/coefficients
        if modelName in categoricalModels:
            j += 1
            means = results3[:, j, :].mean(axis=0)
            moe = st.sem(results3[:, j, :])
            lb = means - moe
            ub = means + moe
            summary3[j, :] = [(p, q, r, s, t) for p, q, r, s, t in zip(lb, means, ub, xcolumns, [modelName] * xx.shape[1])]
    summary3 = summary3.flatten()
    summary3.sort(order="Mean")
    summary3 = np.flip(summary3)
    sortOrder = []
    for el in summary3["Feature"]:
        if el not in sortOrder:
            sortOrder.append(el)

print(summary1.round(3))  # Accuracy
print(summary2.round(3))  # ROC
# Importances
s4columns = ["Lower Bound", "Mean", "Upper Bound"]
for featureName in sortOrder:
    print(f"Importance Confidence Interval for {featureName}:")
    ar1 = summary3[:][np.where(summary3[:]["Feature"] == featureName)]
    ar1.sort(order="Model Name")
    ar = ar1[s4columns]
    df = pd.DataFrame(ar,
                      index=ar1["Model Name"])
    print(df.round(3))

########################################################################
### Visualize Results: Accuracies & ROCs ###############################
########################################################################

metrics = ["Train", "Test"]
accFig1, accFig2 = plot_evaluation_metrics(models, metrics, "Accuracy", results1, numSims, figure_width, figure_height)
rocFig1, rocFig2 = plot_evaluation_metrics(models, metrics, "ROC", results2, numSims, figure_width, figure_height)

########################################################################
### Visualize Results: Feature Importances #############################
########################################################################

if False:
    # Broadcast numpy 3d array to pandas 2d data frame.
    results3_df = pd.DataFrame(results3.reshape((numSims * len(categoricalModels), xx.shape[1])), columns=xcolumns, index=range(1, numSims * len(categoricalModels)+1))
    li = []
    for modelName in categoricalModels:
        li.extend([modelName] * numSims)
    results3_df["Model Name"] = li

    # Arrange data for boxplots, per https://stackoverflow.com/questions/37191983/python-side-by-side-box-plots-on-same-figure
    data_to_plot = []
    labels = []
    box_positions = []
    tick_positions = []
    distance_between_boxplot_brothers = 0.6
    distance_between_boxplot_cousins = 1
    position = 0
    colormap = sns.color_palette("tab10")[:len(categoricalModels)]  # Or: plt.cm.get_cmap("hsv", range(N)), if N is large
    colors = []
    it = -1
    for featureName in xcolumns:
        position += distance_between_boxplot_cousins
        for modelName in categoricalModels:
            it += 1
            position += distance_between_boxplot_brothers
            mask = results3_df["Model Name"] == modelName
            data_to_plot.append(results3_df[mask][featureName])
            labels.append(modelName)
            box_positions.append(position)
            colors.append(colormap[it])
        it = -1
        tick_positions.append(np.mean(box_positions[-2:]))

    figure5 = plt.figure(figsize=(figure_width, figure_height))
    ax = figure5.add_axes([0.1, 0.1, 0.85, 0.85])
    boxplot = ax.boxplot(data_to_plot,
                        positions=box_positions,
                        patch_artist=True)

    for box, color in zip(boxplot["boxes"], colors):
        box.set_facecolor(color)

    ax.set_xlabel("Features", labelpad=10)
    ax.set_ylabel("Feature Importances (GINI importance)", labelpad=10)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(xcolumns)
    handles = [artist for artist in figure5.get_children()[1].get_children() if artist.__class__.__name__ == patches.PathPatch.__name__]
    ax.legend(handles[:len(categoricalModels)], categoricalModels)

    # Histograms to show distribution of feature importances. These may say more about the algorithm, than the actual medical question.
    numRows = 2
    numCols = 8 + 1
    figure6, axs = plt.subplots(numRows, numCols, sharey=True, tight_layout=True)
    axs = axs.flatten()
    handles = []
    it_features = -1
    axs_to_remove = []
    for it_plots in range(numRows * numCols):
        if (it_plots+1) % numCols == 0:
            axs_to_remove.append(it_plots)
        else:
            it_features += 1
            it_colors = 0
            featureName = xcolumns[it_features]
            for modelName in categoricalModels:
                mask = results3_df["Model Name"] == modelName
                handle = axs[it_plots].hist(results3_df[mask][featureName],
                                                alpha=0.5,
                                                color=colormap[it_colors])[-1]
                handles.append(handle)
                it_colors += 1

    for idx in axs_to_remove:
        axs[idx].remove()
    leg = figure6.legend(handles=handles[:len(categoricalModels)],
                labels=categoricalModels,
                loc="center right")
    figure6.set_figwidth(figure_width)
    figure6.set_figheight(figure_height)

########################################################################
### Visualize Decision Tree ############################################
########################################################################

if False:
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
