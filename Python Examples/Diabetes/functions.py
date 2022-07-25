"""
module for `models` script
"""

from matplotlib import patches
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_evaluation_metrics(models, metrics, evalName, results, numSims, figure_width, figure_height):
    # Organize data into pandas data frame
    index = pd.MultiIndex.from_product([results.keys(), metrics], names=["Model Name", evalName])
    results_df = pd.DataFrame(np.zeros((numSims, len(index))), columns=index)
    for modelName, li in results.items():
        ar = np.array(li)
        for it, label in enumerate(metrics):
            mask = results_df.columns==(modelName, label)
            results_df.iloc[:, mask] = ar[:, it]

    # Boxplots
    data_to_plot = []
    labels = []
    box_positions = []
    tick_positions = []
    distance_between_boxplot_brothers = 0.6
    distance_between_boxplot_cousins = 1
    position = 0
    modelNames = list(models.keys())  # modelNames
    colormap = sns.color_palette("tab10")[:len(metrics)]  # Or: plt.cm.get_cmap("hsv", range(N)), if N is large
    colors = []
    it = -1
    for modelName in modelNames:
        position += distance_between_boxplot_cousins
        labels.append(modelName)
        for metric in metrics:
            it += 1
            position += distance_between_boxplot_brothers
            data = results_df[(modelName, metric)]
            data_to_plot.append(data)
            box_positions.append(position)
            colors.append(colormap[it])
        it = -1
        tick_positions.append(np.mean(box_positions[-len(metrics):]))

    figure1 = plt.figure(figsize=(figure_width, figure_height))
    ax = figure1.add_axes([0.1, 0.1, 0.85, 0.85])
    boxplot = ax.boxplot(data_to_plot,
                        positions=box_positions,
                        patch_artist=True)

    for box, color in zip(boxplot["boxes"], colors):
        box.set_facecolor(color)

    ax.set_xlabel("Model Names", labelpad=10)
    ax.set_ylabel(evalName, labelpad=10)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(modelNames)
    handles = [artist for artist in figure1.get_children()[1].get_children() if artist.__class__.__name__ == patches.PathPatch.__name__]
    ax.legend(handles[:len(metrics)], metrics)

    # Histograms
    numRows = 1
    numCols = 5 + 1
    figure2, axs = plt.subplots(numRows, numCols, sharey=True, tight_layout=True)
    axs = axs.flatten()
    handles = []
    axs_to_remove = []
    for it_plots in range(numRows * numCols):
        if (it_plots+1) % numCols == 0:
            axs_to_remove.append(it_plots)
        else:
            modelName = modelNames[it_plots]
            it_colors = 0
            for metric in metrics:
                metric = metrics[it_colors]
                data = results_df[(modelName, metric)]
                handle = axs[it_plots].hist(data,
                                            alpha=0.5,
                                            color=colormap[it_colors])[-1]
                handles.append(handle)
                it_colors += 1
            axs[it_plots].set_xlabel(modelName)

    for idx in axs_to_remove:
        axs[idx].remove()
    leg = figure2.legend(handles=handles[:len(metrics)],
                labels=metrics,
                loc="center right")
    figure2.set_figwidth(figure_width)
    figure2.set_figheight(figure_height * .5)
    figure2.suptitle(f"Distribution of simulation {evalName} values")

    return figure1, figure2

def plot_feature_importances(results, numSims, categoricalModels, xx, xcolumns, figure_width, figure_height):
    # Broadcast numpy 3d array to pandas 2d data frame.
    results_df = pd.DataFrame(results.reshape((numSims * len(categoricalModels), xx.shape[1])), columns=xcolumns, index=range(1, numSims * len(categoricalModels)+1))
    li = []
    for modelName in categoricalModels:
        li.extend([modelName] * numSims)
    results_df["Model Name"] = li

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
            mask = results_df["Model Name"] == modelName
            data_to_plot.append(results_df[mask][featureName])
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
                mask = results_df["Model Name"] == modelName
                handle = axs[it_plots].hist(results_df[mask][featureName],
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

    return figure5, figure6


def get_tree_average(modelObjsResults, xcolumns):
    """
    Not finished

    Export trees for use with R
    Use R package "TreeDist" to get average of trees
    """
    models = modelObjsResults["Decision Tree"]
    tree = models[0].tree_
    node_sample_values = tree.value
    node_feature_split_criteria = tree.feature
    node_feature_split_criteria_as_name = [xcolumns[idx] for idx in node_feature_split_criteria]
    series = pd.Series(models)
    # trees = series.apply(lambda x: x.tree_)
    node_features = series.apply(lambda x: x.tree_.feature.tolist())
    for sim in modelObjsResults["Decision Tree"]:
        pass