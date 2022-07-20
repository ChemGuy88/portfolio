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