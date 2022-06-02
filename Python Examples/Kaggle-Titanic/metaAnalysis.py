"""
Meta-analysis of Kaggle Titanic Competition

Here we analyze some of the meta-information from the competition to contextualize our performance.

Of note are the following:
1. Honesty of the results (1.0 scores from outside sources)
2. Meaningfulness of score (compare to example submission which has similar score)
3. The current distribution of scores.
"""

from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Script settings
get_ipython().run_line_magic('matplotlib', "")  # Not necessary in Jupyter Notebooks
plt.close()

########################################################################
### 1. Honesty of results ##############################################
########################################################################

# Results with a perfect score of 1.0 are from information soruces outside the competition. Several entries show their code in the form of Kaggle Python notebooks. Most notable is this user's notebook: https://www.kaggle.com/vivovinco. His notebook is here: https://www.kaggle.com/code/vivovinco/titanic-real-1-0/notebook?scriptVersionId=86298162. The key takeaway is that those with a perfect score used a dataset similar to the one in this GitHub page: "https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv".

########################################################################
### 2. Meaningfulness of score #########################################
########################################################################

# Our score
myscore = 0.75837

###------------------------------------------------------------------###
### Compare to example ----------------------------------------------###
###------------------------------------------------------------------###

# Submitting the example given (`gender_submission.csv`) yields:
exscore = 0.76555

###------------------------------------------------------------------###
### Compare to competition distribution -----------------------------###
###------------------------------------------------------------------###

# Kaggle Competition Scores
data = pd.read_csv("titanic-publicleaderboard.csv")
mean = data["Score"].mean()
print(mean)
# 0.7596877810006815

myptile = np.percentile(data["Score"], myscore)
exptile = np.percentile(data["Score"], exscore)
stdev = data["Score"].std()
print(stdev)
# 0.07535

factor = 1000
datamin = data["Score"].min()
datamax = data["Score"].max()
binfactor = 10  # the factor by which to reduce binwidth. E.g.,  A value of 10 would make bins 1/10th of a stdev. A value of 100 gives ideal granularity.
# binWidth = int((1/stdev))  # This makes each bin one standard deviation wide.
binWidth = int((1/stdev)*binfactor)
binBoundaries = np.linspace(datamin*factor, datamax*factor, binWidth)

fig = plt.figure()
fignum = fig.number
hist = plt.hist(data["Score"]*factor, bins=binBoundaries)
# TODO Change xticks  # Script performs differently than interpreter. See https://stackoverflow.com/questions/41122923/getting-empty-tick-labels-before-showing-a-plot-in-matplotlib
locs, labels0 = plt.xticks()
# xlim = plt.xlim()
# labels1 = []
# print(labels0)
# for obj in labels0:
#     print(obj)
    # text0 = obj.get_text()
    # print(obj.get_text())
    # text1 = str(float(text0)/factor)
    # labels1.append(obj.set_text(text1))
# plt.xticks(locs, labels1)
# plt.xlim(xlim)
# >>> locs, labels = xticks()  # Get the current locations and labels.
# >>> xticks(np.arange(0, 1, step=0.2))  # Set label locations.
# >>> xticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
# >>> xticks([0, 1, 2], ['January', 'February', 'March'],
# ...        rotation=20)  # Set text labels and properties.
# >>> xticks([])  # Disable xticks.

# TODO
# Get score for top 1, .1, and .01%
# Plot myscore, exscore, meanscore, and top 1, .1, and .01% on histogram axis.
