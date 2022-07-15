"""
Helper file
"""
results4 = np.zeros((numSims, len(categoricalModels), xx.shape[1]))
j = -1
for modelName, simResults in modelObjsResults.items():
    i = -1
    if modelName in categoricalModels:
        j += 1
        for clf in simResults:
            i += 1
            results4[i, j, :] = clf.feature_importances_