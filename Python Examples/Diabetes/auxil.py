"""
Helper file
"""
import numpy as np
import matplotlib.pyplot as plt

no_samples = 8000
x = np.linspace(0, no_samples, 160)
no_labels = int(np.floor(len(x) / 20))
label = [f'{i * no_samples / no_labels:.0f}' for i in range(no_labels+1)]

fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

N = 100
data = np.random.normal(np.tile(100 / (x+1000), N), 0.001).reshape(N, -1)

flyprops = {'markersize':0.01}
colorprops = None
ax.boxplot(data, flierprops=flyprops, showcaps=False,
           boxprops=colorprops, whiskerprops={'color': 'tab:blue'},
           patch_artist=True)

ax.set_xlabel("Samples", labelpad=10)
ax.set_ylabel("Error (MSE)", labelpad=10)
ax.set_ylim(0, 0.11)
ax.set_xticks(range(0, len(x)+1, 20))
ax.set_xticklabels(label)

plt.show()