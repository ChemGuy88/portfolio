"""
Basic stochastic model using Geometric Brownian Motion (GBM)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython

# Script settings
get_ipython().run_line_magic('matplotlib', "")  # Not necessary in Jupyter Notebooks

class GBM:
    def __init__(self, initial_price, drift, volatility, time_period, total_time):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.time_period = time_period
        self.total_time = total_time
        self.prices = []

    def simulate(self):
        while(self.total_time > 0):
            dS = self.current_price*self.drift*self.time_period + self.current_price*self.volatility*np.random.normal(0, np.sqrt(self.time_period))
            self.prices.append(self.current_price + dS)
            self.current_price += dS
            self.total_time -= self.time_period

simulations = []
n = 1000
initial_price = 500
drift = .24
volatility = .4
time_period = 1/365 # Daily
total_time = 1

for i in range(0, n):
    sim = GBM(initial_price, drift, volatility, time_period, total_time)
    sim.simulate()
    simulations.append(sim)

for sim in simulations:
    plt.plot(np.arange(0, len(sim.prices)), sim.prices, color="blue", alpha=0.1)
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

# Get confidence intervals
results = []
for sim in simulations:
    results.append(sim.prices)
df = pd.DataFrame(results)
lo = []
hi = []
lo = np.percentile(df, 5, axis=0)
hi = np.percentile(df, 95, axis=0)
median = np.percentile(df, 50, axis=0)

plt.plot(np.arange(0, len(sim.prices)), lo, color="red", label="95%")
plt.plot(np.arange(0, len(sim.prices)), hi, color="red", label="5%")
plt.plot(np.arange(0, len(sim.prices)), median, color="magenta", label="Median")
plt.plot(np.arange(0, len(sim.prices)), [initial_price] * len(sim.prices), color="black", label="Initial Price")
plt.legend()

figure = plt.gcf()
figure.set_size_inches((16, 8))