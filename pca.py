import pandas as pd
import pandas_ta as ta
import numpy as np
import seaborn as sns
from scipy import linalg as la
import matplotlib.pyplot as plt

data = pd.read_csv('BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')
data = data[data.index < '2020-01-01']

rsi_periods = list(range(2, 25))
rsis = pd.DataFrame()
for p in rsi_periods:
    rsis[p] = ta.rsi(data['close'], p)

rsi_means = rsis.mean()
rsis -= rsi_means
rsis = rsis.dropna()

# Find covariance and compute eigen vectors
cov = np.cov(rsis, rowvar=False)
evals , evecs = la.eigh(cov)
# Sort eigenvectors by size of eigenvalue
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]
evals = evals[idx]

n_components = 4
rsi_pca = pd.DataFrame()
for j in range(n_components):
    rsi_pca['PC' + str(j)] = pd.Series( np.dot(rsis, evecs[j]) , index=rsis.index)

plt.style.use('dark_background')
for j in range(n_components):
    pd.Series(evecs[j], index=rsi_periods).plot(label='PC' + str(j+1))
plt.xlabel("RSI Period")
plt.ylabel("Eigenvector Value")
plt.legend()
plt.show()




