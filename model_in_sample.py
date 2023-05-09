import pandas as pd
import pandas_ta as ta
import numpy as np
import seaborn as sns
from scipy import linalg as la
import matplotlib.pyplot as plt


def pca_linear_model(x: pd.DataFrame, y: pd.Series, n_components: int, thresh: float= 0.01):
    # Center data at 0
    means = x.mean()
    x -= means
    x = x.dropna()

    # Find covariance and compute eigen vectors
    cov = np.cov(x, rowvar=False)
    evals , evecs = la.eigh(cov)
    # Sort eigenvectors by size of eigenvalue
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    # Create data set for model
    model_data = pd.DataFrame()
    for j in range(n_components):
         model_data['PC' + str(j)] = pd.Series( np.dot(x, evecs[j]) , index=x.index)
    
    cols = list(model_data.columns)
    model_data['target'] = y
    model_coefs = la.lstsq(model_data[cols], y)[0]
    model_data['pred'] = np.dot( model_data[cols], model_coefs)

    l_thresh = model_data['pred'].quantile(0.99)
    s_thresh = model_data['pred'].quantile(0.01)

    return model_coefs, means, l_thresh, s_thresh, model_data



data = pd.read_csv('BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')
data = data[data.index < '2020-01-01']

rsi_periods = list(range(2, 25))
rsis = pd.DataFrame()
for p in rsi_periods:
    rsis[p] = ta.rsi(data['close'], p)

target = np.log(data['close']).diff(6).shift(-6)

# Drop nans
rsis['tar'] = target
rsis = rsis.dropna()
target = rsis['tar']
rsis = rsis.drop('tar',axis=1)
coefs, means, l_thresh, s_thresh, model_data =  pca_linear_model(rsis, target, 3)

plt.style.use('dark_background')
model_data.plot.scatter('pred', 'target')
plt.axhline(0.0, color='white')
plt.show()



