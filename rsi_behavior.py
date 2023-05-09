import pandas as pd
import pandas_ta as ta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')

rsi_periods = list(range(2, 25))

rsis = pd.DataFrame()
for p in rsi_periods:
    rsis[p] = ta.rsi(data['close'], p)

plt.style.use('dark_background')

rsis.hist(bins=100)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()

rsis.plot()
plt.show()


sns.heatmap(rsis.corr(), annot=True)
plt.xlabel("RSI Period")
plt.ylabel("RSI Period")

