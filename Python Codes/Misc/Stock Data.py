import warnings

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas_datareader.data as web


style.use('ggplot')

start = dt.datetime(2020,1,1)
end = dt.date.today()

df = web.DataReader('^BVSP', 'yahoo', start, end)

print("Today:", end)
print(df.head())

days = 100

df['Moving Average'] = df['Adj Close'].rolling(window = days, min_periods = 0).mean()

ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['Moving Average'])
ax2.bar(df.index, df['Volume'])
plt.show()

