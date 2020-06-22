# !/usr/bin/env python3
# Author: Erik Davino Vincent

import warnings

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas_datareader.data as web

# Creates moving average of primaryName, if it isn't in secondaryPlot
def movingAverage(dataFrame, primaryName, timePeriod = 15):

    # Create new data frame with new column
    newFrame = dataFrame.copy(deep = True)
    newFrame['Moving Average ' + primaryName] = newFrame[primaryName].rolling(window = timePeriod, min_periods = 0).mean()
    return newFrame

<<<<<<< HEAD
def candleStick(dataFrame, primaryName):
    
    candleFrame = dataFrame[primaryName].resample('1D').ohlc()
    candleFrame.reset_index(inplace = True)
    candleFrame['Date'] = candleFrame['Date'].map(mdates.date2num)

    return candleFrame

# Plot "back-end"
def plotGeneral(dataFrame, primaryName, subplot = [], figsize = (14.4/1.5, 9.6/1.5)):
=======
def plotGraphics(dataFrame, primaryName, subplot = [], figsize = (14.4/1.5, 9.6/1.5)):
>>>>>>> parent of 7baa6db... plotGraphics now accepts different plot styles
    # Creates a figure
    plt.figure(num = primaryName, figsize = figsize)
    
    secondaryPlot = [
        'Volume'
        ]

    # Finds if a second plot window will be needed
    secPlot_key = 0
    for plotName in subplot:
        if plotName in secondaryPlot:
            secPlot_key = 1
            break

    # If there is any subplot and one more window will be needed:
    if len(subplot) > 0 and secPlot_key:
        secPlot_key = 0

        # Prepares primaryName plot
        ax1 = plt.subplot2grid((7,1), (0,0), rowspan = 5, colspan = 1)
<<<<<<< HEAD

        # Plot primaryName with the mode
        if primaryMode == 'std':
            ax1.plot(dataFrame.index, dataFrame[primaryName], label = primaryName)
        elif primaryMode == 'bar':
            ax1.bar(dataFrame.index, dataFrame[primaryName], label = primaryName)
        elif primaryMode == 'candlestick':
            CS = candleStick(dataFrame, primaryName)
            ax1.xaxis_date()
            candlestick_ohlc(ax1, CS.values, width = 2, colorup = 'g')
            
=======
        ax1.plot(dataFrame.index, dataFrame[primaryName], label = primaryName)
>>>>>>> parent of 7baa6db... plotGraphics now accepts different plot styles
        for plotName in subplot:
            # If the secondary plot needs a second window, creates a second window +
            # plots the graphic. Can only have one more window; greedy algorithm (for now).
            if plotName in secondaryPlot and not secPlot_key:
                ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 2, colspan = 1)
                ax2.plot(dataFrame.index, dataFrame[plotName], label = plotName)
                ax2.legend()
                secPlot_key = 1
            # Plots all secondary plots that don't need a second window on main window.
            elif not plotName in secondaryPlot:
                ax1.plot(dataFrame.index, dataFrame[plotName], label = plotName)
        ax1.legend()

    # If more windows are not needed but there are secondary plots:
    elif len(subplot) > 0 and not secPlot_key:
<<<<<<< HEAD
        
        if primaryMode == 'std':
            plt.plot(dataFrame.index, dataFrame[primaryName], label = primaryName)
        elif primaryMode == 'bar':
            plt.bar(dataFrame.index, dataFrame[primaryName], label = primaryName)
        elif primaryMode == 'candlestick':
            CS = candleStick(dataFrame, primaryName)
            plt.xaxis_date()
            candlestick_ohlc(plt, CS.values, width = 2, colorup = 'g')
            
=======
        plt.plot(dataFrame.index, dataFrame[primaryName], label = primaryName)
>>>>>>> parent of 7baa6db... plotGraphics now accepts different plot styles
        for plotName in subplot:
            plt.plot(dataFrame.index, dataFrame[plotName], label = plotName)
        plt.legend()
    # Else, plots primaryName only, even if it is in secondaryPlot
    else:
<<<<<<< HEAD
        
        if primaryMode == 'std':
            plt.plot(dataFrame.index, dataFrame[primaryName], label = primaryName)
        elif primaryMode == 'bar':
            plt.bar(dataFrame.index, dataFrame[primaryName], label = primaryName)
        elif primaryMode == 'candlestick':
            CS = candleStick(dataFrame, primaryName)
            plt.xaxis_date()
            candlestick_ohlc(plt, CS.values, width = 2, colorup = 'g')
            
=======
        plt.plot(dataFrame.index, dataFrame[primaryName], label = primaryName)
>>>>>>> parent of 7baa6db... plotGraphics now accepts different plot styles
        plt.legend()

    plt.show()
    return True

def main():

    # Styling
    style.use('ggplot')

    # Period 
    start = dt.datetime(2020,1,1)
    end = dt.date.today()

    # Stock data
    df = web.DataReader('^BVSP', 'yahoo', start, end)

    # Header
    print("Today:", end)
    print(df.head())
    print()
    input('>>')

    # Creates moving average
    days = 15
    df = movingAverage(df, 'Adj Close', days)
    
    # Plots graphic
<<<<<<< HEAD
    plotGeneral(df, 'Adj Close, candlestick', [])
=======
    plotGraphics(df, 'Adj Close', ['Moving Average Adj Close', 'High'])
>>>>>>> parent of 7baa6db... plotGraphics now accepts different plot styles

main()
