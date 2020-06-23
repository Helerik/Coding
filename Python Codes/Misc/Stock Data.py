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

# Creates moving average of primaryName
def movingAverage(dataFrame, primaryName, timePeriod = 15):

    # Create new data frame with new column
    newFrame = dataFrame.copy(deep = True)
    newFrame['Moving Average ' + primaryName + ' ' + str(timePeriod)] = newFrame[primaryName].rolling(
        window = timePeriod,min_periods = 0
        ).mean()
    return newFrame

# Plot one name from the dataframe
def plotTheName(dataFrame, theName, thePlot = None, theMode = 'Std'):

    if theName == 'Candle' or theName == 'cStick':
        ohlc = dataFrame[['Open', 'High', 'Low', 'Close']].copy()
        ohlc.reset_index(inplace = True)
        ohlc['Date'] = mdates.date2num(ohlc['Date'].values)
        thePlot.xaxis_date()
        candlestick_ohlc(thePlot, ohlc.values, width = 0.9, colorup='#77d879', colordown='#db3f3f') 
    elif theMode == 'Std':
        thePlot.plot(dataFrame.index, dataFrame[theName], label = theName)
    elif theMode == 'Bar':
        thePlot.bar(dataFrame.index, dataFrame[theName], label = theName)

# Plot "back-end"
def plotGraphics(dataFrame, primaryName, subplot = [], figsize = (14.4/1.5, 9.6/1.5)):

    # List of plots that go on a second window
    secondaryPlot = [
        'Volume'
        ]

    # Divides primaryName and primaryMode
    primLis = primaryName.split(', ', 1)
    primaryName = primLis[0]
    if len(primLis) > 1:
        primaryMode = primLis[1]
    else:
        primaryMode = 'Std'
    
    # Finds if a second plot window will be needed and divides name and mode
    secPlot_key = 0
    for i in range(len(subplot)):
        plotName = subplot[i]
        nameLis = plotName.split(', ', 1)
        if len(nameLis) > 1:
            plotName = nameLis
        else:
            plotName = [nameLis[0], 'Std']
        if plotName[0] in secondaryPlot:
            secPlot_key = 1
        subplot[i] = plotName

    # Creates a figure and axis
    if secPlot_key:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (7.5, 5),
                                       gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, sharex = True, figsize = (7.5, 5))

    # Formats date for better vizualization
    fig.autofmt_xdate()
    
    # Window will not be resizable
    manager = plt.get_current_fig_manager()
    manager.window.resizable(False, False)

    # Plot graphics
    plotTheName(dataFrame, primaryName, ax1, primaryMode)
    for plotName in subplot:
        if plotName[0] in secondaryPlot:
            plotTheName(dataFrame, plotName[0], ax2, plotName[1])
            ax2.legend()
        else:
            plotTheName(dataFrame, plotName[0], ax1, plotName[1])
            ax1.legend()

    if primaryName == 'Candle' or primaryName == 'cStick':
        pass
    else:
        ax1.legend()

    ax1.set_title(primaryName)

    plt.show(block = False)
    return True

def main():

    # Styling
    style.use('ggplot')

    # Period 
    start = dt.datetime(2020,1,1)
    end = dt.date.today()

    # Creates moving average
##    days = 3
##    df = movingAverage(df, 'Adj Close', days)

    toPlotList = []
    while True: 
        # Choose a symbol to visualize
        symb = input("Choose a symbol to plot: ")
        df = web.DataReader(symb, 'yahoo', start, end)
        print()
        choice = input("Add more information [y/n]: ").lower()
        print()
        if choice == 'y' or choice == "yes":
            while True:
                secSymb = input("Plot secondary data: ")
                print()
                if secSymb == '':
                    break
                if not secSymb in toPlotList:
                    toPlotList.append(secSymb)
        plt.close()

        # Plot symbol data
        plotGraphics(df, 'Candle', toPlotList)
        toPlotList = []

main()




