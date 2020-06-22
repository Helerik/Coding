#!/usr/bin/env python3

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

    # Separate primaryName from primaryMode
    primaryName = primaryName.split(', ', 1)[0]

    # Create new data frame with new column
    newFrame = dataFrame.copy(deep = True)
    newFrame['Moving Average ' + primaryName] = newFrame[primaryName].rolling(window = timePeriod, min_periods = 0).mean()
    
    return newFrame

def plotGeneral(dataFrame, primaryName, subplot = [], figsize = (14.4/1.5, 9.6/1.5)):
    # Creates a figure
    plt.figure(num = primaryName, figsize = figsize)

    # Separate primaryName from primaryMode
    try:
        NameMode = primaryName.split(', ', 1)
        primaryName = NameMode[0]
        primaryMode = NameMode[1].lower()
    except:
        primaryMode = 'std'
    
    secondaryPlot = [
        'Volume'
        ]

    # Finds if a second plot window will be needed
    secPlot_key = 0
    for i in range(len(subplot)):
        subplot[i] = subplot[i].split(', ', 1)
        if len(subplot[i]) == 1:
            subplot[i] = [subplot[i][0], 'std']
        plotName = subplot[i][0]
            
        if plotName in secondaryPlot:
            secPlot_key = 1 

    # If there is any subplot and one more window will be needed:
    if len(subplot) > 0 and secPlot_key:
        secPlot_key = 0

        # Prepares primaryName plot
        ax1 = plt.subplot2grid((7,1), (0,0), rowspan = 5, colspan = 1)

        # Plot primaryName with the mode
        if primaryMode == 'std':
            ax1.plot(dataFrame.index, dataFrame[primaryName], label = primaryName)
        elif primaryMode == 'bar':
            ax1.bar(dataFrame.index, dataFrame[primaryName], label = primaryName)
            
        for plotName in subplot:
            # If the secondary plot needs a second window, creates a second window +
            # plots the graphic. Can only have one more window; greedy algorithm (for now).
            if plotName[0] in secondaryPlot and not secPlot_key:
                ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 2, colspan = 1)

                if plotName[1].lower() == 'std':
                    ax2.plot(dataFrame.index, dataFrame[plotName[0]], label = plotName[0])
                elif plotName[1].lower() == 'bar':
                    ax2.bar(dataFrame.index, dataFrame[plotName[0]], label = plotName[0])
                
                ax2.legend()
                secPlot_key = 1
            # Plots all secondary plots that don't need a second window on main window.
            elif not plotName[0] in secondaryPlot:
                
                if plotName[1].lower() == 'std':
                    ax1.plot(dataFrame.index, dataFrame[plotName[0]], label = plotName[0])
                if plotName[1].lower() == 'bar':
                    ax1.bar(dataFrame.index, dataFrame[plotName[0]], label = plotName[0])
                    
        ax1.legend()

    # If more windows are not needed but there are secondary plots:
    elif len(subplot) > 0 and not secPlot_key:
        
        if primaryMode == 'std':
            plt.plot(dataFrame.index, dataFrame[primaryName], label = primaryName)
        elif primaryMode == 'bar':
            plt.bar(dataFrame.index, dataFrame[primaryName], label = primaryName)
            
        for plotName in subplot:
            
            if plotName[1].lower() == 'std':
                ax1.plot(dataFrame.index, dataFrame[plotName[0]], label = plotName[0])
            if plotName[1].lower() == 'bar':
                ax1.bar(dataFrame.index, dataFrame[plotName[0]], label = plotName[0])
                
        plt.legend()
    # Else, plots primaryName only, even if it is in secondaryPlot
    else:
        
        if primaryMode == 'std':
            plt.plot(dataFrame.index, dataFrame[primaryName], label = primaryName)
        elif primaryMode == 'bar':
            plt.bar(dataFrame.index, dataFrame[primaryName], label = primaryName)
            
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
    plotGeneral(df, 'Adj Close, Std', ['Moving Average Adj Close', 'High', 'Volume, Bar'])

main()






