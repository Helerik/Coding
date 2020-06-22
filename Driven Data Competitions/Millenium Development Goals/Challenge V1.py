
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import warnings

# Performs cubic interpolation on a row of the dataset
def cubic_interpolate(year, year_lis, values):
    
    x = []
    y = []
    
    for i in range(len(values)):
        if isinstance(values[i], str):
            break
        elif np.isnan(values[i]):
            pass
        else:
            y.append(values[i])
            x.append(year_lis[i])

    if len(x) != 3:
        interp_val = interpolate.splrep(x, y, k = 3)
    elif len(x) == 3:
        interp_val = interpolate.splrep(x, y, k = 2)
    elif len(x) == 2:
        interp_val = interpolate.splrep(x, y, k = 1)
        
    return interpolate.splev(year, interp_val)

def make_specX(data):
    
    big_X = []
    new_X = []

    tmp = data.loc[data.index[0]]['Country Name']
    new_X.append(data.loc[data.index[0]].to_numpy())
    
    for i in data.index[1:]:
        if tmp == data.loc[i]['Country Name']:
            new_X.append(data.loc[i].to_numpy())
        else:
            big_X.append(new_X)
            new_X = []
            tmp = data.loc[i]['Country Name']
            new_X.append(data.loc[i].to_numpy())

    return big_X
            

def main():

    # Create reduced dataset, where predictions will be made on.
    ## to_pred_data = train_data.loc[sub_format.index]
    ## to_pred_data.to_csv('to_predict.csv')

    # Import dataset
    train_data = pd.read_csv('TrainingSet.csv', index_col = 0)
    sub_format = pd.read_csv('Submission_Format.csv', index_col = 0)
    to_pred_data = pd.read_csv('to_predict.csv', index_col = 0)
    interp_data = pd.read_csv('interp_set_to_predict.csv', index_col = 0)
    
    # List of year columns
    years = train_data.columns[:-3].to_numpy()
    # List of years integer numbers
    years_range = range(1972, 2007+1)

    # Visualization of the dataset
    print(train_data.head(), end = '\n\n')
    print(to_pred_data.head(), end = '\n\n')

    key = input('Create interpolated set (y/n): ').lower()
    if key == 'y':
        key = True
    elif key == 'n':
        key = False

    if key == False:
        pass
    else:
        # Creates interpolated set of data for prediction
        interp_data = to_pred_data.copy()
        for i in to_pred_data.index:
            for j in range(len(years)):
                try:
                    interp_data[years[j]][i] = cubic_interpolate(years_range[j], years_range, interp_data.loc[i].to_numpy())
                    if interp_data[years[j]][i] < 0:
                        interp_data[years[j]][i] = 0
                    if interp_data[years[j]][i] > 1:
                        interp_data[years[j]][i] = 1
                except:
                    pass

        print(interp_data.head(), end = '\n\n')
        interp_data.to_csv('interp_set_to_predict.csv')

        key = input('Create create one for the training aswel? (y/n): ').lower()
        if key == 'y':
            
            # Creates interpolated set of data for training
            interp_data = train_data.copy()
            for i in train_data.index:
                for j in range(len(years)):
                    try:
                        interp_data[years[j]][i] = cubic_interpolate(years_range[j], years_range, interp_data.loc[i].to_numpy())
                        if interp_data[years[j]][i] < 0:
                            interp_data[years[j]][i] = 0
                        if interp_data[years[j]][i] > 1:
                            interp_data[years[j]][i] = 1
                    except:
                        pass
                if i%1000 == 0:
                    print(i)
            print(interp_data.head(), end = '\n\n')
            interp_data.to_csv('interp_train_set.csv')
            
        elif key == 'n':
            pass




    # Execute predictions
    preds = []

    k = 0
    for i in interp_data.index:
        print(k)
        print()
        k+=1
        
        scores = []
        classifiers = []

        starting_date = 0

        '''
        Falta testar:
        -linear quadratica e raiz de 0 ate 36
        -linear quadratica utilizando tamanhos variados
        -linear quadratica e raiz utilizando tamanhos variados
        -multidimensional, com  make_specX(...)
        '''

        warnings.filterwarnings('ignore')
        # linear prediction
        x = np.array([[1 for _ in years_range[starting_date:]], years_range[starting_date:]]).T
        y = (interp_data.loc[i].to_numpy()[:-3])[starting_date:]
        clf = LinearRegression().fit(x,y)
        scores.append(r2_score(y, clf.predict(x)))

        # square prediction
        def func1(t,a,b,c):
            return a+b*t+c*t*t
        theta1, _ = curve_fit(func1, np.array(years_range[starting_date:]), y)
        pred = [func1(year,*theta1) for year in years_range[starting_date:]]
        scores.append(r2_score(y, pred))

        # square root prediction
        def func2(t,a,b):
            return a+b*np.sqrt(t)
        theta2, _ = curve_fit(func2, np.array(years_range[starting_date:]), y)
        pred = [func2(year,*theta2) for year in years_range[starting_date:]]
        scores.append(r2_score(y, pred))

        warnings.filterwarnings('default')

        if scores.index(np.max(scores)) == 0:
            sub_format['2008 [YR2008]'][i] = clf.predict([[1, 2008]])[0]
            sub_format['2012 [YR2012]'][i] = clf.predict([[1, 2012]])[0]
        elif scores.index(np.max(scores)) == 1:
            sub_format['2008 [YR2008]'][i] = func1(2008,*theta1)
            sub_format['2012 [YR2012]'][i] = func1(2012,*theta1)
        elif scores.index(np.max(scores)) == 2:
            sub_format['2008 [YR2008]'][i] = func2(2008,*theta2)
            sub_format['2012 [YR2012]'][i] = func2(2012,*theta2)


    sub_format.to_csv('Submission_Format.csv')

 
        
        

        


main()






















