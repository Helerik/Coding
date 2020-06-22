
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

# Cria vetores multidimensionais para os dados (utiliza dados vizinhos sobre o pais como dados relevantes para a classificacao):
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

    # Import datasets:

    # Dados para o treino (nao precisa ser utilizada, pois possui mais dados do que o necessario, alem de
    # nao estar com dados interpolados):
    train_data = pd.read_csv('TrainingSet.csv', index_col = 0)

    # O formulario para envio. Alterado somente ao final do programa:
    sub_format = pd.read_csv('Submission_Format.csv', index_col = 0)

    # Apenas as fileiras que irao ser preditas, para as quais devemos 'fittar' a curva (provavelmente nao sera
    # utilizada):
    to_pred_data = pd.read_csv('to_predict.csv', index_col = 0)

    # Apenas as fileiras que irao ser preditas, com dados interpolados no maximo por um spline cubico.
    interp_data = pd.read_csv('interp_set_to_predict.csv', index_col = 0)
    
    # Lista dos anos a serem 'fittados' como inteiros
    years_range = range(1972, 2007+1)

    pd.set_option('display.max_columns', 8)
    # Visualizar parte dos dados
    print(train_data.sample(5), end = '\n\n')
    print('-'*80, end = '\n\n')
    print(to_pred_data.sample(5), end = '\n\n')
    print('-'*80, end = '\n\n')
    
    print()

    # Visualizar dados interpolados
    print(interp_data.sample(5), end = '\n\n')

    input('Aperte ENTER para continuar: ')
    print()
    
    # Executar predicoes:
    try:
        starting_date = int(input('Definir data inicial no intervalo [1972-2007] (Tente 2004): ')) - 1972
        if starting_date > 35:
            starting_date = 35
        elif starting_date < 0:
            starting_date = 0
    except:
        print('Considera-se data inicial como 1972.')
        starting_date = 0
    
    print('-'*80, end = '\n\n')
    
    k = 0
    for i in interp_data.index: 

        '''
        Falta testar:
        
        -linear quadratica e raiz de 0 ate 36 - OK
        -linear quadratica utilizando tamanhos variados - OK
        -dentre as lineares, a com melhor classificacao - OK
        -dentre as lineares, a com melhor predicao para 2007 - OK
        -use diference between model prediction of 2007 vs actual 2007 to decide the predicting function - OK
        -linear quadratica e raiz utilizando tamanhos variados - OK

        -limitar resultados para serem estritamente entre 0 e 1 - UTILIZAR!
        
        -multidimensional, com  make_specX(...)
        -predict 2008 and use that to predict 2012

        Melhores resultados: linear, utilizando dados de 32 a 35, com limitante superior e inferior.
        '''
        
        print('Ajustando curva para dados da linha %d...' %k)
        print()
        k += 1
        
        warnings.filterwarnings('ignore')

        scores = []    
        losses = []
        
        # linear prediction
        x = np.array([[1 for _ in years_range[starting_date:]], years_range[starting_date:]]).T
        y = (interp_data.loc[i].to_numpy()[:-3])[starting_date:]
        clf = LinearRegression().fit(x,y)
        scores.append(r2_score(y, clf.predict(x)))
        losses.append(np.square(clf.predict([[1, 2007]]) - y[-1]))

        # square prediction
        def func1(t,a,b,c):
            return a+b*t+c*t*t
        y = (interp_data.loc[i].to_numpy()[:-3])[starting_date:]
        theta1, _ = curve_fit(func1, np.array(years_range[starting_date:]), y)
        pred = [func1(year,*theta1) for year in years_range[starting_date:]]
        scores.append(r2_score(y, pred))
        losses.append(np.square(func1(2007,*theta1) - y[-1]))

        # square root prediction
        def func2(t,a,b):
            return a+b*np.sqrt(t)
        y = (interp_data.loc[i].to_numpy()[:-3])[starting_date:]
        theta2, _ = curve_fit(func2, np.array(years_range[starting_date:]), y)
        pred = [func2(year,*theta2) for year in years_range[starting_date:]]
        scores.append(r2_score(y, pred))
        losses.append(np.square(func1(2007,*theta1) - y[-1]))


        warnings.filterwarnings('default')

        
        # make predictions based on best score
##        if scores.index(np.max(scores)) == 0:
##            pred_2008 = clf.predict([[1, 2008]])[0]
##            pred_2012 = clf.predict([[1, 2012]])[0]
##        elif scores.index(np.max(scores)) == 1:
##            pred_2008 = func1(2008,*theta1)
##            pred_2012 = func1(2012,*theta1)
##        elif scores.index(np.max(scores)) == 2:
##            pred_2008 = func2(2008,*theta2)
##            pred_2012 = func2(2012,*theta2)

        # make predictions based on best classifier to predict 2007
        if losses.index(np.min(losses)) == 0:
            pred_2008 = clf.predict([[1, 2008]])[0]
            pred_2012 = clf.predict([[1, 2012]])[0]
        elif losses.index(np.min(losses)) == 1:
            pred_2008 = func1(2008,*theta1)
            pred_2012 = func1(2012,*theta1)
        elif losses.index(np.min(losses)) == 2:
            pred_2008 = func2(2008,*theta2)
            pred_2012 = func2(2012,*theta2)

        # restricitons for the prediction (must be between 0 and 1)
        if pred_2008 < 0:
            pred_2008 = 0
        if pred_2008 > 1:
            pred_2008 = 1
        if pred_2012 < 0:
            pred_2012 = 0
        if pred_2012 > 1:
            pred_2012 = 1

        # write to submission format
        sub_format['2008 [YR2008]'][i] = pred_2008
        sub_format['2012 [YR2012]'][i] = pred_2012

    # submit...
    sub_format.to_csv('Submission_Format.csv')

main()






















