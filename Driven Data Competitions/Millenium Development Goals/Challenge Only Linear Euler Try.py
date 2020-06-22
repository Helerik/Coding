
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        -predict 2008 and use that to predict 2012 - 

        -limitar resultados para serem estritamente entre 0 e 1 - UTILIZAR!

        '''
        
        print('Ajustando curva para dados da linha %d...' %k)
        print()
        k += 1
        
        warnings.filterwarnings('ignore')
        
        # linear prediction

        # 1st classification
        years_range = range(1972, 2007 + 1)
        x = np.array([[1 for _ in years_range[starting_date:]], years_range[starting_date:]]).T
        y = (interp_data.loc[i].to_numpy()[:-3])[starting_date:]

        clf1 = LinearRegression().fit(x,y)
        k1 = clf1.predict([[1, 2008]])[0]

        # 1.1st classification
        y = np.append((interp_data.loc[i].to_numpy()[:-3]), [k1])[starting_date+1:]
        years_range = range(1972, 2008 + 1)
        x = np.array([[1 for _ in years_range[starting_date+1:]], years_range[starting_date+1:]]).T

        clf2 = LinearRegression().fit(x,y)
        k2 = clf2.predict([[1, 2008]])[0]
        
        pred_2008 = (k1+k2)/2
        # restricitons for the prediction (must be between 0 and 1)
        if pred_2008 < 0:
            pred_2008 = 0
        if pred_2008 > 1:
            pred_2008 = 1

        # 2nd classification
        y = np.append((interp_data.loc[i].to_numpy()[:-3]), [pred_2008])[starting_date+1:]
        years_range = range(1972, 2008 + 1)
        x = np.array([[1 for _ in years_range[starting_date+1:]], years_range[starting_date+1:]]).T

        clf1 = LinearRegression().fit(x,y)
        k1 = clf1.predict([[1, 2009]])[0]

        # 2.1st classification
        y = np.append((interp_data.loc[i].to_numpy()[:-3]), [pred_2008, k1])[starting_date+2:]
        years_range = range(1972, 2009 + 1)
        x = np.array([[1 for _ in years_range[starting_date+2:]], years_range[starting_date+2:]]).T

        clf2 = LinearRegression().fit(x,y)
        k2 = clf2.predict([[1, 2009]])[0]

        pred_2009 = (k1+k2)/2
        # restricitons for the prediction (must be between 0 and 1)
        if pred_2009 < 0:
            pred_2009 = 0
        if pred_2009 > 1:
            pred_2009 = 1

        # 3rd classification
        y = np.append((interp_data.loc[i].to_numpy()[:-3]), [pred_2008, pred_2009])[starting_date+2:]
        years_range = range(1972, 2009 + 1)
        x = np.array([[1 for _ in years_range[starting_date+2:]], years_range[starting_date+2:]]).T
        
        clf1 = LinearRegression().fit(x,y)
        k1 = clf1.predict([[1, 2012]])[0]

        # 3.1st classification
        y = np.append((interp_data.loc[i].to_numpy()[:-3]), [pred_2008, pred_2009, k1])[starting_date+3:]
        years_range = range(1972, 2010 + 1)
        x = np.array([[1 for _ in years_range[starting_date+3:]], years_range[starting_date+3:]]).T

        clf2 = LinearRegression().fit(x,y)
        k2 = clf2.predict([[1, 2012]])[0]

        pred_2012 = (k1+k2)/2
        # restricitons for the prediction (must be between 0 and 1)
        if pred_2012 < 0:
            pred_2012 = 0
        if pred_2012 > 1:
            pred_2012 = 1

        warnings.filterwarnings('default')

        # write to submission format
        sub_format['2008 [YR2008]'][i] = pred_2008
        sub_format['2012 [YR2012]'][i] = pred_2012

    # submit...
    sub_format.to_csv('Submission_Format.csv')

main()






















