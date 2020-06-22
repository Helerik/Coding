
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import warnings

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
        -predict 2008 and use that to predict 2012 - 

        -limitar resultados para serem estritamente entre 0 e 1 - UTILIZAR!
        
        -multidimensional, com  make_specX(...)
        

        Melhores resultados: linear, utilizando dados de 32 a 35, com limitante superior e inferior.
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

        clf = LinearRegression().fit(x,y)

        pred_2008 = clf.predict([[1, 2008]])[0]
        # restricitons for the prediction (must be between 0 and 1)
        if pred_2008 < 0:
            pred_2008 = 0
        if pred_2008 > 1:
            pred_2008 = 1

        # 2nd classification
        y = np.append((interp_data.loc[i].to_numpy()[:-3]), [pred_2008])[starting_date+1:]
        years_range = range(1972, 2008 + 1)
        x = np.array([[1 for _ in years_range[starting_date+1:]], years_range[starting_date+1:]]).T

        clf = LinearRegression().fit(x,y)

        pred_2009 = clf.predict([[1, 2009]])[0]
        # restricitons for the prediction (must be between 0 and 1)
        if pred_2009 < 0:
            pred_2009 = 0
        if pred_2009 > 1:
            pred_2009 = 1

        # 3rd classification
        y = np.append((interp_data.loc[i].to_numpy()[:-3]), [pred_2008, pred_2009])[starting_date+2:]
        years_range = range(1972, 2009 + 1)
        x = np.array([[1 for _ in years_range[starting_date+2:]], years_range[starting_date+2:]]).T

        clf = LinearRegression().fit(x,y)

        pred_2010 = clf.predict([[1, 2010]])[0]
        # restricitons for the prediction (must be between 0 and 1)
        if pred_2010 < 0:
            pred_2010 = 0
        if pred_2010 > 1:
            pred_2010 = 1

        # 4th classification
        y = np.append((interp_data.loc[i].to_numpy()[:-3]), [pred_2008, pred_2009, pred_2010])[starting_date+2:]
        years_range = range(1972, 2010 + 1)
        x = np.array([[1 for _ in years_range[starting_date+2:]], years_range[starting_date+2:]]).T

        clf = LinearRegression().fit(x,y)

        pred_2011 = clf.predict([[1, 2011]])[0]
        # restricitons for the prediction (must be between 0 and 1)
        if pred_2011 < 0:
            pred_2011 = 0
        if pred_2011 > 1:
            pred_2011 = 1

        # 5th classification
        y = np.append((interp_data.loc[i].to_numpy()[:-3]), [pred_2008, pred_2009, pred_2010, pred_2011])[starting_date+2:]
        years_range = range(1972, 2011 + 1)
        x = np.array([[1 for _ in years_range[starting_date+2:]], years_range[starting_date+2:]]).T
        
        clf = LinearRegression().fit(x,y)

        pred_2012 = clf.predict([[1, 2012]])[0]
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






















