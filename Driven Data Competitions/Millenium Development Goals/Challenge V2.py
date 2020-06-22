
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

    input('Press ENTER to continue:')
    print()

    
    # Executar predicoes:

    try:
        starting_date = int(input('Definir data inicial no intervalo [1972-2007]: ')) - 1972
        if starting_date > 35:
            starting_date = 35
        elif starting_date < 0:
            starting_date = 0
    except:
        print('Exception: Considera-se data inicial como 1972.')
        starting_date = 0
    
    k = 0
    print('-'*80, end = '\n\n')
    for i in interp_data.index:
        print('Ajustando curva para dados da linha %d...' %k)
        print()
        k+=1
        
        scores = []
        test_dif = []       

        '''
        Falta testar:
        
        -linear quadratica e raiz de 0 ate 36 - OK
        -linear quadratica utilizando tamanhos variados - OK
        -dentre as lineares, a com melhor classificacao - OK
        -dentre as lineares, a com melhor predicao para 2007 - OK

        -limitar resultados para serem estritamente entre 0 e 1 - UTILIZAR!
        
        -linear quadratica de 0 a 36
        -linear raiz de 0 a 36
        -linear quadratica e raiz utilizando tamanhos variados
        -multidimensional, com  make_specX(...)
        -predict 2008 and use that to predict 2012 aswel
        -use diference between model prediction of 2007 vs actual 2007 to decide the predicting function
        '''

        warnings.filterwarnings('ignore')
        
        # linear prediction
##        x = np.array([[1 for _ in years_range[starting_date:]], years_range[starting_date:]]).T
##        y = (interp_data.loc[i].to_numpy()[:-3])[starting_date:]
##        clf = LinearRegression().fit(x,y)
##        scores.append(r2_score(y, clf.predict(x)))
##        test_dif.append(abs(clf.predict([[1,2007]]) - y[-1]))
##
##        if scores.index(np.max(scores)) == 0:
##            pred_2008 = clf.predict([[1, 2008]])[0]
##            pred_2012 = clf.predict([[1, 2012]])[0]
##            
##            if pred_2008 < 0:
##                pred_2008 = 0
##            if pred_2008 > 1:
##                pred_2008 = 1
##            if pred_2012 < 0:
##                pred_2012 = 0
##            if pred_2012 > 1:
##                pred_2012 = 1
##                
##            sub_format['2008 [YR2008]'][i] = pred_2008
##            sub_format['2012 [YR2012]'][i] = pred_2012

        # multiple linear predictions:
        classifiers = []
        best_score = 0
        least_loss = 10
        best_score_idx = None
        least_loss_idx = None
        for j in range(0, len(years_range)):

            x = np.array([[1 for _ in years_range[j:]], years_range[j:]]).T
            y = (interp_data.loc[i].to_numpy()[:-3])[j:]
            clf = LinearRegression().fit(x,y)
            classifiers.append(clf)
            score = r2_score(y, clf.predict(x))
            loss = abs(clf.predict([[1, 2007]]) - y[-1])[0]
            
            if score >= best_score:
                best_score = score
                best_score_idx = j

            if loss <= least_loss:
                least_loss = loss
                least_loss_idx = j

##        pred_2008 = classifiers[best_score_idx].predict([[1, 2008]])[0]
##        pred_2012 = classifiers[best_score_idx].predict([[1, 2012]])[0]

        pred_2008 = classifiers[least_loss_idx].predict([[1, 2008]])[0]
        pred_2012 = classifiers[least_loss_idx].predict([[1, 2012]])[0]
        
        if pred_2008 < 0:
            pred_2008 = 0
        if pred_2008 > 1:
            pred_2008 = 1
        if pred_2012 < 0:
            pred_2012 = 0
        if pred_2012 > 1:
            pred_2012 = 1
            
        sub_format['2008 [YR2008]'][i] = pred_2008
        sub_format['2012 [YR2012]'][i] = pred_2012
            

        # square prediction
##        def func1(t,a,b,c):
##            return a+b*t+c*t*t
##        y = (interp_data.loc[i].to_numpy()[:-3])[30:]
##        theta1, _ = curve_fit(func1, np.array(years_range[30:]), y)
##        pred = [func1(year,*theta1) for year in years_range[30:]]
##        scores.append(r2_score(y, pred))
##
##
##        # square root prediction
##        def func2(t,a,b,c):
##            if t+c < 0:
##                c = -t
##            return a+b*np.sqrt(t+c)
##        y = (interp_data.loc[i].to_numpy()[:-3])[30:]
##        theta2, _ = curve_fit(func2, np.array(years_range[starting_date:]), y)
##        pred = [func2(year,*theta2) for year in years_range[starting_date:]]
##        scores.append(r2_score(y, pred))
##

##        print('linear:\n R2 = %f, Dif = %f' %(scores[0], test_dif[0]))
##        print('quadratic:\n R2 = %f, Dif = %f' %(scores[1], test_dif[1]))
##        print('square root:\n R2 = %f, Dif = %f' %(scores[2], test_dif[2]))
##        print('\nBest R2: %f, index = %d\nBest Dif: %f, index = %d' %(np.max(scores), scores.index(np.max(scores)),
##                                                                      np.min(test_dif), test_dif.index(np.min(test_dif))))
##        
##        input()

        warnings.filterwarnings('default')

##        if scores.index(np.max(scores)) == 0:
##            sub_format['2008 [YR2008]'][i] = clf.predict([[1, 2008]])[0]
##            sub_format['2012 [YR2012]'][i] = clf.predict([[1, 2012]])[0]
##        elif scores.index(np.max(scores)) == 1:
##            sub_format['2008 [YR2008]'][i] = func1(2008,*theta1)
##            sub_format['2012 [YR2012]'][i] = func1(2012,*theta1)
##        elif scores.index(np.max(scores)) == 2:
##            sub_format['2008 [YR2008]'][i] = func2(2008,*theta2)
##            sub_format['2012 [YR2012]'][i] = func2(2012,*theta2)


    sub_format.to_csv('Submission_Format.csv')

 
        
        

        


main()






















