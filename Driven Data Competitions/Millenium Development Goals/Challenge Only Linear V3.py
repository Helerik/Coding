
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
    new_X.append(data.loc[data.index[0]].to_numpy().tolist())
    
    for i in data.index[1:]:
        if tmp == data.loc[i]['Country Name']:
            new_X.append(data.loc[i].to_numpy().tolist())
        else:
            big_X.append(new_X)
            new_X = []
            tmp = data.loc[i]['Country Name']
            new_X.append(data.loc[i].to_numpy().tolist())

    return big_X
            
def main():

    # Import datasets:

    # Dados para o treino (nao precisa ser utilizada, pois possui mais dados do que o necessario, alem de
    # nao estar com dados interpolados):
    train_data = pd.read_csv('TrainingSet.csv', index_col = 0)

    # O formulario para envio. Alterado somente ao final do programa:
    sub_format = pd.read_csv('Submission_Format VV.csv', index_col = 0)

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
        
        print('Ajustando curva para dados da linha %d...' %k)
        print()
        k += 1
        
        warnings.filterwarnings('ignore')

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
        sub_format['2009 [YR2009]'][i] = pred_2009
        sub_format['2010 [YR2010]'][i] = pred_2010
        sub_format['2011 [YR2011]'][i] = pred_2011
        sub_format['2012 [YR2012]'][i] = pred_2012

    to_append = np.array(make_specX(interp_data))
    k = -1
    k2 = -1
    k3 = -1
    k4 = -1
    k5 = -1
    m = None
    for lis in to_append:
        years_range = range(1972, 2007 + 1)
        x = np.array([[1 for _ in years_range[starting_date:]], years_range[starting_date:]])
        for lis1 in lis:
            x = np.append(x, [lis1[starting_date:-3]], axis = 0)

        m = k+1
        
        to_append1 = [1, 2009]
        for j in range(2, len(x)):
            k += 1
            print('Reajustando curva para dados da linha %d...' %k)
            print()
            i = interp_data.index[k]

            t_x = np.delete(x, j, 0).T
            y = (interp_data.loc[i].to_numpy()[:-3])[starting_date:]

            pred_2008 = [1, 2008]
            p = 2
            for l in range(len(x)-2):
                i2 = interp_data.index[l+m]
                if l != j - 2:
                    acpt_val = abs(np.corrcoef([interp_data.loc[i2].to_numpy()[:-3].astype(float),
                                interp_data.loc[interp_data.index[j-2+m]].to_numpy()[:-3].astype(float)])[1,0])
                    if np.isnan(acpt_val):
                        acpt_val = 0
                    if acpt_val >= 0.9:
                        pred_2008.append(sub_format['2008 [YR2008]'][i2])
                        p += 1
                    else:
                        t_x = np.delete(t_x, p, 1)

            clf = LinearRegression().fit(t_x,y)
            
            pred_2008 = clf.predict([pred_2008])[0]

            # restricitons for the prediction (must be between 0 and 1)
            if pred_2008 < 0:
                pred_2008 = 0
            if pred_2008 > 1:
                pred_2008 = 1
            
            # write to submission format
            sub_format['2008 [YR2008]'][i] = pred_2008
            to_append1.append(pred_2008)
            
        x = np.append(x, np.array([to_append1]).T ,1)
        x = np.delete(x, 0, 1)
        to_append2 = [1, 2010]
        for j in range(2, len(x)):
            k2 += 1

            i = interp_data.index[k2]

            t_x = np.delete(x, j, 0).T
            y = np.append((interp_data.loc[i].to_numpy()[:-3]), [to_append1[j]])[starting_date+1:]

            pred_2009 = [1, 2009]
            p = 2
            for l in range(len(x)-2):
                i2 = interp_data.index[l+m]
                if l != j - 2:
                    acpt_val = abs(np.corrcoef([interp_data.loc[i2].to_numpy()[:-3].astype(float),
                                interp_data.loc[interp_data.index[j-2+m]].to_numpy()[:-3].astype(float)])[1,0])
                    if np.isnan(acpt_val):
                        acpt_val = 0
                    if acpt_val >= 0.9:
                        pred_2009.append(sub_format['2009 [YR2009]'][i2])
                        p += 1
                    else:
                        t_x = np.delete(t_x, p, 1)

            clf = LinearRegression().fit(t_x,y)
            
            pred_2009 = clf.predict([pred_2009])[0]

            # restricitons for the prediction (must be between 0 and 1)
            if pred_2009 < 0:
                pred_2009 = 0
            if pred_2009 > 1:
                pred_2009 = 1
            
            # write to submission format
            sub_format['2009 [YR2009]'][i] = pred_2009
            to_append2.append(pred_2009)

        x = np.append(x, np.array([to_append2]).T ,1)
        x = np.delete(x, 0, 1)
        to_append3 = [1, 2011]
        for j in range(2, len(x)):
            k3 += 1

            i = interp_data.index[k3]

            t_x = np.delete(x, j, 0).T
            y = np.append((interp_data.loc[i].to_numpy()[:-3]), [to_append1[j], to_append2[j]])[starting_date+2:]

            pred_2010 = [1, 2010]
            p = 2
            for l in range(len(x)-2):
                i2 = interp_data.index[l+m]
                if l != j - 2:
                    acpt_val = abs(np.corrcoef([interp_data.loc[i2].to_numpy()[:-3].astype(float),
                                interp_data.loc[interp_data.index[j-2+m]].to_numpy()[:-3].astype(float)])[1,0])
                    if np.isnan(acpt_val):
                        acpt_val = 0
                    if acpt_val >= 0.9:
                        pred_2010.append(sub_format['2010 [YR2010]'][i2])
                        p += 1
                    else:
                        t_x = np.delete(t_x, p, 1)

            clf = LinearRegression().fit(t_x,y)
            
            pred_2010 = clf.predict([pred_2010])[0]

            # restricitons for the prediction (must be between 0 and 1)
            if pred_2010 < 0:
                pred_2010 = 0
            if pred_2010 > 1:
                pred_2010 = 1
            
            # write to submission format
            sub_format['2010 [YR2010]'][i] = pred_2010
            to_append3.append(pred_2010)

        x = np.append(x, np.array([to_append3]).T ,1)
        to_append4 = [1, 2012]
        for j in range(2, len(x)):
            k4 += 1

            i = interp_data.index[k4]

            t_x = np.delete(x, j, 0).T
            y = np.append((interp_data.loc[i].to_numpy()[:-3]), [to_append1[j], to_append2[j], to_append3[j]])[starting_date+2:]

            pred_2011 = [1, 2011]
            p = 2
            for l in range(len(x)-2):
                i2 = interp_data.index[l+m]
                if l != j - 2:
                    acpt_val = abs(np.corrcoef([interp_data.loc[i2].to_numpy()[:-3].astype(float),
                                interp_data.loc[interp_data.index[j-2+m]].to_numpy()[:-3].astype(float)])[1,0])
                    if np.isnan(acpt_val):
                        acpt_val = 0
                    if acpt_val >= 0.9:
                        pred_2011.append(sub_format['2011 [YR2011]'][i2])
                        p += 1
                    else:
                        t_x = np.delete(t_x, p, 1)

            clf = LinearRegression().fit(t_x,y)
            
            pred_2011 = clf.predict([pred_2011])[0]

            # restricitons for the prediction (must be between 0 and 1)
            if pred_2011 < 0:
                pred_2011 = 0
            if pred_2011 > 1:
                pred_2011 = 1
            
            # write to submission format
            sub_format['2011 [YR2011]'][i] = pred_2011
            to_append4.append(pred_2011)

        x = np.append(x, np.array([to_append4]).T ,1)
        for j in range(2, len(x)):
            k5 += 1

            i = interp_data.index[k5]

            t_x = np.delete(x, j, 0).T
            y = np.append((interp_data.loc[i].to_numpy()[:-3]), [to_append1[j], to_append2[j], to_append3[j], to_append4[j]])[starting_date+2:]

            pred_2012 = [1, 2012]
            p = 2
            for l in range(len(x)-2):
                i2 = interp_data.index[l+m]
                if l != j - 2:
                    acpt_val = abs(np.corrcoef([interp_data.loc[i2].to_numpy()[:-3].astype(float),
                                interp_data.loc[interp_data.index[j-2+m]].to_numpy()[:-3].astype(float)])[1,0])
                    if np.isnan(acpt_val):
                        acpt_val = 0
                    if acpt_val >= 0.9:
                        pred_2012.append(sub_format['2012 [YR2012]'][i2])
                        p += 1
                    else:
                        t_x = np.delete(t_x, p, 1)

            clf = LinearRegression().fit(t_x,y)
            
            pred_2012 = clf.predict([pred_2012])[0]

            # restricitons for the prediction (must be between 0 and 1)
            if pred_2012 < 0:
                pred_2012 = 0
            if pred_2012 > 1:
                pred_2012 = 1
            
            # write to submission format
            sub_format['2012 [YR2012]'][i] = pred_2012

        
            

    # submit...
    sub_format.to_csv('Submission_Format VV.csv')

main()






















