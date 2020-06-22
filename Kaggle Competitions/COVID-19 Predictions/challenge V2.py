
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale


def oneHotEnc(X, column, categories):
    
    new_cols = np.zeros((X.shape[0], len(categories)))
    for i in range(X.shape[0]):
        for j in range(len(categories)):
            if X[i][column] == categories[j]:
                new_cols[i][j] = 1
                break
    X = np.delete(X, column, 1)
    X = np.concatenate((X, new_cols.astype(int)), axis = 1)
    
    return X

def clean_data(data):
    
    # Drop fully empty columns
    data = data.dropna(axis = 1, how = 'all')

    # Replaces binary words for numbers
    data = data.replace('negative', 0)
    data = data.replace('positive', 1)
    data = data.replace('not_detected', 0)
    data = data.replace('detected', 1)
    data = data.replace('Não Realizado', 0)
    data = data.replace('absent', 0)
    data = data.replace('not_done', 0)
    data = data.replace('normal', 0)
    data = data.replace('present', 1)

    # Creates dataset1, a modiffied dataset
    data.to_excel('dataset1.xlsx')
    
    data = data.to_numpy()

    # Performs One Hot Encoding on selected columns
    data = oneHotEnc(data, 71, ['altered_coloring', 'clear', 'cloudy', 'lightly_cloudy'])
    data = oneHotEnc(data, 81-1, ['Ausentes', 'Oxalato de Cálcio +++', 'Oxalato de Cálcio -++', 'Urato Amorfo --+',
                                  'Urato Amorfo +++'])
    data = oneHotEnc(data, 86-2, ['citrus_yellow', 'light_yellow', 'orange', 'yellow'])

    # Creates handled data.
    data = pd.DataFrame(data)
    data.to_excel('self_handled_data.xlsx')

    return

def fill_in(data, column, Xs):
    
    t_Xs = np.copy(Xs)
    t_Xs = t_Xs.T
    Ys = np.copy(data[:, column])
    
    k = 0
    for i in range(len(Ys)):
        if np.isnan(Ys[i+k]):
            Ys = np.delete(Ys, i+k)
            t_Xs = np.delete(t_Xs, i+k, axis = 0)
            k -= 1

    reg = LinearRegression().fit(t_Xs, Ys)
    
    Ys = np.copy(data[:, column])

    for i in range(len(Ys)):
        if np.isnan(Ys[i]):
            Ys[i] = reg.predict(np.array([Xs.T[i]]))[0]
            
    data[:, column] = Ys
    return data

def fill_all_in(data, cols, bin_cols, Xs):
    
    for col in cols:
        data = fill_in(data, col, Xs)

    for col in bin_cols:
        data = fill_in(data, col, Xs)
        for i in range(len(data[:, col])):
            if data[i, col] >= 0.5:
                data[i, col] = 1
            else:
                data[i, col] = 0

    return data








def main():
    
##    # Read dataset
##    data = pd.read_excel('dataset.xlsx')
##    
####    clean_data(data)
####
####    input('Handle data > ')
##
##    # Import feature data and label data, respectively
##    Xdata = pd.read_excel('self_handled_data.xlsx').to_numpy()[:, 1:]
##    Ydata = pd.read_excel('self_handled_data_labels.xlsx').to_numpy()
##
##    # Final data handling: convert last string values to integers or floats
##    for i in range(Xdata.shape[0]):
##        for j in range(Xdata.shape[1]):
##            try:
##                Xdata[i][j] = float(Xdata[i][j])
##            except:
##                if Xdata[i][j] == '<1000':
##                    Xdata[i][j] = 500
##
##    # Fills in the missing data doing Linear Regression
##    Xs = np.array([np.array([1 for _ in range(len(Xdata[:, 0]))]), Xdata[:, 0], np.square(Xdata[:, 0])])
##    Xdata = fill_all_in(Xdata,
##                        [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,36,37,38,39,40,41,44,45,46,47,48,49,50,51,
##                         53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
##                         80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100],
##                        [19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,42,43,52], Xs)
##    pd.DataFrame(Xdata).to_excel('self_handled_data_fill_in.xlsx')

    # (Optional) Skip to reading the excel files directly
    Xdata = pd.read_excel('self_handled_data_fill_in.xlsx').to_numpy()[:, 1:]
    Ydata = pd.read_excel('self_handled_data_labels.xlsx').to_numpy().T[0]

    # Create train and test sets
    train_X, test_X, train_y, test_y = train_test_split(
            Xdata, Ydata, test_size = 0.2, random_state=42)
    train_X = scale(train_X)
    test_X = scale(test_X)

    # Classify
    mlp = MLPClassifier(
        hidden_layer_sizes = (100,100,75,30,15),
        alpha = 0.0001,
        learning_rate_init = 0.01,
        max_iter = 10000,
        verbose = True,
        n_iter_no_change = 40,
        tol = 0.00001
        )

    # Fit
    mlp.fit(train_X, train_y)

    # Predict
    preds = mlp.predict(train_X)
    print()
    print('Results on training:\n')
    print(classification_report(train_y, preds, zero_division = 1))
    print('Micro averaged F1 Score: ', f1_score(train_y, preds, average='micro'), '\n')
    print('Macro averaged F1 Score: ', f1_score(train_y, preds))

    preds = mlp.predict(test_X)
    print()
    print('Results on test:\n')
    print(classification_report(test_y, preds, zero_division = 1))
    print('Micro averaged F1 Score: ', f1_score(test_y, preds, average='micro'), '\n')
    print('Macro averaged F1 Score: ', f1_score(test_y, preds))
    
    
    

main()




