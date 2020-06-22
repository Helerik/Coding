
import pandas as pd
import numpy as np
from sklearn import svm, preprocessing
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import winsound

frequency = 165  # Set Frequency To 165 Hertz
duration = 750  # Set Duration To 750 ms == 0.75 second


# Custom class of MLPClassifier to work with AdaBoost
class customMLPClassifer(MLPClassifier):
    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        y_train_resampled = np.zeros((len(y_train)), dtype=np.int)
        for i in range(len(X_train)):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled


    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))


# One hot encoding turns each non-numeric value of a feature into a new binary feature and removes the old one
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

# Ordinal encoding turns each non-numeric value of a feature into a number
def OrdinalEnc(X, column, categories):
    
    for i in range(X.shape[0]):
        for j in range(len(categories)):
            if X[i][column] == categories[j]:
                X[i][column] = j
                break
    
    return X

# Scales the selected columns of the data and performs mean normalization
def scaleCols(X, colLis):
    for i in colLis:
        X[:, i] = preprocessing.scale(X[:, i])
    return X

# Performs one of the selected preprocessings on the data
def preProcess(X, proc_type = None):
    
    if proc_type == None:
        return X
    
    if proc_type == 'OneHotEncoding':
        
        # Preprocess data with one hot encoding (for every non-numerical feature, makes new features)
        # There should be a total of 68 new features
        
        X = oneHotEnc(X, 7, ['t', 'o', 'n'])
        X = oneHotEnc(X, 7, ['h', 'i', 'r', 'u', 'w'])
        X = oneHotEnc(X, 7, ['n', 'q', 'x'])
        X = oneHotEnc(X, 7, ['f', 'm', 'v', 'x', 'z'])
        X = oneHotEnc(X, 7, ['j', 'q', 's', 'x'])
        X = oneHotEnc(X, 7, ['j', 'o', 's', 't'])
        X = oneHotEnc(X, 7, ['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'])
        X = oneHotEnc(X, 18, ['a', 'r', 'v', 'w']) # Optional

        return X

    if proc_type == 'Ordinal':
        
        # Preprocess data with ordinal encoding (for every non-numerical value, assign a integer value)
        
        X = OrdinalEnc(X, 7, ['t', 'o', 'n'])
        X = OrdinalEnc(X, 8, ['h', 'i', 'r', 'u', 'w'])
        X = OrdinalEnc(X, 9, ['n', 'q', 'x'])
        X = OrdinalEnc(X, 10, ['f', 'm', 'v', 'x', 'z'])
        X = OrdinalEnc(X, 11, ['j', 'q', 's', 'x'])
        X = OrdinalEnc(X, 12, ['j', 'o', 's', 't'])
        X = OrdinalEnc(X, 13, ['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'])
        X = OrdinalEnc(X, 25, ['a', 'r', 'v', 'w'])
        
        return X

# Similar to voting algorithm
# Uses binary classifiers to vote
def multNN(X, y, test_X, test_y):
  
    X12, y12 = [], []
    X23, y23 = [], []
    X31, y31 = [], []
    for i in range(len(y)):
        if y[i] == 1:
            y12.append(1)
            y31.append(1)
            X12.append(X[i])
            X31.append(X[i])
        if y[i] == 2:
            y12.append(2)
            y23.append(2)
            X23.append(X[i])
            X12.append(X[i])
        if y[i] == 3:
            y23.append(3)
            y31.append(3)
            X23.append(X[i])
            X31.append(X[i])

    inp = input('Choose between hard and soft voting: ')
    print()

    print('-- Training multi-Classifier --\n')

    mlp12 = MLPClassifier(hidden_layer_sizes = (20,20), max_iter = 10000, activation = 'logistic')
    mlp23 = MLPClassifier(hidden_layer_sizes = (20,20), max_iter = 10000, activation = 'logistic')
    mlp31 = MLPClassifier(hidden_layer_sizes = (20,20), max_iter = 10000, activation = 'logistic')
    mlp123 = MLPClassifier(hidden_layer_sizes = (20,20), max_iter = 10000, activation = 'logistic')

    mlp12.fit(X12, y12)
    mlp23.fit(X23, y23)
    mlp31.fit(X31, y31)
    mlp123.fit(X, y)

    eclf = VotingClassifier(
            estimators = [('mlp123', mlp123), ('mlp12', mlp12), ('mlp23', mlp23), ('mlp31', mlp31)], voting = inp)

    eclf = eclf.fit(X, y)
    preds_t = eclf.predict(X)
    
    print('Results on cross-validation set:')
    print(classification_report(y, preds_t, zero_division = 1))
    print('Micro averaged F1 Score: ', f1_score(y, preds_t, average='micro'), '\n')

    preds = eclf.predict(test_X)

    print('Results on cross-validation set:')
    print(classification_report(test_y, preds, zero_division = 1))
    print('Micro averaged F1 Score: ', f1_score(test_y, preds, average='micro'), '\n')

    winsound.Beep(frequency, duration)

    return eclf

def submit(classifier):
    
    # Imports the testset.
    X = pd.read_csv('test_values.csv', index_col = 0).to_numpy()

    # Preprocesses data (one hot or ordinal encoding)
    X = preProcess(X, 'OneHotEncoding')

    # Performs feature scaling on selected columns
    X = scaleCols(X, [0,1,2,4,5,6])

    # (Optional) Removes features that I find to be useless
##    for i in range(18,31):
##        X = np.delete(X, 18, 1)

    y = classifier.predict(X)

    form = pd.read_csv('submission_format.csv', index_col = 0)
    form['damage_grade'] = y.astype(int)

    form.to_csv('submission_format.csv')
































def main():

    # Imports the dataset and labels and turns them into numpy arrays.
    X = pd.read_csv('train_values.csv', index_col = 0).to_numpy()
    y = np.array(pd.read_csv('train_labels.csv', index_col = 0).to_numpy().T[0])

    # Preprocesses data (one hot or ordinal encoding)
    X = preProcess(X, 'OneHotEncoding')

    # Performs feature scaling on selected columns
    X = scaleCols(X, [0,1,2,4,5,6])

    # (Optional) Removes features that I find to be useless
##    for i in range(18,31):
##        X = np.delete(X, 18, 1)

    # Set size multiplier
    k = 200
    
    # Size of training set
    n = 100*k

    # Size of cross-validation set
    m = 20*k

    # Creates train set and test set
##    train_X = X[:n]
##    train_y = y[:n]
##    test_X = X[n:n+m]
##    test_y = y[n:n+m]

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size = 0.2, random_state=42)
    

    # C parameter (Large C = low bias // Small C = high bias)
    C = 2

    # Test values of C?
    test_C = 0

    last_classifier = None








    # 'Front-end'
    while True:
        print("Choose the training model: ")
        print(' - SVM\n - Network\n - Vote\n - Mult\n')
        inp = input('> ').lower()
        print()
        
        if inp == 'svm':
            if test_C == 1:
                C = [0.01, 0.03, 0.1, 0.3, 1, 2, 3, 5, 10]
                for c in C:

                    print('Testing C =', c, '\n')
                    
                    clf = svm.SVC(decision_function_shape='ovr', C = c)
                    clf.fit(train_X, train_y)
                    t_preds = clf.predict(train_X)

                    print('Results on training set:')
                    print('Micro averaged F1 Score: ', f1_score(train_y, t_preds, average='micro'), '\n')

                    print('-'*80, '\n')

                    preds = clf.predict(test_X)

                    print('Results on cross-validation set:')
                    print('Micro averaged F1 Score: ', f1_score(test_y, preds, average='micro'), '\n')

                    print('='*80, '\n')

            else:

                print('-- Training SVM --\n')
                  
                # Performs SVM classification
                clf = svm.SVC(decision_function_shape='ovr', C = C)
                clf.fit(train_X, train_y)
                t_preds = clf.predict(train_X)

                print('Results on training set:')
                print(classification_report(train_y, t_preds, zero_division = 1))
                print('Micro averaged F1 Score: ', f1_score(train_y, t_preds, average='micro'), '\n')

                print('-'*80, '\n')

                preds = clf.predict(test_X)

                print('Results on cross-validation set:')
                print(classification_report(test_y, preds, zero_division = 1))
                print('Micro averaged F1 Score: ', f1_score(test_y, preds, average='micro'), '\n')

                last_classifier = clf

                winsound.Beep(frequency, duration)

            print('='*80, '\n')

        elif inp == 'network':

            inp = int(input('Define maximum number of iterations: '))
            print()

            inp2 = [int(x) for x in input("Define network architecture: ").split(',')]
            print()

            alpha = input("Define regularization term alpha (default = 0.0001): ")
            if alpha == '':
                alpha = 0.0001
            print()

            activation = input("Define ctivation function (default = 'relu'): ")
            if activation == '':
                activation = 'relu'
            print()

            l_rate = input("Define initial learning rate (default = 0.001): ")
            if l_rate == '':
                l_rate = 0.001
            print()

            decay = input("Define learning rate decay (default = 'constant'): ")
            if decay == '':
                decay = 'constant'
            print()

            print('-- Training neural network --\n')

            mlp = MLPClassifier(
                hidden_layer_sizes = inp2, max_iter = inp, alpha = float(alpha), activation = activation,
                learning_rate = decay, learning_rate_init = float(l_rate), verbose = True)
            mlp.fit(train_X, train_y)
            
            t_preds = mlp.predict(train_X)

            print('Results on training set:')
            print(classification_report(train_y, t_preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(train_y, t_preds, average='micro'), '\n')

            print('-'*80, '\n')

            preds = mlp.predict(test_X)

            print('Results on cross-validation set:')
            print(classification_report(test_y, preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(test_y, preds, average='micro'), '\n')

            last_classifier = mlp

            winsound.Beep(frequency, duration)

            print('='*80, '\n')

        elif inp == 'vote':

            inp = input('Choose between hard and soft voting: ')
            print()

            print('-- Training Voting Classifier --\n')

            mlp1 = MLPClassifier(hidden_layer_sizes = (20,20), max_iter = 10000, activation = 'logistic')
            mlp2 = MLPClassifier(hidden_layer_sizes = (40,40), max_iter = 10000, activation = 'logistic')
            mlp3 = MLPClassifier(hidden_layer_sizes = (30,30,30,30), max_iter = 10000, activation = 'logistic')
            mlp4 = MLPClassifier(hidden_layer_sizes = (20,40), max_iter = 10000, activation = 'logistic')

            eclf = VotingClassifier(
                estimators = [('mlp1', mlp1), ('mlp2', mlp2), ('mlp3', mlp3), ('mlp4', mlp4)], voting = inp)

            eclf = eclf.fit(train_X, train_y)

            t_preds = eclf.predict(train_X)

            print('Results on training set:')
            print(classification_report(train_y, t_preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(train_y, t_preds, average='micro'), '\n')

            print('-'*80, '\n')

            preds = eclf.predict(test_X)

            print('Results on cross-validation set:')
            print(classification_report(test_y, preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(test_y, preds, average='micro'), '\n')

            last_classifier = eclf

            winsound.Beep(frequency, duration)

            print('='*80, '\n')

        elif inp == 'mult':
            clf = multNN(train_X, train_y, test_X, test_y)

            last_classifier = clf

            print('='*80, '\n')

        else:
            print('Want to submit last classifier?')
            inp = input('(y/n): ').lower()
            if inp == 'y':
                submit(last_classifier)
                return
            else:
                return

        




main()




# logistic: 20,20 // 40,40 // 20,40 // 20,40,80 // 30,30,30 // 30,30,30,30
#     relu: 15,15,15,15,15 // 30,35,35,40,35,35,30



















