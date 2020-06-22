
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, preprocessing
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import winsound

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


frequency = 165  # Set Frequency To 165 Hertz
duration = 750  # Set Duration To 750 ms == 0.75 second

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

def submit(classifier):
    
    # Imports the testset.
    X = pd.read_csv('test_values.csv', index_col = 0).to_numpy()

    # Preprocesses data (one hot or ordinal encoding)
    X = preProcess(X, 'OneHotEncoding')

    # Performs feature scaling on selected columns
    X = scaleCols(X, [0,1,2,4,5,6,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])

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
    X = scaleCols(X, [0,1,2,4,5,6,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])

    print("Do you want to visualize dataset with PCA?\n")
    pca_inp = input('(y/n): ').lower()
    print()
    if pca_inp == 'y':

        size = int(input("Define amount of samples to perform PCA on: "))
        
        pca = PCA(n_components = 2)
        pca.fit(X[:size])
        
        PCX = pca.transform(X)

        plt.scatter(PCX[:size,0], PCX[:size,1], c = y[:size])
        plt.show()
        print()
    else:
        print()
        
    print("Use full set?\n")
    full = input('(y/n): ')
    print()

    if full == 'y':
        # Create training set and test set
        
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size = 0.2, random_state=42)
        
    else:
        # (Optional) Reduced set
        
        k = int(input("Set size multipliyer k (set size = 100 x k): "))
        print()

        if k > 2084:
            k = 2084
        
        m = 100*k
        n = 20*k
        
        train_X = X[:m]
        train_y = y[:m]
        test_X = X[m:m+n]
        test_y = y[m:m+n]

    # C parameter (Large C = low bias // Small C = high bias) (SVM)
    C = 2

    # Test values of C?
    test_C = 0

    last_classifier = None






    # 'Front-end'
    while True:
    
        print("Choose the training model: ")
        print(' - SVM\n - Network\n - Boost\n')
        inp = input('>> ').lower()
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

            inp2 = [int(x) for x in input("Define network architecture: ").replace(' ', '').split(',')]
            print()

            alpha = input("Define regularization term alpha (default = 0.0001): ")
            if alpha == '':
                alpha = 0.0001
            print()

            eps = input("Define stability term epsilon (default = 1e-8): ")
            if eps == '':
                eps = 0.00000001
            print()

            activation = input("Define activation function (default = 'relu'): ")
            if activation == '':
                activation = 'relu'
            print()

            l_rate = input("Define initial learning rate (default = 0.001): ")
            if l_rate == '':
                l_rate = 0.001
            print()

            solver = 'adam'
            decay = input("Define learning rate decay (default = 'constant'): ")
            if decay == '':
                decay = 'constant'
            print()

            beta_1 = input("Define beta 1 (default = 0.999): ")
            if beta_1 == '':
                beta_1 = 0.999
            print()

            beta_2 = input("Define beta 2 (default = 0.999): ")
            if beta_2 == '':
                beta_2 = 0.999
            print()

            print("Want early stopping (default = False)?")
            early = input('(y/n): ').lower()
            if early == 'y':
                early = True
            else:
                early = False
            print()

            print("Want warm start (default = False)?")
            warm = input('(y/n): ').lower()
            if early == 'y':
                warm = True
            else:
                warm = False
            print()

            num = input("Define number of iterations without change, to declare convergence (default = 10): ")
            if num == '':
                num = 10
            print()

            
            tol = input("Finally, define the tolerance (default = 0.0001): ")
            if tol == '':
                tol = 0.0001
            print()
            
            print('-- Training neural network --\n')

            mlp = MLPClassifier(
                hidden_layer_sizes = inp2, max_iter = inp, alpha = float(alpha), activation = activation,
                learning_rate = decay, learning_rate_init = float(l_rate), verbose = True,
                early_stopping = early, epsilon = float(eps), validation_fraction = 0.2, solver = solver,
                beta_1 = float(beta_1), beta_2 = float(beta_2), warm_start = warm, tol = float(tol), n_iter_no_change = int(num))

            mlp.fit(train_X, train_y)

            print()
            
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

        elif inp == 'boost':
            print("Define AdaBoost base estimator (default = Decision Tree): ")
            print("1 - SVM\n2 - Log Reg\n3 - MLP\n4 - Default\n")
            est = input('>> ')
            print()

            algo = 'SAMME.R'
            if est == '1':
                est = svm.SVC(decision_function_shape='ovr', C = C)
                algo = 'SAMME'
            elif est == '2':
                max_iter = input("Define number of iterations: ")
                print()
                est = LogisticRegression(max_iter = int(max_iter), verbose = 1, C = 0.001)
            elif est == '3':
                est = customMLPClassifer(hidden_layer_sizes = (30,35,35,40,35,35,30), max_iter = 10000, alpha = 0.1, verbose = True,
                                         warm_start = True)
                algo = 'SAMME'
            else:
                est = None

            n_est = input("Define number of estimators (default = 50): ")
            if n_est == '':
                n_est = 50
            print()
            
            l_rate = input("Define learning rate (default = 1): ")
            if l_rate == '':
                l_rate = 1
            print()

            print("-- Training AdaBoost --\n")
            
            ABC = AdaBoostClassifier(base_estimator = est, n_estimators = int(n_est), learning_rate = float(l_rate),
                                     algorithm = algo)
            
            ABC.fit(train_X, train_y)

            print()

            t_preds = ABC.predict(train_X)

            print('Results on training set:')
            print(classification_report(train_y, t_preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(train_y, t_preds, average='micro'), '\n')

            print('-'*80, '\n')

            preds = ABC.predict(test_X)

            print('Results on cross-validation set:')
            print(classification_report(test_y, preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(test_y, preds, average='micro'), '\n')

            last_classifier = ABC

            winsound.Beep(frequency, duration)
            

            print('='*80, '\n')
            
            

        else:
            print('Want to submit?')
            inp = input('(y/n): ').lower()
            print()
            if inp == 'y':
                print('Want to fit for entire dataset?')
                inp = input('(y/n): ').lower()
                print()
                if inp == 'y':
                    last_classifier.fit(X, y)
                submit(last_classifier)
                return
            else:
                pass
        




main()


# logistic: 20,20 // 40,40 // 20,40 // 20,40,80 // 30,30,30 // 30,30,30,30
#     relu: 15,15,15,15,15 // 30,35,35,40,35,35,30
# having lower alpha is good...


















