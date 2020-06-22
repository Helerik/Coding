# Standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SkLearn
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier

# LightGBM
import lightgbm as lgb

# Hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt.fmin import generate_trials_to_calculate

# Others
import winsound
import warnings

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
        X = oneHotEnc(X, 18, ['a', 'r', 'v', 'w'])

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

    # Performs feature scaling on selected columns
    X = scaleCols(X, [0,1,2,3,4,5,6])
        
    return X

def submit(classifier,):

    print('-- Submitting --\n')
    
    # Imports the testset.
    X = pd.read_csv('test_values.csv', index_col = 0).to_numpy()

    # Preprocesses data (one hot or ordinal encoding)
    X = preProcess(X, 'OneHotEncoding')

    y = classifier.predict(X)

    form = pd.read_csv('submission_format.csv', index_col = 0)
    form['damage_grade'] = y.astype(int)

    # Saves predictions to submission format
    form.to_csv('submission_format.csv')

def main():

    # Imports the dataset and labels and turns them into numpy arrays.
    X = pd.read_csv('train_values.csv', index_col = 0).to_numpy()
    y = np.array(pd.read_csv('train_labels.csv', index_col = 0).to_numpy().T[0])
    
    # Preprocesses data (one hot or ordinal encoding)
    X = preProcess(X, 'OneHotEncoding')

    # Performs PCA on dataset for better visualizaion
    print("Do you want to visualize dataset with PCA?\n")
    pca_inp = input('(y/n): ').lower()

    if pca_inp == 'y':

        size = int(input("Define amount of samples to visualize with PCA: "))
        
        pca = PCA(n_components = 2)
        pca.fit(X)
        
        PCX = pca.transform(X)

        plt.scatter(PCX[:size,0], PCX[:size,1], c = y[:size])
        plt.show()
        print()
    else:
        print()

    # Option to use entire dataset or a reduced set with a more balanced amount of each label.
    print("Use reduced set?\n")
    full = input('(y/n): ')
    print()

    # Create training set and test set
    if full == 'y':
                
        X1 = [X[i] for i in range(len(X)) if y[i] == 1]
        X2 = [X[i] for i in range(len(X)) if y[i] == 2]
        X3 = [X[i] for i in range(len(X)) if y[i] == 3]

        y1 = [y[i] for i in range(len(y)) if y[i] == 1]
        y2 = [y[i] for i in range(len(y)) if y[i] == 2]
        y3 = [y[i] for i in range(len(y)) if y[i] == 3]

        size = min(len(X1),len(X2),len(X3))

        Xp = np.concatenate((X1[:size], X2[:size], X3[:size]))
        yp = np.concatenate((y1[:size], y2[:size], y3[:size]))

        train_X, test_X, train_y, test_y = train_test_split(
            Xp, yp, test_size = 0.2)
        
    else:
      
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size = 0.2, random_state=42)


    # Initializes last_classifier variable.
    last_classifier = None

    # 'Front-end'
    while True:
    
        print("Choose the training model: ")
        print(' - Network\n - GBM\n - LGBM\n - GridSearch\n - Hyperopt\n')
        inp = input('>> ').lower()
        print()

################################################################################################################################################################
        
        if inp == 'network':

            print('-- SkLearn MLP Classifier (neural network) --\n')

            it = int(input('Define maximum number of iterations: '))
            print()

            layers = [int(x) for x in input("Define network architecture: ").replace(' ', '').split(',')]
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
            
            print('\n-- Training neural network --\n')

            mlp = MLPClassifier(
                hidden_layer_sizes = layers, max_iter = it, alpha = float(alpha), activation = activation,
                learning_rate = decay, learning_rate_init = float(l_rate), verbose = True,
                early_stopping = early, epsilon = float(eps), validation_fraction = 0.2, solver = solver,
                beta_1 = float(beta_1), beta_2 = float(beta_2), warm_start = warm, tol = float(tol), n_iter_no_change = int(num))

            mlp.fit(train_X, train_y)

            print()
            
            preds = mlp.predict(train_X)


            print('Results on training set:')
            print(classification_report(train_y, preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(train_y, preds, average='micro'), '\n')

            print('-'*80, '\n')

            preds = mlp.predict(test_X)

            print('Results on cross-validation set:')
            print(classification_report(test_y, preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(test_y, preds, average='micro'), '\n')

            last_classifier = mlp

            winsound.Beep(frequency, duration)

            print('='*80, '\n')

################################################################################################################################################################

        elif inp == 'gbm':

            print('-- SkLearn GBM Classifier --\n')

            learning_rate = input("Define learning rate (default = 0.1): ")
            if learning_rate == '':
                learning_rate = 0.1
            print()

            n_estimators = input("Define number of estimators (default = 100): ")
            if n_estimators == '':
                n_estimators = 100
            print()

            subsample = input("Define subsample percentage (default = 100 %): ")
            if subsample == '' or float(subsample) > 100:
                subsample = 100
            print()

            min_samples_split = input("Define minimum number of samples to split node (default = 2): ")
            if min_samples_split == '':
                min_samples_split = 2
            print()

            min_samples_leaf = input("Define minimum number of samples to make node a leaf (default = 1): ")
            if min_samples_leaf == '':
                min_samples_leaf = 1
            print()

            max_depth = input("Define maximum depth of individual estimators (default = 3): ")
            if max_depth == '':
                max_depth = 3
            print()

            print("Do you want cross-validation for early-stopping?")
            n_iter_no_change = input("(y/n): ")
            if n_iter_no_change == 'y':
                n_iter_no_change = 15
            else:
                n_iter_no_change = None
            print("\n-- Training GBM --\n")

            gbm = GradientBoostingClassifier(
                learning_rate = float(learning_rate), n_estimators = int(n_estimators), subsample = float(subsample)/100,
                min_samples_split = int(min_samples_split), min_samples_leaf = int(min_samples_leaf), max_depth = int(max_depth),
                verbose = True, n_iter_no_change = n_iter_no_change)

            gbm.fit(train_X, train_y)

            preds = gbm.predict(train_X)

            print('Results on training set:')
            print(classification_report(train_y, preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(train_y, preds, average='micro'), '\n')

            print('-'*80, '\n')

            preds = gbm.predict(test_X)

            print('Results on cross-validation set:')
            print(classification_report(test_y, preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(test_y, preds, average='micro'), '\n')

            last_classifier = gbm

            winsound.Beep(frequency, duration)

            print('='*80, '\n')

################################################################################################################################################################

        elif inp == 'lgbm':

            print('-- SkLearn LightGBM API --\n')

            max_depth = input('Define max depth of tree (default = -1): ')
            if max_depth == '':
                max_depth = -1
            print()

            min_data_in_leaf = input('Define min data in leaf (default = 20): ')
            if min_data_in_leaf == '':
                min_data_in_leaf = 20
            print()

            feature_fraction = input('Define feature fraction for random subsampling (default = 1): ')
            if feature_fraction == '' or not (0 < float(feature_fraction) <= 1):
                feature_fraction = 1
            print()

            bagging_freq = 0
            bagging_fraction = input('Define bagging fraction for random sampling (default = 1): ')
            if bagging_fraction == '' or not (0 < float(bagging_fraction) <= 1):
                bagging_fraction = 1
            print()
            if float(bagging_fraction) < 1:
                bagging_freq = input('Define frequecy for bagging (default = 0): ')
                if bagging_freq == '':
                    bagging_freq = 0
                print()

            alpha = input('Define regularization alpha (default = 0): ')
            if alpha == '':
                alpha = 0
            print()

            lamb = input('Define regularization lambda (default = 0): ')
            if lamb == '':
                lamb = 0
            print()

            min_gain_to_split = input('Define min gain to split node (default = 0): ')
            if min_gain_to_split == '':
                min_gain_to_split = 0
            print()

            objective = input('Define objective (default = softmax)\n\nOptions: \n- num_class\n- softmax\n- ovr\n\n>> ').lower()
            if objective == '':
                objective = 'softmax'
            print()

            num_boost_round = input('Define number of iterations (default = 100): ')
            if num_boost_round == '':
                num_boost_round = 100
            print()

            l_rate = input('Define learning rate (default = 0.1): ')
            if l_rate == '':
                l_rate = 0.1
            print()

            num_leaves = input('Define max number of leaves in a tree (default = 31): ')
            if num_leaves == '':
                num_leaves = 31
            print()

            max_bin = input('Define max number of bins (default = 255): ')
            if max_bin == '':
                max_bin = 255
            print()

            min_sum_hessian_in_leaf = input('Define min hessian in leaf (default = 0.001): ')
            if min_sum_hessian_in_leaf == '':
                min_sum_hessian_in_leaf = 0.001
            print()

            print('Compensate for unbalanced dataset?')
            is_unbalance = input('(y/n): ').lower()
            if is_unbalance == 'y':
                is_unbalance = True
            else:
                is_unbalance = False
            print()

            print('-- Training Light GBM --\n')
            
            gbm = lgb.LGBMClassifier(max_depth = int(max_depth),
                min_data_in_leaf = int(min_data_in_leaf),
                feature_fraction = float(feature_fraction),
                bagging_fraction = float(bagging_fraction),
                bagging_freq = int(bagging_freq),
                lambda_l1 = float(alpha),
                lambda_l2 = float(lamb),
                min_gain_to_split = float(min_gain_to_split),
                objective = objective,
                num_boost_round = int(num_boost_round),
                learning_rate = float(l_rate),
                num_leaves = int(num_leaves),
                gpu_use_dp = True,
                num_threads = 2,
                num_class = 3,
                is_unbalance = is_unbalance,
                verbosity = 10,
                max_bin = int(max_bin),
                min_sum_hessian_in_leaf = float(min_sum_hessian_in_leaf),
                )
            
            warnings.filterwarnings("ignore")
            gbm.fit(train_X, train_y)
            warnings.filterwarnings("default")

            preds = gbm.predict(train_X)

            print('Results on training set:')
            print(classification_report(train_y, preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(train_y, preds, average='micro'), '\n')

            print('-'*80, '\n')

            preds = gbm.predict(test_X)

            print('Results on cross-validation set:')
            print(classification_report(test_y, preds, zero_division = 1))
            print('Micro averaged F1 Score: ', f1_score(test_y, preds, average='micro'), '\n')

            last_classifier = gbm

            winsound.Beep(frequency, duration)

            print('='*80, '\n')

################################################################################################################################################################

        elif inp == 'gridsearch':

            print('-- Random GridSearchCV on LightGBM classifier --\n')

            iters = input('Define number of iterations (default = 10): ')
            if iters == '':
                iters = 10
            print()

            cv = input('Define number of CV (default = 2): ')
            if cv == '':
                cv = 2
            print()
            
            param_distributions = {
                'max_depth': [-1,10,30,60,100,200],
                'min_data_in_leaf': [4,8,16,32,64,128],
                'feature_fraction': [0.1,0.25,0.5,0.75,1],
                'bagging_fraction': [0.1,0.25,0.5,0.75,1],
                'bagging_freq': [0,2,8,32,128],
                'lambda_l1': [0,0.1,1,2,4,175,512,1000],
                'lambda_l2': [0,0.1,1,2,4,175,512,1000],
                'min_gain_to_split': [0,0.01,0.03,0.1,0.3,0.5,0.9,1],
                'num_boost_round': [100, 500, 1000, 2500, 5000, 10000, 15000, 20000],
                'learning_rate': [0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
                'num_leaves': [10, 31, 62, 124, 200, 500, 750, 1000],
                'objective': ['softmax'],
                'gpu_use_dp': [True],
                'num_threads': [1],
                'num_class': [3],
                'max_bin': [128, 256, 512, 1024, 2048, 3000, 4000, 5000, 6000]
                }

            print('-- Finding best parameters --\n')

            rSearch = RandomizedSearchCV(estimator = lgb.LGBMClassifier(), param_distributions = param_distributions, scoring = 'f1_micro', n_jobs = 2,
                                         cv = int(cv), verbose = 10, n_iter = int(iters))

            warnings.filterwarnings("ignore")
            rSearch.fit(train_X, train_y)
            warnings.filterwarnings("default")

            print()

            print('Best parameters:')
            print(rSearch.best_params_, '\n')

            print('Score:')
            print(rSearch.best_score_, '\n')
            
            winsound.Beep(frequency, duration)

            print('='*80, '\n')

################################################################################################################################################################

        elif inp == 'hyperopt':
            
            print('-- Bayesian Optimization on LightGBM classifier (with Hyperopt) --\n')

            iters = input('Define number of evaluations (default = 50): ')
            if iters == '':
                iters = 50
            print()
            
            def objective_fun(space):

                print()

                warnings.filterwarnings('ignore')
                model = lgb.LGBMClassifier(**space)
                accuracy = cross_val_score(model, X, y, cv = 3, scoring = 'f1_micro').mean()
                warnings.filterwarnings('default')

                return {'loss': - accuracy, 'status': STATUS_OK}

            param_space = {
                'max_depth': 0,
                'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 1, 5000, 1)),
                'feature_fraction': hp.uniform('feature_fraction', 0, 1),
                'bagging_fraction': hp.uniform('bagging_fraction', 0, 1),
                'bagging_freq': scope.int(hp.quniform('bagging_freq', 0, 100 ,1)),
                'lambda_l1': hp.uniform('lambda_l1', 0, 10000),
                'lambda_l2': hp.uniform('lambda_l2', 0, 10000),
                'min_gain_to_split': hp.uniform('min_gain_to_split', 0, 11),
                'num_boost_round': scope.int(hp.quniform('num_boost_round', 100, 20000, 1)),
                'learning_rate': hp.uniform('learning_rate', 0.000001, 1),
                'num_leaves': scope.int(hp.quniform('num_leaves', 2, 2000, 1)),
                'objective': 'softmax',
                'gpu_use_dp': True,
                'num_threads': 2,
                'num_class': 3,
                'max_bin': scope.int(hp.quniform('max_bin', 32, 4096, 1)),
                'min_sum_hessian_in_leaf': hp.uniform('min_sum_hessian_in_leaf', 0, 5)
                }

            param_init_trials = {
                'max_depth': 0,
                'min_data_in_leaf': 40,
                'feature_fraction': 0.5,
                'bagging_fraction': 0.9,
                'bagging_freq': 1,
                'lambda_l1': 0,
                'lambda_l2': 10,
                'min_gain_to_split': 0,
                'num_boost_round': 10000,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'objective': 'softmax',
                'gpu_use_dp': True,
                'num_threads': 2,
                'num_class': 3,
                'max_bin': 256,
                'min_sum_hessian_in_leaf': 0.1
                }

            trials = generate_trials_to_calculate([param_init_trials])
            best = fmin(fn = objective_fun, space = param_space,
                        algo = tpe.suggest, max_evals = int(iters),
                        trials = trials)

            print()
            print(best, '\n')
            input()

            


################################################################################################################################################################

        elif last_classifier != None:

            # Decides if wants to submit
            print('Want to submit?')
            inp = input('(y/n): ').lower()
            print()
            if inp == 'y':

                # Decides if wants to fit classifier again for the entirety of the dataset
                print('Want to fit for entire dataset?')
                inp = input('(y/n): ').lower()
                print()
                if inp == 'y':
                    print('-- Training the last classifier --\n')
                    warnings.filterwarnings("ignore")
                    last_classifier.fit(X, y)
                    warnings.filterwarnings("default")
                    print()
                submit(last_classifier)
                return
            else:
                pass

main()


'''

    Best of neural network: 10000 , (30,35,35,40,35,35,30), 0.00001, 0.00000001, relu, 0.001, constant, 0.999, 0.999, False, False
    f1_micro ~= 0.685

    Best of GBM: 0.1-0.2, 1000-2000, 25-100, 4, 3, 5, False
    f1_micro ~= 0.690-0.720

    Best of LGBM: 0.1-0.2, 1000-3000, 90-100, 15-25, -1, 25-100, 0, 0-100
    f1_micro ~= 0.735-0.745
    

Define min data in leaf (default = 20): 40

Define feature fraction for random subsampling (default = 1): 0.5

Define bagging fraction for random sampling (default = 1): 0.8

Define frequecy for bagging (default = 0): 20

Define regularization alpha (default = 0): 

Define regularization lambda (default = 0): 300

Define min gain to split node (default = 0): 

Define objective (default = softmax)

Options: 
- num_class
- softmax
- ovr

>> 

Define number of iterations (default = 100): 5000

Define learning rate (default = 0.1): 0.2

Define max number of leaves in a tree (default = 31): 200

Define max number of bins (default = 255): 512

Compensate for unbalanced dataset?
(y/n): 

''' 



















