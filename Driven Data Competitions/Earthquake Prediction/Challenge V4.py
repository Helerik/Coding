# Standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SkLearn
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
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

def evolveSelect(X, y, parameter_space, generations, population_size, change_size, cv, common_ancestor = None):
    
    population_size = int(population_size)
    if population_size % 2 != 0:
        population_size += 1
    
    param = parameter_space.copy()
    population = []
    parameters = []

    if change_size < 0:
        change_size = 0

    if common_ancestor == None:
    
        for i in range(population_size):
            for k in parameter_space.keys():
                param.update({k: np.random.choice(parameter_space[k])})
            parameters.append(param)
            warnings.filterwarnings('ignore')
            population.append(lgb.LGBMClassifier(**param))
            warnings.filterwarnings('default')
    
    else:
        warnings.filterwarnings('ignore')
        population.append(lgb.LGBMClassifier(**common_ancestor))
        warnings.filterwarnings('default')
        parameters.append(common_ancestor)
        for i in range(population_size-1):
            for k in common_ancestor.keys():
                if len(parameter_space[k]) == 1:
                    new_param = common_ancestor[k]
                else:
                    typ = type(common_ancestor[k])
                    if typ == int:
                        new_param = int(np.random.normal(common_ancestor[k], change_size))
                        while not (parameter_space[k][0] <= int(new_param) <= parameter_space[k][-1]):
                            new_param = int(np.random.normal(common_ancestor[k], change_size))
                    elif typ == float:
                        new_param = np.random.normal(common_ancestor[k], change_size)
                        while not (parameter_space[k][0] <= new_param <= parameter_space[k][-1]):
                            new_param = np.random.normal(common_ancestor[k], change_size)
                param.update({k: new_param})
            parameters.append(param)
            warnings.filterwarnings('ignore')
            population.append(lgb.LGBMClassifier(**param))
            warnings.filterwarnings('default')

    for i in range(generations):

        print('\nIter:', i+1)

        scores = []
        for clf in population:
            warnings.filterwarnings('ignore')
            accuracy = cross_val_score(clf, X, y, cv = cv, scoring = 'f1_micro').mean()
            warnings.filterwarnings('default')
            print('\nTrained classifier with an accuracy of:', accuracy)
            scores.append(accuracy)

        val = len(population)
        while val > population_size/2:
            population.pop(scores.index(np.min(scores)))
            parameters.pop(scores.index(np.min(scores)))
            scores.pop(scores.index(np.min(scores)))
            val = len(population)

        new_pop = []
        for j in range(len(population)):
            tmp_parameters = parameters[j].copy()
            for k in tmp_parameters.keys():
                if len(parameter_space[k]) == 1:
                    new_param = parameters[j][k]
                else:
                    typ = type(parameters[j][k])
                    if typ == int:
                        new_param = int(np.random.normal(parameters[j][k], change_size))
                        while not (parameter_space[k][0] <= int(new_param) <= parameter_space[k][-1]):
                            new_param = int(np.random.normal(parameters[j][k], change_size))
                    elif typ == float:
                        new_param = np.random.normal(parameters[j][k], change_size)
                        while not (parameter_space[k][0] <= new_param <= parameter_space[k][-1]):
                            new_param = np.random.normal(parameters[j][k], change_size)
                    
                parameters[j].update({k: new_param})
            warnings.filterwarnings('ignore')
            new_pop.append(lgb.LGBMClassifier(**parameters[j]))
            warnings.filterwarnings('default')
            parameters.append(parameters[j])
        population += new_pop

    for clf in population:
        warnings.filterwarnings('ignore')
        accuracy = cross_val_score(clf, X, y, cv = cv, scoring = 'f1_micro').mean()
        warnings.filterwarnings('default')
        scores.append(accuracy)

    best_score = np.max(scores)
    best = population[scores.index(best_score)]
    best_parameters = parameters[scores.index(best_score)]
   
    print('Best classifier score:', best_score)
    print('Used parameters:', best_parameters)

    return  

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
        print(' - LGBM\n - GridSearch\n - Hyperopt\n - Evolve\n')
        inp = input('>> ').lower()
        print()

################################################################################################################################################################

        if inp == 'lgbm':

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
                'bagging_freq': [0,2,8,32,128,256],
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
                'max_depth': [0],
                'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 1, 5000, 1)),
                'feature_fraction': hp.uniform('feature_fraction', 0, 1),
                'bagging_fraction': hp.uniform('bagging_fraction', 0, 1),
                'bagging_freq': scope.int(hp.quniform('bagging_freq', 0, 1000 ,1)),
                'lambda_l1': hp.uniform('lambda_l1', 0, 10000),
                'lambda_l2': hp.uniform('lambda_l2', 0, 10000),
                'min_gain_to_split': hp.uniform('min_gain_to_split', 0, 1),
                'num_boost_round': scope.int(hp.quniform('num_boost_round', 100, 20000, 1)),
                'learning_rate': hp.uniform('learning_rate', 0.000001, 1),
                'num_leaves': scope.int(hp.quniform('num_leaves', 2, 2000, 1)),
                'objective': ['softmax'],
                'gpu_use_dp': [True],
                'num_threads': [2],
                'num_class': [3],
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

        elif inp == 'evolve':

            print('-- Evolutionary optimization on LightGBM --\n')

            iters = input('Define number of generations (default = 10): ')
            if iters == '':
                iters = 10
            print()

            cv = input('Define number of cross-validation sets (default = 2): ')
            if cv == '':
                cv = 2
            print()

            c_size = input('Define size of change on each iteration (default = 1): ')
            if c_size == '':
                c_size = 1
            print()

            pop_size = input('Define size of population (default = 10): ')
            if pop_size == '':
                pop_size = 10

            param_space = {
                'max_depth': [-1],
                'min_data_in_leaf': np.linspace(0, 5000, 5002, dtype = int),
                'feature_fraction': np.linspace(0, 1, 1000),
                'bagging_fraction': np.linspace(0, 1, 1000),
                'bagging_freq': np.linspace(0, 100 ,102, dtype = int),
                'lambda_l1': np.linspace(0, 10000, 100000),
                'lambda_l2': np.linspace(0, 10000, 100000),
                'min_gain_to_split': np.linspace(0, 0.9, 100),
                'num_boost_round': np.linspace(100, 20000, 19902, dtype = int),
                'learning_rate': np.linspace(0.0001, 1, 10000),
                'num_leaves': np.linspace(2, 2000, 2000, dtype = int),
                'objective': ['softmax'],
                'gpu_use_dp': [True],
                'num_threads': [2],
                'num_class': [3],
                'max_bin': np.linspace(32, 4096, 4066, dtype = int),
                'min_sum_hessian_in_leaf': np.linspace(0, 5, 100)
                }

            common_ancestor = {
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.5,
                'bagging_fraction': 0.9,
                'bagging_freq': 1,
                'lambda_l1': 0.,
                'lambda_l2': 10.,
                'min_gain_to_split': 0.,
                'num_boost_round': 2000,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'objective': 'softmax',
                'gpu_use_dp': True,
                'num_threads': 2,
                'num_class': 3,
                'max_bin': 256,
                'min_sum_hessian_in_leaf': 0.1
                }

            evolveSelect(X, y, param_space, int(iters), int(pop_size), float(c_size), int(cv), common_ancestor)
                

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




















