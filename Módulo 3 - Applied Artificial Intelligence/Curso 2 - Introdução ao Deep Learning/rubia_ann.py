# this project is prepared to use [mse] for regression and [accuracy] for classification problems
# a couple of adjusts may be needed for other metrics, not tested
# for instance, in the method runGeneration, mae should be replaced by neg_mean_absolute_error before
#    calling GridSearchCV while [mae] should still be fine to keras.compile method


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid all those experimental tf2 messages

import gc
import time
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.wrappers.scikit_learn
from tensorflow.keras import backend as K     
from tensorflow.python.keras import losses   
from tensorflow.python.keras.utils.generic_utils import has_arg
#from keras.callbacks import EarlyStopping
import sklearn.preprocessing
import sklearn.metrics as metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import scipy.stats as stats




class myKerasWrapper():

    def __init__(self, build_fn=None, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params
    
    def get_params(self, **params):  # pylint: disable=unused-argument
        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn': self.build_fn})
        return res

    def set_params(self, **params):
        self.sk_params.update(params)
        return self

    def fit(self, x, y, **kwargs):
        y = np.array(y) 
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)
        self.model = self.build_fn(**self.sk_params)
        if (losses.is_categorical_crossentropy(self.model.loss) and len(y.shape) != 2):
            y = to_categorical(y)
        fit_args = copy.deepcopy(self.filter_sk_params(models.Sequential.fit))
        fit_args.update(kwargs)
        history = self.model.fit(x, y, **fit_args)
        return history

    def filter_sk_params(self, fn, override=None):
        override = override or {}
        res = {}
        for name, value in self.sk_params.items():
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res

    def predict(self, x, **kwargs):
        kwargs = self.filter_sk_params(models.Sequential.predict_classes, kwargs)
        classes = self.model.predict_classes(x, **kwargs)
        return self.classes_[classes]

    def predict_proba(self, x, **kwargs):
        kwargs = self.filter_sk_params(models.Sequential.predict_proba, kwargs)
        probs = self.model.predict_proba(x, **kwargs)
        # check if binary classification
        if probs.shape[1] == 1:
            probs = np.hstack([1 - probs, probs])
        return probs

    def score(self, x, y, **kwargs):
        y = np.searchsorted(self.classes_, y)
        kwargs = self.filter_sk_params(models.Sequential.evaluate, kwargs)
        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        outputs = self.model.evaluate(x, y, **kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name in ['accuracy', 'acc']:
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                     'You should pass `metrics=["accuracy"]` to '
                     'the `model.compile()` method.')


class myKerasWrapperR(myKerasWrapper):

    def predict(self, x, **kwargs):
        kwargs = self.filter_sk_params(models.Sequential.predict, kwargs)
        return np.squeeze(self.model.predict(x, **kwargs))

    def score(self, x, y, **kwargs):
        kwargs = self.filter_sk_params(models.Sequential.evaluate, kwargs)
        loss = self.model.evaluate(x, y, **kwargs)
        if isinstance(loss, list):
            return -loss[0]
        return -loss


class gcCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        i = 0
        #gc.collect()


class simpler_keras():
    
    def __init__(self, X, y, mode, gpu=False, workers=4, fixed_genes={}):        
        print('\nAvailable modes: regression, binary, multi, multioutput')
        self.mode = mode
        self.gpu = gpu
        self.workers = workers
        self.genes = fixed_genes
        self.loadData(X, y)
        
    def loadData(self, X, y):
        #self.X = tf.convert_to_tensor(X)
        #self.y = tf.convert_to_tensor(y)  
        self.X = X
        self.y = y
        
    def release(self):
        tf.keras.backend.clear_session()
        gc.collect()

    def check(self):
        self.importers()  
        self.getGenes()
        print('\nAvailable genes and variations:')
        for k, v in self.genes.items():
            print('{:15s}'.format(k), v)
        print('{:15s}'.format('Mutable:'), self.mutable)
    
    def describe(self):
        print('\nNeural network and genetics set:')
        print('{:25s}'.format('X data shape:'), self.X.shape)
        print('{:25s}'.format('y data shape:'), self.y.shape)
        print('{:25s}'.format('Max topology dept:'), self.topology)
        print('{:25s}'.format('Population size:'), self.pop)
        print('{:25s}'.format('Number of generations:'), self.gens)
        print('{:25s}'.format('Proportion on strain:'), self.parent_portion)
        print('{:25s}'.format('DNA length:'), len(self.individuals[0]))
        print('{:25s}'.format('Genes available to mutate:\n'), self.mutable)
        
    def importers(self):        
        print('\nVersions:')
        print("Keras :" , keras.__version__)
        print("Tensorflow :" , tf.__version__) 
        if not self.gpu: 
            print('GPU disabled!')
            if len(tf.config.list_physical_devices('GPU')) > 0:
                print('You have at least one available GPU device not in use!')
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print('Using %d CPU workers' % self.workers)
        else:
            if len(tf.config.list_physical_devices('GPU')) < 1:
                print('GPU was not found. Using CPU instead!')
            else:
                print('GPU enabled!')
                if len(tf.config.list_physical_devices('GPU')) < self.workers:
                    self.workers = len(tf.config.list_physical_devices('GPU'))
                    print('Number of workers adjusted to fit the GPUs available')
                print('Using %d GPU workers' % self.workers)
                self.testGPU()
        
    def testGPU(self):
        cpu_slot, gpu_slot = 0, 0
        with tf.device('/CPU:' + str(cpu_slot)):
            start = time.time()
            tf.eye(10000,10000)
            end = time.time() - start
            print('CPU test for EYE(10000): ', end)
        with tf.device('/GPU:' + str(cpu_slot)):
            start = time.time()
            tf.eye(10000,10000)
            end = time.time() - start
            print('GPU test for EYE(10000): ', end)

    def getGenes(self):
        if not 'epochs' in self.genes.keys():
            self.genes.update({'epochs': [10 * i for i in range(1, 15)]})
        if not 'batch_size' in self.genes.keys():
            self.genes.update({'batch_size': [2 ** i for i in range(8)]})
        if not 'activators' in self.genes.keys():
            reg_activation = ['elu', 'relu', 'linear']
            clf_activation = ['sigmoid', 'softmax', 'elu', 'relu']
            self.genes.update({'activators': reg_activation if self.mode=='regression' else clf_activation})
        if not 'last_activation' in self.genes.keys():
            reg_activation = ['relu', 'linear']
            clf_activation = ['sigmoid', 'softmax']
            self.genes.update({'last_activation': reg_activation if self.mode=='regression' else clf_activation})
        if not 'losses' in self.genes.keys():
            reg_loss = ['mean_squared_error', 'mean_absolute_error', 'hinge', 'logcosh']
            clf_binary_loss = ['mean_squared_error', 'mean_absolute_error',
                                  'binary_crossentropy', 'categorical_hinge', 'hinge', 'logcosh']
            clf_multi_loss = ['sparse_categorical_crossentropy', 'categorical_hinge', 'hinge', 'logcosh']
            clf_multioutput_loss = ['mean_squared_error', 'mean_absolute_error', 
                                   'binary_crossentropy', 'categorical_crossentropy',
                                   'categorical_hinge', 'hinge', 'logcosh']
            loss = reg_loss if self.mode=='regression' else clf_binary_loss if self.mode=='binary' else clf_multi if self.mode=='multi' else clf_multioutput_loss
            self.genes.update({'losses': loss})
        if not 'optimizers' in self.genes.keys():
            self.genes.update({'optimizers': ['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta']})
        if not 'denses' in self.genes.keys():
            self.genes.update({'denses': [0] + [2 ** i for i in range(0, 7)]})
        if not 'dropout' in self.genes.keys():
            self.genes.update({'dropout': [True, False]})
        if not 'dropout_rate' in self.genes.keys():
            self.genes.update({'dropout_rate': [i/10 for i in range(1, 5)]})
        self.mutable = [key for key in self.genes.keys() if len(self.genes[key]) > 1]

    def sampler(self, key):
        return random.sample(self.genes[key], 1)[0]

    def crossover(self, parents):
        p1 = self.individuals[parents[0]]
        p2 = self.individuals[parents[1]]
        #print(p1)
        #print(p2)
        child = {}
        for k, v in p1.items():
            child.update({k: p1[k] if random.random() > 0.5 else p2[k]})
        #print(child)
        return child

    def mutate(self, individual):
        mutable = []
        for k, v in individual.items():
            if v[0] in self.mutable:
                mutable.append(k)
        mutate_on = random.sample(mutable, 1)[0]
        mutate_key = individual[mutate_on][0]
        variations = [gene for gene in self.genes[mutate_key] if gene != individual[mutate_on][1]]
        mutation = random.sample(variations, 1)[0]
        individual[mutate_on] = (mutate_key, mutation)
        return individual
    
    def setGenetic(self, input_dim, output_dim, metrics=None, topology=['Dense','Dense'], population=10, generations=3, keep_portion=0.2):
        self.topology = topology
        self.pop = population
        self.gens = generations
        self.parent_portion = keep_portion
        self.input_dim = input_dim
        self.output_dim = output_dim
        if metrics:
            self.metrics = metrics
        else:
            self.metrics = ['mse'] if self.mode == 'regression' else ['accuracy']
        self.individuals = []
        for individual in range(self.pop):
            DNA = {}
            seq = 0
            for i, layer in enumerate(topology):
                if layer == 'Dense':
                    neurons, activator = self.sampler('denses'), self.sampler('activators')
                    dropout, dropout_rate = self.sampler('dropout'), self.sampler('dropout_rate')
                    if neurons == 0 and i == 0: neurons = 1
                    DNA.update({seq: ('denses', neurons)})
                    DNA.update({seq+1: ('activators', activator)})
                    DNA.update({seq+2: ('dropout', dropout)})
                    DNA.update({seq+3: ('dropout_rate', dropout_rate)})
                    seq += 4
            activator = self.sampler('last_activation')
            DNA.update({seq: ('last_activation', activator)})
            seq += 1
            optimizer, loss = self.sampler('optimizers'), self.sampler('losses')
            DNA.update({seq: ('optimizers', optimizer)})
            DNA.update({seq+1: ('losses', loss)})
            seq += 2
            DNA.update({seq: ('epochs', self.sampler('epochs'))})
            DNA.update({seq+1: ('batch_size', self.sampler('batch_size'))})
            self.individuals.append(DNA)
            del DNA

    def prepareModel(self, **kwargs):
        individual = self.individuals[kwargs['individual']]
        seq = 0
        model = models.Sequential()
        for i, layer in enumerate(self.topology):
            if individual[seq][0] == 'denses':
                if individual[seq][1] > 0 and i == 0:
                    model.add(layers.Dense(individual[seq][1], activation=individual[seq+1][1], input_dim=self.input_dim))
                elif individual[seq][1] > 0:
                    model.add(layers.Dense(individual[seq][1], activation=individual[seq+1][1]))
                if individual[seq][1] > 0 and individual[seq+2][1]:
                    model.add(layers.Dropout(individual[seq+3][1]))
                seq += 4
        model.add(layers.Dense(self.output_dim, activation=individual[seq][1]))
        seq += 1
        model.compile(optimizer=individual[seq][1], loss=individual[seq+1][1], metrics=self.metrics)
        return model

    def runGenerations(self, cv=2, validation_split=0.2, n_mutations=2, crossover=0.5, performance_cap=0.1, verbose=0):
        self.describe()
        print('\nNeural network train starting...')
        scores = np.zeros(self.pop)
        metric = self.metrics[0]
        for i, generation in enumerate(range(self.gens)):
            start = time.time()
            print('\nRunning generation:', i+1)
            if i > 0:
                n_keep = int(self.parent_portion * self.pop)
                if n_keep < 1: n_keep = 1
                parents = np.argpartition(scores, -n_keep)[-n_keep:] # select the best candidates to mutate
                #print('Parent kept:', parents)
                for j in range(self.pop):
                    if not j in parents: 
                        if random.random() <= crossover and n_keep > 1:
                            #print('Crossing over')
                            self.individuals[j] = self.crossover(random.sample(list(parents), 2))
                        else:
                            #print('Mutating')
                            for k in range(n_mutations):
                                self.individuals[j] = self.mutate(self.individuals[j])
            DNA_len = len(self.individuals[0])
            param_grid = [{'individual': [ind], 'epochs': [individual[DNA_len-2][1]], 'batch_size': [individual[DNA_len-1][1]]} for ind, individual in enumerate(self.individuals)]
            if self.mode != 'regression':
                model = myKerasWrapper(build_fn=self.prepareModel, **{'verbose': verbose, 'mode': self.mode, 
                                                            'metrics': metric, 'validation_split': validation_split})
                cvf = KFold(n_splits=cv, shuffle=True)
            else:
                metric = metric.replace('mse','neg_mean_squared_error')
                model = myKerasWrapperR(build_fn=self.prepareModel, **{'verbose': verbose, 'mode': self.mode, 
                                                            'metrics': metric, 'validation_split': validation_split})
                cvf = StatifiedKFold(n_splits=cv, shuffle=True)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=self.workers, cv=cvf, verbose=verbose, 
                                scoring=metric, return_train_score=True)
            grid_result = grid.fit(self.X, self.y)
            scores = self.evaluate(grid_result, performance_cap)
            #print(np.array(self.individuals))
            #print('Scores:', [round(score, 2) for score in grid_result.cv_results_['mean_test_score']])
            print('Current Average Score:', round(np.array(grid_result.cv_results_['mean_test_score']).mean(), 2))
            print('Total Elapsed Time (s):', round((time.time()-start), 2))
            print('Average Time per Cycle (s):', round((time.time()-start)/self.pop, 2))
            self.release()
        print('\nModel overall performance')
        print('Best score: %.2f' % grid_result.best_score_)
        #print('Best params:', grid_result.best_params_)
        best_model_desc = self.individuals[grid_result.best_params_['individual']]
        print('Best model:', ' - '.join([(v[0]+': '+str(v[1])).replace('denses','\ndenses').replace('optim','\noptim').replace('epochs','\nepochs') for k, v in best_model_desc.items()]))
        self.best_model = self.prepareModel(**grid_result.best_params_)
        self.best_model.summary()
        epochs = best_model_desc[DNA_len-2][1]
        batch = best_model_desc[DNA_len-1][1]
        history = self.best_model.fit(self.X, self.y, validation_split=validation_split, verbose=verbose, epochs=epochs, batch_size=batch)
        self.printHistory(history.history)

    def evaluate(self, results, performance_cap):
        scaler = MinMaxScaler()
        mean_train = results.cv_results_['mean_train_score'].reshape(-1, 1)
        mean_train = scaler.fit_transform(mean_train)
        mean_test = results.cv_results_['mean_test_score'].reshape(-1, 1)
        mean_test = scaler.fit_transform(mean_test)        
        cv_train = scaler.fit_transform((results.cv_results_['std_train_score'] / results.cv_results_['mean_train_score']).reshape(-1, 1))
        cv_test = scaler.fit_transform((results.cv_results_['std_test_score'] / results.cv_results_['mean_test_score']).reshape(-1, 1))
        timing = scaler.fit_transform(results.cv_results_['mean_fit_time'].reshape(-1, 1))
        mean_score = (mean_train + mean_test) / 2
        perf_score = (0.4 * cv_train + 0.4 * cv_test + 0.2 * timing) * performance_cap
        scores = (mean_score * (1 - perf_score)).flatten()
        #print('Scores:', scores)
        params = results.cv_results_['params']
        return scores

    def printHistory(self, history):
        metric = self.metrics[0]
        print('Performance (train/test): %.2f / %.2f' % (history[metric][-1], history['val_%s'%metric][-1]))
        plt.plot(history[metric], label='Train')
        plt.plot(history['val_%s'%metric], label='Test')
        plt.xlabel('Epoch')
        plt.title('Model Performance')
        plt.legend()
        plt.show()        
        plt.plot(history['loss'], label='Train')
        plt.plot(history['val_loss'], label='Test')
        plt.xlabel('Epoch')
        plt.title('Model Loss')
        plt.legend()
        plt.show()   
        if self.mode == 'regression':
            self.printPerformanceR()
        else:
            self.printPerformanceC()   

    def calc_rss(self, residual):
        return float(((residual) ** 2).sum())         
    def calc_rmse(self, y, y_hat):
        return np.sqrt(mean_squared_error(y_hat, y))
    def calc_r2(self, y, y_hat):
        return r2_score(y_hat, y)

    def printPerformanceR(self):
        y = self.y.reshape(-1,1)
        y_hat = self.best_model.predict(self.X).reshape(-1,1)
        sample_size = len(y_hat)
        res = y - y_hat
        obs = np.arange(1, sample_size+1).reshape(-1, 1)
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        fig.suptitle('Residual Analysis', fontsize=14)
        plt.subplots_adjust(hspace=0.32, wspace=0.2)
        # residual by observation, desired behaviour: stability, stochastic
        ax[0][0].scatter(obs, res, marker='o', c= 'r', alpha=0.8, edgecolors='none')
        ax[0][0].plot(obs, res, c= 'k', lw=0.5, alpha=0.8)
        ax[0][0].plot([0, sample_size], [0, 0], c='k')
        ax[0][0].set_title('Residual vs Observation', size=14)
        # residual normality, desired behaviour: stochastic and normal distributed residual
        a, result = stats.probplot(res.ravel(), plot=ax[0][1], dist='norm')
        _, p = stats.normaltest(res)
        ax[0][1].text(0.2, 0.8, 'r=%.2f\np-value=%.4f'%(result[2], p), ha='center', va='center', transform=ax[0][1].transAxes)
        ax[0][1].set_title('Normality (pp-plot)', size=14)
        # residual over leverage, desired behaviour: homoscedastic
        ax[1][0].scatter(y_hat, res, marker='o', c= 'r', alpha=0.8, edgecolors='none')
        ax[1][0].plot([0, y_hat.max()], [0, 0], c='k')
        ax[1][0].set_title('Residual vs Fitted', size=14)
        # residual distribution, desired behaviour: normal distributed residual
        ax[1][1].hist(res, density=True, facecolor='b', alpha=0.5, edgecolor='gray')
        rv = stats.norm(res.mean(), res.std())
        x = np.linspace(res.min(), res.max(), 100) 
        h = plt.plot(x, rv.pdf(x), c='b', lw=2)
        ax[1][1].set_title('Residual Histogram', size=14)
        plt.show()
        print('RMSE: %.2f | R2: %.2f' % (self.calc_rmse(y, y_hat), self.calc_r2(y, y_hat)))

    def printPerformanceC(self):
        y_true = self.y 
        y_pred = self.best_model.predict_classes(self.X)
        report = classification_report(y_true, y_pred, output_dict=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.title('Confusion Matrix')
        #print(y_true.shape, y_true)
        #print(y_pred.shape, y_pred)
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='YlGn', fmt='d',)
        plt.xlabel('Predicted')
        plt.ylabel('True Class')
        plt.show()
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.title('Classification Report')
        sns.heatmap(pd.DataFrame(report).iloc[0:3].T, annot=True, vmin=0, vmax=1, cmap='BrBG', fmt='.2g')
        plt.xlabel('Score')
        plt.show()
        print('ACCURACY: ', round(accuracy_score(y_true, y_pred)*100, 1), '%')



# demo of the use of this simpler_keras class
run_demo = True

if run_demo:

    mode = 'binary'
    topology = ['Dense','Dense']

    if mode == 'binary':
        data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv', header=None)
        data = np.array(data)
        X, y = data[:,:-1], data[:,-1]
        scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
        scalarX.fit(X)
        X = scalarX.transform(X)
        input_dim = X.shape[1]
        output_dim = 1
        metric = ['accuracy']
    if mode == 'regression':
        data = pd.read_csv('https://raw.githubusercontent.com/advinstai/Aprendizagem-estatistica-regressao/master/exercicios/Advertising.csv')
        data = np.array(data)
        X, y = data[:,:-1], data[:,-1]
        scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
        scalarX.fit(X)
        X = scalarX.transform(X)
        input_dim = X.shape[1]
        output_dim = 1
        metric = ['mse']

    GENES = {
        'optimizers': ['rmsprop','adam'],
        'losses': ['mse'],
        'activators': ['relu','elu'],
        'last_activation': ['relu','linear'],
        'denses': [2 ** i for i in range(2, 5)],
        'dropout': [False],
        'dropout_rate': [0.1],
        'epochs': [5, 10, 20],
        'batch_size': [10, 200, 400, 800]
    }

    # gpu and fixed_genes below are optional parameters 
    start = time.time()
    k = simpler_keras(X=X, y=y, mode=mode, gpu=False, workers=10, fixed_genes=GENES)
    k.check()

    k.setGenetic(topology=topology, population=5, generations=2, keep_portion=0.5, input_dim=input_dim, output_dim=output_dim, metrics=metric)

    k.runGenerations(cv=2, validation_split=0.1, n_mutations=3, crossover=0.8, performance_cap=0.1, verbose=0)
    print('Total elapsed time (s): %.2f' % (time.time() - start))




    
       
        
        
    


# testar o wrapper para regression

# aplicar o algoritmo final no meu exercicio da aula