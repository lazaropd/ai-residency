
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['seaborn', 'ggplot', 'seaborn-white'])

import scipy
from scipy import stats
from scipy.io import arff

from itertools import combinations_with_replacement

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold

from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# regressors
from sklearn.linear_model import LinearRegression as lr, SGDRegressor as sgdr
from sklearn.linear_model import Ridge as rr, RidgeCV as rcvr, Lasso as lassor, LassoCV as lassocvr
from sklearn.neighbors import KNeighborsRegressor as knnr
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.svm import SVR as svr
from sklearn.ensemble import RandomForestRegressor as rfr, AdaBoostRegressor as abr, GradientBoostingRegressor as gbr

# classifiers
from sklearn.linear_model import SGDClassifier as sgdc, LogisticRegression as logitc, RidgeClassifier as rc
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.svm import SVC as svc
from sklearn.naive_bayes import GaussianNB as gnbc, BernoulliNB as bnbc, MultinomialNB as mnbc
from sklearn.ensemble import RandomForestClassifier as rfc, GradientBoostingClassifier as gbc

# multiclass
from sklearn.multiclass import OneVsRestClassifier as ovrc, OneVsOneClassifier as ovoc




class rubia_models:


    def __init__(self, df, width=100, debug=False):
        self.data_raw = df
        self.M = df
        self.report_width = width
        self.graph_width = 1.3 * width // 10
        self.debug = debug
        if not self.debug: # remove warnings when not running in debug mode
            import warnings
            warnings.filterwarnings("ignore")
    
    
    # check columns dtype
    def checkDtypes(self, df):
        cols_resume = []
        for col in df.columns:
            coltype = str(df[col].dtype)
            cols_resume.append('%s (%s)' % (col, coltype))
        # beware in the presence of non numeric categorical data
        print('* COLUMNS INFO: ', ', '.join(cols_resume))
        return None

    
    # show a few general info about the dataset
    def describe(self, df):
        print(self.report_width * '*', '\n*')
        print('* DATA OVERVIEW FOR THIS DATASET \n*')
        print('* DATA SHAPE: ', df.shape)
        self.checkDtypes(df)
        print('* ')
        print(self.report_width * '*')
        print('\nDATA SAMPLE: ')
        display(df.sample(5))
        print('\nSTATISTICS: ')
        display(df.describe(include='all').T)
        print('\n\n')
        return None
        

    # run a basic and repetitive EDA for a given pandas dataframe
    def explore(self, df, y_col, ig_cols, graph=False):
        X_cols = [col for col in df.columns if col not in [y_col] and col not in ig_cols]
        self.X = df.loc[:, X_cols]
        self.y = df.loc[:, [y_col]]
        self.M = df.loc[:, X_cols + [y_col]]
        print(self.report_width * '*', '\n*')
        print('* FEATURE EXTRACTION REPORT \n*')
        print('* X: ', ' | '.join(X_cols))
        print('* y: ', y_col)
        print('* M: ', (self.X.shape), '|', self.y.shape)
        #self.checkDtypes(df)
        print('* ')
        print(self.report_width * '*' + '\n')

        if graph:   
            size = self.graph_width
            # balance between every output class: pay special attention with unbalanced data
            if len(df[y_col].unique()) <= 10:
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                df[y_col].value_counts().nlargest(10).plot(kind='bar')
                plt.title('Classes Balance')
                plt.xticks(rotation=45)
                plt.show()
            # histogram for every feature: pay attention to outliers, data distribution and dimension
            COLS = 3
            ROWS = len(self.X.columns) // COLS + (1 if len(self.X.columns) % COLS != 0 else 0)
            fig, ax = plt.subplots(ROWS, COLS, figsize=(size, 4 * ROWS))
            row, col = 0, 0
            for i, feature in enumerate(self.X.columns):
                if col == (COLS - 1):
                    row += 1
                    plt.subplots_adjust(hspace=0.2, top = 0.92)
                else:
                    plt.subplots_adjust(hspace=0.2, top = 0.80)
                col = i % COLS    
                cax = ax[row, col] if ROWS > 1 else ax[col]
                if len(df[y_col].unique()) <= 10:
                    for cat in df[y_col].unique():
                        df[df[y_col]==cat][feature].hist(bins=30, alpha=0.5, edgecolor='white', ax=cax).set_title(feature)
                else:
                    df[feature].hist(bins=30, alpha=0.5, edgecolor='white', ax=cax).set_title(feature)
                plt.legend(df[y_col].unique())    
            fig.suptitle('Data Distribution', fontsize=14)
            plt.show()
            # pairplot and density plot for every column
            if len(df[y_col].unique()) <= 10:
                g = sns.pairplot(self.M, hue=y_col, plot_kws={'alpha':0.5, 's': 20})
                handles = g._legend_data.values()
                labels = g._legend_data.keys()
                g._legend.remove()
                g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3)
            else:
                g = sns.pairplot(self.M, plot_kws={'alpha':0.5, 's': 20})
            g.fig.set_figwidth(0.75 * size)
            g.fig.set_figheight(0.75 * size)
            plt.subplots_adjust(top = 0.92, bottom=0.08)
            g.fig.suptitle('Pairplot and Density Matrix', fontsize=14)
            plt.show()
            # correlation heatmap matrix
            fig, ax = plt.subplots(figsize=(0.95 * size, 0.95 * size))
            corr = self.M.corr()
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(self.M.corr(), ax=ax, mask=mask, annot = True, vmin = -1, vmax = 1, center = 0, cmap = 'RdBu_r')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.title('Correlation Matrix')
            plt.show()

        return None


    # add higher level and interaction terms to the model
    def calcTerms(self, df, cols):
        if len(cols) > 1:
            product = df[cols].product(axis=1)
        else:
            product = np.sqrt((df[cols]))
        return product
    def addTerms(self, X, y, levels=2, interaction=True, root=False):
        cols = X.columns
        if levels > 1: #higher level terms makes sense only for k > 1
            #calculating a combination of n elements in groups of k with replacement
            n, k = cols, levels
            if interaction:
                for comb in combinations_with_replacement(n,k):
                    comb = list(comb)
                    self.X['*'.join(comb)] = self.calcTerms(X, comb)
            # or just the polynomial terms if interaction = False
            else:
                for col in cols:
                    for order in range(2, k+1):
                        comb = [col for elem in range(1, order+1)]
                        self.X['*'.join(comb)] = self.calcTerms(X, comb)
            if root:
                for col in cols:
                    comb = [col]
                    self.X['sqroot_'+col] = self.calcTerms(X, comb)
        self.M = pd.concat([self.X, self.y], axis=1)
        return None

    
    # encode all non numeric features
    def encode(self):
        le = LabelEncoder()    
        for col in self.X.columns:
            if str(self.X[col].dtype) == 'object' or str(self.X[col].dtype) == 'string':
                self.X[col] = le.fit_transform(self.X[col])
        for col in self.y.columns:
            if str(self.y[col].dtype) == 'object' or str(self.y[col].dtype) == 'string':
                self.y[col] = le.fit_transform(self.y[col])
        self.M = pd.concat([self.X, self.y], axis=1)
        return None


    # analyse if this is a regression or classification problem
    def analyse(self, y_col):
        if len(self.y[y_col].unique()) > 10 or str(self.y[y_col].dtype) == 'float64':
            self.strategy = 'regression'
        else:
            self.strategy = 'classification'
        print('Problem identified as', self.strategy)
        return None


    # apply transformation to data
    def transform(self, who, transform, graph=False):
        size = self.graph_width
        if who == 'X':
            if transform == 'xnone':
                self.scalerX = None 
                self.Xt_train = self.X_train
                self.Xt_test = self.X_test
            if transform == 'xstandard' or transform == 'xminmax':
                if transform == 'xstandard':
                    self.scalerX = StandardScaler()
                if transform == 'xminmax':
                    self.scalerX = MinMaxScaler()
                self.scalerX.fit(self.X_train)
                self.Xt_train = self.scalerX.transform(self.X_train)
                self.Xt_test = self.scalerX.transform(self.X_test)
                if graph:
                    fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                    for i in range(self.Xt_train.shape[1]):
                        sns.kdeplot(self.Xt_train[:][i])
                    plt.title('X-Features after Transformation (training set)')
                    plt.show()   
        if who == 'y':
            if transform == 'ynone':
                self.scalery = None
                self.yt_train = self.y_train 
                self.yt_test = self.y_test   
            if transform == 'yboxcox':
                self.yt_train, self.scalery = stats.boxcox(self.y_train)
                self.yt_train = pd.DataFrame(self.yt_train, columns=[self.y.columns])
                self.yt_test = stats.boxcox(self.y_test, lmbda=self.scalery) 
        return None
    
    # apply regression models
    def regression(self, metric, folds=10, alphas=[], graph=False):
        size = self.graph_width

        models = {}
        models["Linear regressor"]                  = lr()
        models["Lasso regressor"]                   = lassor()
        models["Lasso CV regressor"]                = lassocvr()
        models["Ridge regressor"]                   = rr(alpha=0, normalize=True)
        models["Ridge CV regressor"]                = rcvr(alphas = alphas)
        models["K nearest neighbors regressor K2u"] = knnr(n_neighbors=2, weights='uniform')
        models["K nearest neighbors regressor K2d"] = knnr(n_neighbors=2, weights='distance')
        models["K nearest neighbors regressor K5"]  = knnr(n_neighbors=5)
        models["K nearest neighbors regressor K10"] = knnr(n_neighbors=10)
        models["SGD regressor"]                     = sgdr(max_iter=10000, warm_start=True)
        models["Decision tree regressor"]           = dtr()
        models["Decision tree regressor D3"]        = dtr(max_depth=3)
        models["Random forest regressor"]           = rfr()
        models["Ada boost regressor"]               = abr()
        models["Gradient boost regressor"]          = gbr()
        models["Support vector regressor"]          = svr()
        self.models = models

        print('\n')
        print(self.report_width * '*', '\n*')
        print('* REGRESSION RESULTS - BEFORE PARAMETERS BOOSTING \n*')
        #kf = StratifiedKFold(n_splits=folds, shuffle=True)
        kf = KFold(n_splits=folds, shuffle=True)
        results = []
        names = []
        for model_name in models:
            cv_scores = -1 * cross_val_score(models[model_name], self.Xt_train, self.yt_train.values.ravel(), cv=kf, scoring=metric)  
            results.append(cv_scores)
            names.append(model_name)
        print(self.report_width * '*', '')
        report = pd.DataFrame({'Regressor': names, 'Score': results})
        report['Score (avg)'] = report.Score.apply(lambda x: x.mean())
        report['Score (std)'] = report.Score.apply(lambda x: x.std())
        report['Score (VC)'] = 100 * report['Score (std)'] / report['Score (avg)']
        report.sort_values(by='Score (avg)', inplace=True)
        report.drop('Score', axis=1, inplace=True)
        display(report)
        print('\n')
        if graph:
            fig, ax = plt.subplots(figsize=(size, 0.5 * size))
            plt.title('Regressor Comparison')
            #ax = fig.add_subplot(111)
            plt.boxplot(results)
            ax.set_xticklabels(names)
            plt.xticks(rotation=45)
            plt.subplots_adjust(hspace=0.0)
            plt.show()             
        return None


    # apply classification models
    def classification(self, metric, folds, alphas, graph):
        size = self.graph_width

        if len(self.y.iloc[:,0].unique()) > 2:
            struct = 'multiclass'
        else:
            struct = 'binary'

        models = {}
        models["K nearest neighbors classifier K2"]  = knnc(n_neighbors=2)
        models["K nearest neighbors classifier K5"]  = knnc(n_neighbors=5)
        models["K nearest neighbors classifier K10"] = knnc(n_neighbors=10)        
        models["Decision tree classifier"]           = dtc()
        models["SVM classifier with RBF kernel"]     = svc(gamma='scale')
        models["SVM classifier with linear kernel"]  = svc(kernel='linear')
        models["Gaussian naive bayes"]               = gnbc()
        models["Bernoulli naive bayes"]              = bnbc(binarize=0.5)
        models["Multinomial naive bayes"]            = mnbc()
        models["SGD classifier"]                     = sgdc(max_iter=10000)
        models["Ridge classifier"]                   = rc()
        models["Random forest classifier"]           = rfc(n_estimators=100)
        models["Gradient boosting classifier"]       = gbc()

        if struct == 'multiclass':
            models["Logistic classifier multinomial"]= logitc(multi_class='multinomial', solver='lbfgs')
            models["Logistic classifier auto"]       = logitc(multi_class='auto')
            models["Logistic One vs Rest"]           = ovrc(logitc())
            models["Logistic One vs One"]            = ovoc(logitc())

        if struct == 'binary':
            models["Logistic classifier"]            = logitc(max_iter=2000)

        self.models = models

        print('\n')
        print(self.report_width * '*', '\n*')
        print('* CLASSIFICATION RESULTS - BEFORE PARAMETERS BOOSTING \n*')
        kf = StratifiedKFold(n_splits=folds, shuffle=True)
        results = []
        names = []
        for model_name in models:
            cv_scores = cross_val_score(models[model_name], self.Xt_train, self.yt_train.values.ravel(), cv=kf, scoring=metric, error_score=np.nan)  
            results.append(cv_scores)
            names.append(model_name)
        print(self.report_width * '*', '')
        report = pd.DataFrame({'Classifier': names, 'Score': results})
        report['Score (avg)'] = report.Score.apply(lambda x: x.mean())
        report['Score (std)'] = report.Score.apply(lambda x: x.std())
        report['Score (VC)'] = 100 * report['Score (std)'] / report['Score (avg)']
        report.sort_values(by='Score (avg)', inplace=True, ascending=False)
        report.drop('Score', axis=1, inplace=True)
        display(report)
        print('\n')
        if graph:
            fig, ax = plt.subplots(figsize=(size, 0.5 * size))
            plt.title('Classifier Comparison')
            #ax = fig.add_subplot(111)
            plt.boxplot(results)
            ax.set_xticklabels(names)
            plt.xticks(rotation=45)
            plt.subplots_adjust(hspace=0.0)
            plt.show()             
        return None

    
    # residual analysis for regression problems
    def calc_rss(self, residual):
        return float(((residual) ** 2).sum())         
    def calc_r2(self, y, y_hat):
        return r2_score(y_hat, y)
    def residual(self, y, y_hat, model_name, graph=False):
        size = self.graph_width
        # do some data conversion because of different methods data types
        sample_size = len(y_hat)
        if isinstance(y, pd.DataFrame): 
            y = np.array(y.values.ravel())
        y = y.reshape(-1, 1)
        y_hat = np.array(y_hat).reshape(-1, 1)
        res = y - y_hat
        obs = np.arange(1, sample_size+1).reshape(-1, 1)

        print('\n')
        print(self.report_width * '*', '\n*')
        print('* MODEL PERFORMANCE \n*')
        print('* MODEL NAME: ', model_name)
        print('* TEST SAMPLE SIZE: ', sample_size)
        print('* RSS: %.2f'%self.calc_rss(res))
        print('* R2: %.2f'%self.calc_r2(y, y_hat))
        print('* ')
        print(self.report_width * '*', '\n')

        if graph:
            fig, ax = plt.subplots(2, 2, figsize=(size, 0.5 * size))
            fig.suptitle('Residual Analysis', fontsize=14)
            plt.subplots_adjust(hspace=0.32, wspace=0.2)
            # residual by observation, desired behaviour: stability, stochastic
            ax[0][0].scatter(obs, res, marker='o', c= 'r', alpha=0.8, edgecolors='none')
            ax[0][0].plot(obs, res, c= 'k', lw=0.5, alpha=0.8)
            ax[0][0].plot([0, sample_size], [0, 0], c='k')
            ax[0][0].set_title('Residual', size=14)
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
        return None


    # evaluate some models
    def evaluate(self, test_size=0.2, transformX='xnone', transformY='ynone', folds=10, alphas=[], graph=False, metric=''):
        if self.strategy == 'regression':
            if metric == '': metric = 'neg_mean_squared_error'
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=True)
            # transform data
            self.transform('X', transformX, graph) #model transf for X_train
            self.transform('y', transformY, graph) #model transf for y_train
            self.regression(metric, folds, alphas, graph)
        else:
            if metric == '': metric = 'accuracy'
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=True, stratify=self.y)
            # transform data
            self.transform('X', transformX, graph) #model transf for X_train
            self.transform('y', transformY, graph) #model transf for y_train
            self.classification(metric, folds, alphas, graph)
        return None


    # given a model name, evaluate y_hat/y_pred and the overall performance of such model
    def test(self, model_name, graph=False):
        size = self.graph_width
        model = self.models[model_name]
        # fit using the train subset
        X, y = self.Xt_train, self.yt_train
        model.fit(X, y)

        # evaluate using the test subset
        X, y = self.Xt_test, self.yt_test
        
        if self.strategy == 'regression':
            y_hat = model.predict(X)
            # show residual analysis
            self.residual(y, y_hat, model_name, graph)
            if graph:
                # show the correlation between y and y_hat
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Model Overall Performance')
                plt.scatter(y, y_hat, color='g')
                viewer = lr()
                plt.plot(y, viewer.fit(y, y_hat).predict(y), color='k')
                plt.xlabel('Observed')
                plt.ylabel('Predicted')
                plt.show()

        else:
            y_pred = model.predict(X)
            sample_size = len(y_pred)
            print('\n')
            print(self.report_width * '*', '\n*')
            print('* MODEL PERFORMANCE \n*')
            print('* MODEL NAME: ', model_name)
            print('* TEST SAMPLE SIZE: ', sample_size)
            print('* ACCURACY: ', round(accuracy_score(y, y_pred)*100, 1), '%')
            print('* ')
            print(self.report_width * '*', '\n')
            report = classification_report(y, y_pred, output_dict=True)
            if graph:
                fig, ax = plt.subplots(figsize=(size, 0.3 * size))
                plt.title('Confusion Matrix')
                sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap='YlGn', fmt='d',)
                plt.xlabel('Predicted')
                plt.ylabel('True Class')
                plt.show()
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Classification Report')
                sns.heatmap(pd.DataFrame(report).iloc[0:3].T, annot=True, vmin=0, vmax=1, cmap='BrBG', fmt='.2g')
                plt.xlabel('Score')
                plt.show()
            else:
                display(pd.DataFrame(report).T)
        return None



# Rubia Models demo cases

def selectDemo(id):
    if id == 0:
        data, meta = scipy.io.arff.loadarff('dataset/scene_arff.arff')
        df = pd.DataFrame(data)
        y_cols = 'Beach'
        #y_cols = ['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']
        ignore_cols = ['Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']
    elif id == 1:
        df = pd.read_csv('Advertising.csv', index_col=0)
        y_cols = 'sales'
        ignore_cols = []
    elif id == 2:
        df = pd.read_csv('SAheart.csv')
        y_cols = 'chd'
        ignore_cols = []
    elif id == 3:
        df = pd.read_csv('pima-indians-diabetes.csv')
        y_cols = 'Class'
        ignore_cols = []
    else:
        df = pd.read_csv('iris.csv')
        y_cols = 'species'
        ignore_cols = []
    return df, y_cols, ig_cols


# load data as a pandas.dataframe object and pass it to the class
df, y_col, ignore_cols = selectDemo(0)

# load the class rubia_models and show important info about the dataset
# flag debug mode to True to show warning messages
rm = rubia_models(df, debug=False)
rm.describe(rm.data_raw)

# columns listed as ignored will be discarded while modeling
# flag graph to true to show some exploratory and correlation graphs on the dataset
rm.explore(rm.data_raw, y_col, ignore_cols, graph=False) #updates X, y, M

# encode every column of type object or string to categorical numbers
rm.encode()

# add higher level and interaction terms to the model
# be carefull when using higher level terms and graphs together, less powerfull hardware can bottleneck with higher complexity
rm.addTerms(rm.X, rm.y, levels=1, interaction=False, root=False)
rm.explore(rm.M, y_col, ignore_cols, graph=False) #updates X, y, M

# analyse if this is a regression or a classification problem and evaluate some models
# when y is float or has more then 10 different classes, the algorithm turns into a regression algorithm automatically
# else it will perform a classification modeling
rm.analyse(y_col)

# evaluate the performance of a mix of models
rm.evaluate(test_size=0.3, transformX='xstandard', transformY='ynone', folds=10, alphas=alphas, graph=False, metric='neg_mean_squared_error')

# apply tuning to the best models
alphas = 10 ** np.linspace(10, -2, 100) * 0.5
rm.test('Logistic One vs One', graph=True)
rm.test('Logistic classifier multinomial', graph=True)








# To get all coefficients for a given model:
#   lasso.coef_, lassocv.coef_, ridge.coef_, ridgecv.coef_
#   rfc.feature_importances_
#   logit.classes_, coef_, intercept_, n_iter_
#   nbc.class_count_, class_prior_, classes_, sigma_, theta_






# TO DO



# adaptar o rubia_models para receber y no formato multiclass
# acrescentar gridsearch para os modelos (pode ser na avaliação)
# acrescentar feature_selection



# mudar o alphas para um dict com params
# dentro de regression e classification, extrair os params do dict


# BALANCEAR CLASSES


# from sklearn.linear_model import ElasticNet
# rnc = RadiusNeighborsClassifier(radius=5, weights='distance')

# from sklearn.neighbors.nearest_centroid import NearestCentroid

# svc = SVC(kernel='poly')
# svc = SVC(kernel='linear')


# inverso do ravel
# label = label[:,np.newaxis]




# porque meu MNB nao funciona??? 
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import confusion_matrix


# train_x, test_x, train_y, test_y = train_test_split(data_1, data.chd)
# mnb = MultinomialNB(class_prior=[.25,.75])
# mnb.fit(train_x,train_y)
# y_pred = mnb.predict(test_x)
# print(confusion_matrix(y_true=test_y,y_pred=y_pred))
# print('Score MultinomialNB: ',mnb.score(test_x,test_y))




#
# from sklearn.feature_selection import SelectKBest, chi2
# # selecionar um modelo para otimizar
    
# pipetree = Pipeline([('scl', StandardScaler()), ('clf', DecisionTreeClassifier())])
# pipe = [pipetree]    

# param_range = [3, 5]
# grid_params = [{'clf__criterion': ['gini', 'entropy'],
#                 'clf__max_depth': param_range,
#                 'clf__min_samples_leaf': param_range,
#                 'clf__min_samples_split': param_range[1:]
#               }]


# scores = ['accuracy', 'recall_macro', 'precision_macro']
# for score in scores:
    
#     kfolds = StratifiedKFold(n_splits=2, shuffle=True)
#     cv = kfolds.split(X_train, y_train)

#     print("\n\n# Tuning hyper-parameters for %s" % score)
#     gs = GridSearchCV(estimator=pipetree, param_grid=grid_params, scoring=score, cv=cv)
#     gs.fit(X_train, y_train)
#     print('\nBest accuracy: %.3f' % gs.best_score_)
#     print('\nBest params:\n', gs.best_params_)

#     print("\nGrid scores on development set:")
#     means = gs.cv_results_['mean_test_score']
#     stds = gs.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, gs.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
    
#     print("\nClassification report:")
#     print()
#     y_true, y_pred = y_test, gs.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()   



# estimator.get_params().keys()



# quanto tem muitas preditoras, uma opcao é eliminar as menos relevantes
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# X, y = load_iris(return_X_y=True)
# print(X.shape)
# X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
# print(X_new.shape)




# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# #---------------------
# df = pd.DataFrame({
#     'age':[33,44,22,44,55,22],
#     'gender':['Male','Female','Male','Female','Male','Male']
# })
# #------------------------------
# le = LabelEncoder()
# df['gender_tf'] = le.fit_transform(df.gender)
# print(df)
# print(OneHotEncoder().fit_transform(df[['gender_tf']]).toarray())



# 
# from sklearn.preprocessing import RobustScaler


# 1.7 Caracteristicas polinomiais
# deriva caracteristica nao linear convertendo os dados para um grau maior
# Usado com Linear Regression para modelo de aprendizado de alto grau
# from sklearn.preprocessing import PolynomialFeatures
# #-------------------------
# df = pd.DataFrame({'A':[1,2,3,4,5], 'B':[2,3,4,5,6]})
# #----------------------------

# pol = PolynomialFeatures(degree=2)
# pol.fit_transform(df)



# X, y = make_blobs(n_features=2, n_samples=1000, cluster_std=2, centers = 2)
# #---------Plotting----------------------
# #f = plt.figure()
# #plt.scatter(X[:,0], X[:,1], c=y, s=10)
# #plt.show()
# #-------------------------------------
# h = .02
# x_min = X[:,0].min() - .5
# x_max = X[:,0].max() + .5
# y_min = X[:,1].min() - .5
# y_max = X[:,1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
# #---------------------------------------
# lr = LogisticRegression()
# lr.fit(X,y)
# Z = lr.predict(np.c_[xx.ravel(),yy.ravel()])
# #--------------------------------------
# Z = Z.reshape(xx.shape)
# plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired)
# plt.scatter(X[:,0],X[:,1], c=y, s=10)
# plt.show()