
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['seaborn', 'ggplot', 'seaborn-white'])

from scipy import stats

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
from sklearn.linear_model import SGDClassifier as sgdc, LogisticRegression as logitc
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.svm import SVC as svc
from sklearn.naive_bayes import GaussianNB as gnbc, BernoulliNB as bnbc
from sklearn.ensemble import RandomForestClassifier as rfc, GradientBoostingClassifier as gbc





class rubia_models:


    def __init__(self, df, width=100, debug=False):
        self.data_raw = df
        self.M = df
        self.report_width = width
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
        self.checkDtypes(df)
        print('* ')
        print(self.report_width * '*' + '\n')

        if graph:   
            size = 1.3 * self.report_width // 10
            # balance between every output class: pay special attention with unbalanced data
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
                row += 1 if col == (COLS - 1) else 0
                col = i % COLS    
                cax = ax[row, col] if ROWS > 1 else ax[col]
                if len(df[y_col].unique()) <= 10:
                    for cat in df[y_col].unique():
                        df[df[y_col]==cat][feature].hist(bins=30, alpha=0.5, edgecolor='white', ax=cax).set_title(feature)
                else:
                    df[feature].hist(bins=30, alpha=0.5, edgecolor='white', ax=cax).set_title(feature)
                plt.legend(df[y_col].unique())    
            plt.subplots_adjust(hspace=0.2, top = 0.92)
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
            sns.heatmap(self.M.corr(), ax=ax, mask=mask, annot = True, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.title('Correlation Matrix')
            plt.show()

        return None


    # encode all non numeric features
    def encode(self):
        le = LabelEncoder()    
        for col in self.M.columns:
            if str(self.M[col].dtype) == 'object' or str(self.M[col].dtype) == 'string':
                self.M[col] = le.fit_transform(self.M[col])


    # analyse if this is a regression or classification problem
    def analyse(self, y_col):
        if len(self.y[y_col].unique()) > 10 or str(self.y[y_col].dtype) == 'float64':
            self.strategy = 'regression'
        else:
            self.strategy = 'classification'


    # apply transformation to data
    def transform(self, who, transform, graph=False):
        size = 1.3 * self.report_width // 10
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

    
    # apply regression models
    def regression(self, metric="root_mean_squared_error", folds=10, alphas=[], graph=False):
        size = 1.3 * self.report_width // 10

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
        kf = KFold(n_splits=folds)
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


    # apply classification models
    def classification(self, metric, folds, alphas, graph):
        size = 1.3 * self.report_width // 10

        models = {}
        models["K nearest neighbors classifier K2"]  = knnc(n_neighbors=2)
        models["K nearest neighbors classifier K5"]  = knnc(n_neighbors=5)
        models["K nearest neighbors classifier K10"] = knnc(n_neighbors=10)        
        models["Decision tree classifier"]           = dtc()
        models["Logistic classifier"]                = logitc()
        models["SVM classifier with RBF kernel"]     = svc(gamma='scale')
        models["SVM classifier with linear kernel"]  = svc(kernel='linear')
        models["Gaussian naive bayes"]               = gnbc()
        models["Bernoulli naive bayes"]              = bnbc()
        models["SGD classifier"]                     = sgdc(max_iter=10000)
        models["Random forest classifier"]           = rfc(n_estimators=100)
        models["Gradient boosting classifier"]       = gbc()
        self.models = models

        print('\n')
        print(self.report_width * '*', '\n*')
        print('* CLASSIFICATION RESULTS - BEFORE PARAMETERS BOOSTING \n*')
        kf = StratifiedKFold(n_splits=folds, shuffle=True)
        results = []
        names = []
        for model_name in models:
            cv_scores = cross_val_score(models[model_name], self.Xt_train, self.yt_train.values.ravel(), cv=kf, error_score=np.nan)  
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

    
    # residual analysis for regression problems
    def calc_rss(self, residual):
        return float(((residual) ** 2).sum())         
    def calc_r2(self, y, y_hat):
        return r2_score(y_hat, y)
    def residual(self, y, y_hat, model_name, graph=False):
        size = 1.3 * self.report_width // 10
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


    # evaluate some models
    def evaluate(self, test_size=0.2, transformX='xnone', transformY='ynone', folds=10, alphas=[], graph=False, metric='neg_mean_squared_error'):
        if self.strategy == 'regression':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=True)
            # transform data
            rm.transform('X', transformX, graph) #model transf for X_train
            rm.transform('y', transformY, graph) #model transf for y_train
            self.regression(metric, folds, alphas, graph)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=True, stratify=self.y)
            # transform data
            rm.transform('X', transformX, graph) #model transf for X_train
            rm.transform('y', transformY, graph) #model transf for y_train
            self.classification(metric, folds, alphas, graph)


    # given a model name, evaluate y_hat/y_pred and the overall performance of such model
    def test(self, model_name, graph=False):
        size = 1.3 * self.report_width // 10
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
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Confusion Matrix')
                sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="YlGn", fmt='d',)
                plt.xlabel('Predicted')
                plt.ylabel('True Class')
                plt.show()
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Classification Report')
                sns.heatmap(pd.DataFrame(report).iloc[0:3].T, annot=True, vmin=0, vmax=1, cmap="YlGn", fmt='.2g')
                plt.xlabel('Score')
                plt.show()
            else:
                display(pd.DataFrame(report).T)





# load data as a pandas.dataframe object and pass it to the class
df = pd.read_csv('pima-indians-diabetes.csv')

# load the class rubia_models and show important info about the dataset
# flag debug mode to True to show warning messages
rm = rubia_models(df, debug=False)
rm.describe(rm.data_raw)

# please inform the output column here, a list of columns can also be ignored if necessary
# columns listed as ignored will be discarded while modeling
# flag graph to true to show some exploratory and correlation graphs on the dataset
y_col = 'Class'
ignore_cols = []
rm.explore(rm.data_raw, y_col, ignore_cols, graph=False) #updates X, y, M

# encode every column of type object or string to categorical numbers
rm.encode()
rm.explore(rm.M, y_col, ignore_cols, graph=False) #updates X, y, M

# analyse if this is a regression or a classification problem and evaluate some models
# when y is float or has more then 10 different classes, the algorithm turns into a regression algorithm automatically
# else it will perform a classification modeling
rm.analyse(y_col)

alphas = 10 ** np.linspace(10, -2, 100) * 0.5
rm.evaluate(test_size=0.3, transformX='xstandard', transformY='ynone', folds=10, alphas=alphas, graph=False, metric='neg_mean_squared_error')
rm.test('SGD classifier', graph=False)
rm.test('Logistic classifier', graph=True)




#df = pd.read_csv('Advertising.csv', index_col=0)
#rm = rubia_models(df, debug=False)
#rm.describe(rm.data_raw)
#y_col = 'sales'
#ignore_cols = []
#rm.explore(rm.data_raw, y_col, ignore_cols, graph=False) #updates X, y, M
#rm.encode()
#rm.explore(rm.M, y_col, ignore_cols, graph=False) #updates X, y, M
#rm.analyse(y_col)
#alphas = 10 ** np.linspace(10, -2, 100) * 0.5
#rm.evaluate(test_size=0.3, transformX='xstandard', transformY='ynone', folds=10, alphas=alphas, graph=False, metric='neg_mean_squared_error')
#rm.test('Linear regressor', graph=False)
#rm.test('Gradient boost regressor', graph=False)


#df = pd.read_csv('SAheart.csv')
#rm = rubia_models(df, debug=False)
#rm.describe(rm.data_raw)
#y_col = 'chd'
#ignore_cols = []
#rm.explore(rm.data_raw, y_col, ignore_cols, graph=False) #updates X, y, M
#rm.encode()
#rm.explore(rm.M, y_col, ignore_cols, graph=False) #updates X, y, M
#rm.analyse(y_col)
#alphas = 10 ** np.linspace(10, -2, 100) * 0.5
#rm.evaluate(test_size=0.3, transformX='xstandard', transformY='ynone', folds=10, alphas=alphas, graph=False, metric='neg_mean_squared_error')
#rm.test('SGD classifier', graph=True)
#rm.test('Logistic classifier', graph=True)




# To get all coefficients for a given model:
#   lasso.coef_, lassocv.coef_, ridge.coef_, ridgecv.coef_
#   rfc.feature_importances_

    

