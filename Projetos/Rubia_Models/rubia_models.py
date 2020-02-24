
import time

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['seaborn', 'ggplot', 'seaborn-white'])
from IPython.core.display import display

import scipy
from scipy import stats
from scipy.io import arff

from itertools import cycle, islice, combinations_with_replacement, product

import sklearn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# regressors
from sklearn.linear_model import LinearRegression as lr, SGDRegressor as sgdr, ElasticNet as enr
from sklearn.linear_model import Ridge as rr, RidgeCV as rcvr, Lasso as lassor, LassoCV as lassocvr
from sklearn.neighbors import KNeighborsRegressor as knnr
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.svm import SVR as svr
from sklearn.ensemble import RandomForestRegressor as rfr, AdaBoostRegressor as abr, GradientBoostingRegressor as gbr

# classifiers
from sklearn.linear_model import SGDClassifier as sgdc, LogisticRegression as logitc, RidgeClassifier as rc
from sklearn.neighbors import KNeighborsClassifier as knnc, NearestCentroid as ncc, RadiusNeighborsClassifier as rnc
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.svm import SVC as svc
from sklearn.naive_bayes import GaussianNB as gnbc, BernoulliNB as bnbc, MultinomialNB as mnbc
from sklearn.ensemble import RandomForestClassifier as rfc, GradientBoostingClassifier as gbc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as ldac

# clustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import KMeans as kmg, MeanShift as msg, MiniBatchKMeans as mbg
from sklearn.cluster import AgglomerativeClustering as acg, SpectralClustering as scg, AffinityPropagation as apg
from sklearn.cluster import DBSCAN as dbg, OPTICS as optg, Birch as big
from sklearn.mixture import GaussianMixture as gmg

# multiclass
from sklearn.multiclass import OneVsRestClassifier as ovrc, OneVsOneClassifier as ovoc

# redux
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

# pipeline and optimization
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# keras & tensorflow
from sklearn.metrics import f1_score
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 


class rubia_models:


    def __init__(self, df, width=100, debug=False):
        self.data_raw = df
        self.M = df
        self.report_width = width
        self.graph_width = 1.3 * width // 10
        self.graphs_expl = []
        self.graphs_model = []
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
        return cols_resume

    
    # show a few general info about the dataset
    def describe(self, df, printt=True):
        self.cols_dtypes = self.checkDtypes(df)
        if printt:
            print(self.report_width * '*', '\n*')
            print('* DATA OVERVIEW FOR THIS DATASET \n*')
            print('* DATA SHAPE: ', df.shape)
            print('* COLUMNS INFO: ', ', '.join(self.cols_dtypes))
            print('* ')
            print(self.report_width * '*')
            print('\nDATA SAMPLE: ')
            display(df.sample(5))
            print('\nSTATISTICS: ')
            display(df.describe(include='all').T)
            print('\n\n')
        return None
        

    # run a basic and repetitive EDA for a given pandas dataframe and remove constant columns
    def explore(self, df, y_cols, ig_cols, printt=True, graph=False):
        X_cols = [col for col in df.columns if col not in y_cols and col not in ig_cols]
        self.X = df.loc[:, X_cols]
        self.y = df.loc[:, y_cols]
        for col in self.X.columns:
            if len(self.X[col].unique()) == 1: 
                self.X.drop(col, axis=1, inplace=True)
                print('Column removed (constant value):', col)
        for col in self.y.columns:
            if len(self.y[col].unique()) == 1: 
                self.y.drop(col, axis=1, inplace=True)
                print('Column removed (constant value):', col)
        self.M = pd.concat([self.X, self.y], axis=1)

        if printt:
            print(self.report_width * '*', '\n*')
            print('* FEATURE EXTRACTION REPORT \n*')
            print('* X: ', ' | '.join(self.X.columns))
            print('* y: ', ' | '.join(self.y.columns))
            print('* M: ', self.X.shape, '|', self.y.shape)
            print('* ')
            print(self.report_width * '*' + '\n')

        if graph:  
            self.graphs_expl = [] 
            size = self.graph_width
            # balance between every output class: pay special attention with unbalanced data
            unique = []
            for y_col in y_cols:
                unique += list(df[y_col].unique())
                if len(df[y_col].unique()) <= 10:
                    fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                    df[y_col].value_counts().nlargest(10).plot(kind='bar')
                    plt.title('Classes Balance (%s)' % y_col)
                    plt.xticks(rotation=45)
                    self.graphs_expl.append(fig)
                    plt.show()
            # histogram for every feature: pay attention to outliers, data distribution and dimension
            if df.shape[1] <= 20: # for larger datasets, graphs are not recommended
                dfg = df.copy()
                if len(dfg) > 1000: dfg = dfg.sample(1000)
                COLS = 3
                ROWS = len(dfg.columns) // COLS + (1 if len(dfg.columns) % COLS != 0 else 0)
                fig, ax = plt.subplots(ROWS, COLS, figsize=(size, 4 * ROWS))
                row, col = 0, 0
                for i, feature in enumerate(dfg.columns):
                    if col == (COLS - 1):
                        row += 1
                        plt.subplots_adjust(hspace=0.2, top = 0.92)
                    else:
                        plt.subplots_adjust(hspace=0.2, top = 0.80)
                    col = i % COLS    
                    cax = ax[row, col] if ROWS > 1 else ax[col]
                    if len(unique) <= 10 and len(y_cols) == 1: # discriminate only one-level and few classes cases
                        for cat in dfg[y_cols[0]].unique():
                            dfg[dfg[y_cols[0]]==cat][feature].hist(bins=30, alpha=0.5, edgecolor='white', ax=cax).set_title(feature)
                    else:
                        dfg[feature].hist(bins=30, alpha=0.5, edgecolor='white', ax=cax).set_title(feature)
                fig.suptitle('Data Distribution', fontsize=14)
                self.graphs_expl.append(fig)
                plt.show()
                # pairplot and density plot for every column
                if len(unique) <= 10 and len(y_cols) == 1: # discriminate only one-level and few classes cases
                    g = sns.pairplot(dfg, hue=y_cols[0], plot_kws={'alpha':0.5, 's': 20})
                    handles = g._legend_data.values()
                    labels = g._legend_data.keys()
                    g._legend.remove()
                    g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3)
                else:
                    g = sns.pairplot(dfg, plot_kws={'alpha':0.5, 's': 20})
                g.fig.set_figwidth(0.75 * size)
                g.fig.set_figheight(0.75 * size)
                plt.subplots_adjust(top = 0.92, bottom=0.08)
                g.fig.suptitle('Pairplot and Density Matrix', fontsize=14)
                self.graphs_expl.append(g.fig)
                plt.show()
                # correlation heatmap matrix
                fig, ax = plt.subplots(figsize=(0.95 * size, 0.95 * size))
                corr = dfg.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                sns.heatmap(dfg.corr(), ax=ax, mask=mask, annot = True, vmin = -1, vmax = 1, center = 0, cmap = 'RdBu_r')
                plt.xticks(rotation=45)
                plt.yticks(rotation=45)
                plt.title('Correlation Matrix')
                self.graphs_expl.append(fig)
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
        cols = []
        for col in X.columns: #lets add polynomial terms only for valid data types
            if not str(self.X[col].dtype) == 'object' and not str(self.X[col].dtype) == 'string':
                cols.append(col)
        if levels > 1: #higher level terms makes sense only for k > 1
            #calculating a combination of n elements in groups of k with replacement
            n, k = cols, levels
            if interaction:
                for comb in combinations_with_replacement(n,k):
                    comb = list(comb)
                    self.X['_'.join(comb)] = self.calcTerms(X, comb)
            # or just the polynomial terms if interaction = False
            else:
                for col in cols:
                    for order in range(2, k+1):
                        comb = [col for elem in range(1, order+1)]
                        self.X['_'.join(comb)] = self.calcTerms(X, comb)
        if root:
            for col in cols:
                comb = [col]
                try: # cannot apply root transform to some data types or data values
                    self.X['sqroot_'+col] = self.calcTerms(X, comb)
                except:
                    pass
        self.M = pd.concat([self.X, self.y], axis=1)
        return None

    
    # encode all non numeric features
    def encode(self, encoder='LabelEncoder', who='both'):
        if encoder == 'LabelEncoder':
            le = LabelEncoder()   
            if who == 'both' or who == 'X':
                for col in self.X.columns:
                    if str(self.X[col].dtype) == 'object' or str(self.X[col].dtype) == 'string':
                        self.X[col] = le.fit_transform(self.X[col])
            if who == 'both' or who == 'Y':
                for col in self.y.columns:
                    if str(self.y[col].dtype) == 'object' or str(self.y[col].dtype) == 'string':
                        self.y[col] = le.fit_transform(self.y[col])
        else:
            #le = OneHotEncoder() 
            if who == 'both' or who == 'X':
                for col in self.X.columns:
                    if str(self.X[col].dtype) == 'object' or str(self.X[col].dtype) == 'string':
                        self.X = pd.concat([self.X, pd.get_dummies(self.X[col], prefix=col, dummy_na=True)], axis=1).drop([col], axis=1)
            if who == 'both' or who == 'Y':
                for col in self.y.columns:
                    if str(self.y[col].dtype) == 'object' or str(self.y[col].dtype) == 'string':
                        self.y = pd.concat([self.y, pd.get_dummies(self.y[col], prefix=col, dummy_na=True)], axis=1).drop([col], axis=1)
        self.M = pd.concat([self.X, self.y], axis=1)
        return None


    # auto balance the dataset M 
    # only when classes equal or less than 10 classes
    def balance(self, tol, df, y_cols, ig_cols):
        if len(y_cols) == 1:
            y_col = y_cols[0]
            if len(df[y_col].unique()) <= 10:
                size_before = len(df)
                size_smallest = df[y_col].value_counts().min()
                newdf = pd.DataFrame()
                for yclass in df[y_col].unique():
                    sample = df.loc[df[y_col]==yclass].sample(size_smallest)
                    newdf = pd.concat([newdf, sample], axis=0)
                size_after = len(newdf)
                if (abs(size_after-size_before)/size_before) < tol:
                    X_cols = [col for col in newdf.columns if col not in y_cols and col not in ig_cols]
                    newdf.reset_index(inplace=True, drop=True)
                    self.X = newdf.loc[:, X_cols]
                    self.y = newdf.loc[:, y_cols]
                    self.M = pd.concat([self.X, self.y], axis=1)
        return None


    # analyse if this is a regression or classification problem
    # more than 10 classes transforms the process automatically to regression
    def analyse(self, y_cols, priority='Classical'):
        if priority != 'Classical' or self.y.shape[1] > 1:
            strategy = []
            for y_col in y_cols:
                if len(self.y[y_col].unique()) > 10 or str(self.y[y_col].dtype) == 'float64':
                    strategy.append('regression_nn')
                else:
                    strategy.append('classification_nn')
            self.strategy = 'regression_nn' if 'regression_nn' in strategy else 'classification_nn'
        elif len(y_cols) < 1:
            self.strategy = 'clustering'
        else:
            strategy = []
            for y_col in y_cols:
                if len(self.y[y_col].unique()) > 10 or str(self.y[y_col].dtype) == 'float64':
                    strategy.append('regression')
                else:
                    strategy.append('classification')
            self.strategy = 'regression' if 'regression' in strategy else 'classification'
        print('Problem identified as', self.strategy)
        return None


    # dimensionality reduction
    def redux(self, k=10, mode='chi-square', transform='None'):
        if k == 'auto':
            k = 10 # require deeper implementation
        if mode == 'chi-square' and self.X.shape[1] >= k and self.y.shape[1] > 0:
            selector = SelectKBest(chi2, k=k)
            best_features = selector.fit_transform(self.X, self.y)
            mask = selector.get_support(indices=True)
            self.X = self.X.iloc[:,mask]
        elif mode == 'pca' and self.X.shape[1] >= k:
            if transform != 'None':
                scaler = MinMaxScaler() # only minmax supported right now
                self.scalerX_prepca = scaler.fit(self.X)
                scaledX = self.scalerX_prepca.transform(self.X)
            else:
                scaledX = self.X
            self.scalerX_pca = PCA(n_components=k).fit(scaledX)
            X_pca = pd.DataFrame(self.scalerX_pca.transform(scaledX))
            X_pca.columns = ['PC_%d'%(i+1) for i in range(k)]
            self.X = X_pca
        self.M = pd.concat([self.X, self.y], axis=1)
        return None


    # apply transformation to data
    def transform(self, who, transform, graph=False):
        size = self.graph_width
        if who == 'X':
            if transform == 'None':
                self.scalerX = None 
                self.Xt_train = self.X_train
                if len(self.X_test) > 0: self.Xt_test = self.X_test
            if transform == 'Standard' or transform == 'MinMax' or transform == 'Robust':
                if transform == 'Standard':
                    self.scalerX = StandardScaler()
                if transform == 'MinMax':
                    self.scalerX = MinMaxScaler()
                if transform == 'Robust':
                    self.scalerX = RobustScaler()
                self.scalerX.fit(self.X_train)
                self.Xt_train = self.scalerX.transform(self.X_train)
                if len(self.X_test) > 0: self.Xt_test = self.scalerX.transform(self.X_test)
                if graph:
                    fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                    if self.Xt_train.shape[1] == 1:
                        sns.kdeplot(self.Xt_train[:,0])
                    else:
                        for i in range(self.Xt_train.shape[1]):
                            sns.kdeplot(self.Xt_train[:,i])
                    plt.title('X-Features after Transformation (training set)')
                    self.graphs_model.append(fig)
                    plt.show()   
        if who == 'y':
            if transform == 'None':
                self.scalery = None
                self.yt_train = self.y_train 
                self.yt_test = self.y_test   
            if transform == 'BoxCox':
                # check if negative/null numbers exists
                miny = self.y.min().item()
                # if so, add the average of y_train to all y to avoid invalid data during boxcox transf
                # using the mean makes the model to robust to new and unknown y data
                deslocy = 0 if miny > 0 else (self.y.mean().item() + 1)                 
                ytransf = self.y_train + deslocy # add offset if necessary
                self.yt_train, lambday = stats.boxcox(ytransf)
                self.scalery = (deslocy, lambday)
                print(self.y_test + deslocy)
                self.yt_test = stats.boxcox(self.y_test + deslocy, lmbda=lambday) 
                if graph:
                    fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                    if self.yt_train.shape[1] == 1:
                        sns.kdeplot(self.yt_train[:,0])
                    else:
                        for i in range(self.yt_train.shape[1]):
                            sns.kdeplot(self.yt_train[:,i])
                    plt.title('y-Features after Transformation (training set)')
                    self.graphs_model.append(fig)
                    plt.show()  
        return None
    

    # apply clustering models
    def clustering(self, metric, xy=(0,0), printt=True, graph=False):
        size = self.graph_width
        X = np.array(self.Xt_train)

        # significant model setup differences should be list as different models
        bandwidth = estimate_bandwidth(X, quantile=0.3)
        connectivity = kneighbors_graph(X, n_neighbors=5, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        models = {}
        models["KMeans K2"]             = kmg(n_clusters=2)
        models["KMeans K5"]             = kmg(n_clusters=5)
        models["KMeans K10"]            = kmg(n_clusters=10)
        models["Mean Shift"]            = msg(bandwidth=bandwidth, bin_seeding=True)
        models["Mini Batch K5"]         = mbg(n_clusters=5)
        models["Agglomerative Ward K5"] = acg(n_clusters=5, linkage='ward', connectivity=connectivity)
        models["Agglomerative Avg K5"]  = acg(linkage="average", affinity="cityblock", n_clusters=5, connectivity=connectivity)
        models["Spectral K5"]           = scg(n_clusters=5, eigen_solver='arpack', affinity="nearest_neighbors")
        models["DBScan Euclidean"]      = dbg(eps=0.5, min_samples=10, metric='euclidean')
        models["DBScan Manhattan"]      = dbg(eps=0.5, min_samples=10, metric='manhattan')
        models["DBScan Cityblock"]      = dbg(eps=0.5, min_samples=10, metric='cityblock')
        models["Optics"]                = optg(min_samples=10, xi=0.05, min_cluster_size=0.1)
        models["Affinity Propagation"]  = apg(damping=0.9, preference=-200)
        models["Birch K5"]              = big(n_clusters=5)
        models["Gaussian Mixture K5"]   = gmg(n_components=5, covariance_type='full')
        
        self.models = models

        # for clustering methods, evaluation of best model will be delegated visually to the user
        names = []
        et = []
        results = []
        ROWS, COLS = len(models) // 3 + (1 if len(models) % 3 != 0 else 0), 3
        fig, ax = plt.subplots(figsize=(0.85 * size, ROWS * 0.2 * size))
        fig.suptitle('Clustering Analysis', fontsize=18)
        #plt.figure(figsize=(size, ROWS * 0.2 * size))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.93, wspace=.03, hspace=.20)
        plot_num = 1
        for model_name in models:
            start = time.time()
            models[model_name].fit(X)
            if hasattr(models[model_name], 'labels_'):
                y_pred = models[model_name].labels_.astype(np.int)
            else:
                y_pred = models[model_name].predict(X)
            if len(np.unique(y_pred)) > 1:
                results.append(silhouette_score(X, y_pred, metric=metric))
            else:
                results.append(0)
            elapsed = (time.time() - start)
            names.append(model_name)
            et.append(elapsed)
            plt.subplot(ROWS, COLS, plot_num)
            plt.title(model_name, size=14)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                        int(max(y_pred) + 1))))
            # add gray color for outliers (if any)
            colors = np.append(colors, ["#bbbbbb"])
            plt.scatter(X[:, xy[0]], X[:, xy[1]], s=10, color=colors[y_pred])
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % elapsed).lstrip('0'), transform=plt.gca().transAxes, size=14, horizontalalignment='right')
            plot_num += 1
        self.graphs_model.append(fig)
        plt.show()  

        report = pd.DataFrame({'Model': names, 'Elapsed Time': et, 'Score (silhouette)': results})
        report.sort_values(by='Score (silhouette)', ascending=False, inplace=True)
        report.reset_index(inplace=True, drop=True)
        self.report_performance = report
        
        if printt:
            print('\n')
            print(self.report_width * '*', '\n*')
            print('* CLUSTERING RESULTS - BEFORE PARAMETERS BOOSTING \n*')
            print(self.report_width * '*', '')
            display(report)
            print('\n')

        if graph:
            fig, ax = plt.subplots(figsize=(size, 0.5 * size))
            ax.set_xticks(np.arange(len(report)))
            plt.title('Silhouette Comparison')
            plt.plot(report['Score (silhouette)'])
            ax.set_xticklabels(report.Model)
            plt.xticks(rotation=45)
            plt.subplots_adjust(hspace=0.0, bottom=0.25)
            self.graphs_model.append(fig)
            plt.show()             
        return None


    # apply regression models
    def regression(self, metric, folds=10, alphas=[], printt=True, graph=False):
        size = self.graph_width

        # significant model setup differences should be list as different models
        models = {}
        models["Linear regressor"]                  = lr()
        models["Lasso regressor"]                   = lassor()
        models["Lasso CV regressor"]                = lassocvr()
        models["Ridge regressor"]                   = rr(alpha=0, normalize=True)
        models["Ridge CV regressor"]                = rcvr(alphas = alphas)
        models["Elastic net regressor"]             = enr()
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
        models["Support vector regressor RBF"]      = svr()
        models["Support vector regressor Linear"]   = svr('linear')
        models["Support vector regressor Poly"]     = svr(kernel='poly')
        self.models = models

        kf = KFold(n_splits=folds, shuffle=True)
        results = []
        names = []
        et = []
        for model_name in models:
            start = time.time()
            cv_scores = -1 * cross_val_score(models[model_name], self.Xt_train, self.yt_train, cv=kf, scoring=metric)  
            results.append(cv_scores)
            names.append(model_name)
            et.append((time.time() - start))
        report = pd.DataFrame({'Model': names, 'Score': results, 'Elapsed Time': et})
        report['Score (avg)'] = report.Score.apply(lambda x: np.sqrt(x).mean())
        report['Score (std)'] = report.Score.apply(lambda x: np.sqrt(x).std())
        report['Score (VC)'] = 100 * report['Score (std)'] / report['Score (avg)']
        report.sort_values(by='Score (avg)', inplace=True)
        report.drop('Score', axis=1, inplace=True)
        report.reset_index(inplace=True, drop=True)
        self.report_performance = report
        
        if printt:
            print('\n')
            print(self.report_width * '*', '\n*')
            print('* REGRESSION RESULTS - BEFORE PARAMETERS BOOSTING \n*')
            print(self.report_width * '*', '')
            display(report)
            print('\n')

        if graph:
            fig, ax = plt.subplots(figsize=(size, 0.5 * size))
            plt.title('Regressor Comparison')
            #ax = fig.add_subplot(111)
            plt.boxplot(results)
            ax.set_xticklabels(names)
            plt.xticks(rotation=45)
            plt.subplots_adjust(hspace=0.0, bottom=0.25)
            self.graphs_model.append(fig)
            plt.show()             
        return None


    # apply classification models
    def classification(self, metric, folds, printt=True, graph=False):
        size = self.graph_width

        if len(self.y.iloc[:,0].unique()) > 2:
            struct = 'multiclass'
        else:
            struct = 'binary'

        # significant model setup differences should be list as different models
        models = {}
        models["Linear discriminant analysis"]          = ldac()
        models["Nearest centroid classifier euclidian"] = ncc(metric='euclidean')
        models["Nearest centroid classifier manhattan"] = ncc(metric='manhattan')
        models["K nearest neighbors classifier K2"]     = knnc(n_neighbors=2)
        models["K nearest neighbors classifier K5"]     = knnc(n_neighbors=5)
        models["K nearest neighbors classifier K10"]    = knnc(n_neighbors=10)        
        models["Decision tree classifier"]              = dtc()
        models["Gaussian naive bayes"]                  = gnbc()
        models["Bernoulli naive bayes"]                 = bnbc(binarize=0.5)
        models["Multinomial naive bayes"]               = mnbc()
        models["SGD classifier"]                        = sgdc(max_iter=10000)
        models["Ridge classifier"]                      = rc()

        if len(self.Xt_train) < 10000:
            models["SVM classifier RBF"]                = svc(gamma='scale')
            models["SVM classifier Linear"]             = svc(kernel='linear')
            models["SVM classifier Poly"]               = svc(kernel='poly')

        if self.Xt_train.shape[0] < 10000 or self.Xt_train.shape[1] < 5:
            models["Gradient boosting classifier"]      = gbc()
            models["Random forest classifier"]          = rfc(n_estimators=100)

        if struct == 'multiclass':
            models["Logistic classifier multinomial"]   = logitc(multi_class='multinomial', solver='lbfgs')
            models["Logistic classifier auto"]          = logitc(multi_class='auto')
            models["Logistic One vs Rest"]              = ovrc(logitc())
            models["Logistic One vs One"]               = ovoc(logitc())

        if struct == 'binary':
            models["Logistic classifier"]               = logitc(max_iter=2000)

        self.models = models

        kf = StratifiedKFold(n_splits=folds, shuffle=True)
        results = []
        names = []
        et = []
        for model_name in models:
            start = time.time()
            cv_scores = cross_val_score(models[model_name], self.Xt_train, self.yt_train, cv=kf, scoring=metric, error_score=np.nan)  
            results.append(cv_scores)
            names.append(model_name)
            et.append((time.time() - start))
            #print(model_name, time.time() - start)
        report = pd.DataFrame({'Model': names, 'Score': results, 'Elapsed Time': et})
        report['Score (avg)'] = report.Score.apply(lambda x: x.mean())
        report['Score (std)'] = report.Score.apply(lambda x: x.std())
        report['Score (VC)'] = 100 * report['Score (std)'] / report['Score (avg)']
        report.sort_values(by='Score (avg)', inplace=True, ascending=False)
        report.drop('Score', axis=1, inplace=True)
        report.reset_index(inplace=True, drop=True)
        self.report_performance = report

        if printt:
            print('\n')
            print(self.report_width * '*', '\n*')
            print('* CLASSIFICATION RESULTS - BEFORE PARAMETERS BOOSTING \n*')
            print(self.report_width * '*', '')
            display(report)
            print('\n')

        if graph:
            fig, ax = plt.subplots(figsize=(size, 0.5 * size))
            plt.title('Classifier Comparison')
            #ax = fig.add_subplot(111)
            plt.boxplot(results)
            ax.set_xticklabels(names)
            plt.xticks(rotation=45)
            plt.subplots_adjust(hspace=0.0, bottom=0.25)
            self.graphs_model.append(fig)
            plt.show()             
        return None

 
    # residual analysis for regression problems
    def calc_rss(self, residual):
        return float(((residual) ** 2).sum())         
    def calc_rmse(self, y, y_hat):
        return np.sqrt(mean_squared_error(y_hat, y))
    def calc_r2(self, y, y_hat):
        return r2_score(y_hat, y)
    def residual(self, y, y_hat, model_name, printt=True, graph=False):
        size = self.graph_width
        # do some data conversion because of different methods data types
        sample_size = len(y_hat)
        if isinstance(y, pd.DataFrame): 
            y = np.array(y.values.ravel())
        y = y.reshape(-1, 1)
        y_hat = np.array(y_hat).reshape(-1, 1)
        res = y - y_hat
        obs = np.arange(1, sample_size+1).reshape(-1, 1)

        if printt:
            print('\n')
            print(self.report_width * '*', '\n*')
            print('* MODEL PERFORMANCE \n*')
            print('* MODEL NAME: ', model_name)
            print('* TEST SAMPLE SIZE: ', sample_size)
            print('* RMSE: %.2f'%self.calc_rmse(y, y_hat))
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
            self.graphs_model.append(fig)
            plt.show()
        return 'RMSE: %.2f | R2: %.2f' % (self.calc_rmse(y, y_hat), self.calc_r2(y, y_hat))


    # add a layer of type Dense to an existing NN model
    def addDense(self, neurons, activation='default', input_dim=0):
        if activation == 'default':
            if input_dim > 0:
                self.model.add(Dense(units=neurons, input_dim=input_dim))
            else:
                self.model.add(Dense(units=neurons))
        else:
            if input_dim > 0:
                self.model.add(Dense(units=neurons, activation=activation, input_dim=input_dim))
            else:
                self.model.add(Dense(units=neurons, activation=activation))
        return None
    
    
    # add a dropout layer to an existing NN model
    def addDropout(self, perc=0.2):
        self.model.add(Dropout(perc))
        return None


    # create a Sequential topology using the pre defined scheme 
    def topology(self, metric, scheme, loss, optimizer):
        self.model = Sequential() 
        n_output = self.yt_train.shape[1]
        # multioutput regression not available so far
        if n_output == 1 and self.strategy == 'classification_nn': 
            if len(self.y.ix[:,0].unique()) > 2: 
                n_output = len(self.y.ix[:,0].unique())   
            else:
                n_output = 1 # only one neuron for binary single classification   
        
        # add each layer to the model
        for i, (name, layer) in enumerate(scheme.items()):
            input_dim = 0
            if len(self.model.layers) == 0:
                input_dim = self.Xt_train.shape[1] # set input dim for the first layer
            if i + 1 == len(scheme): # set the number of neurons in the last layer equal len(target)
                self.addDense(n_output, layer['activation'], input_dim) 
            else:
                if layer['type'] == 'Dense':
                    self.addDense(layer['neurons'], layer['activation'], input_dim)  
                if len(self.model.layers) != 0 and layer['type'] == 'Dropout':
                    self.addDropout(layer['perc'])  
                    
        # updates the last layer shape to the output shape
        scheme[name].update({'type': layer['type'], 
                                      'neurons': n_output,
                                      'activation': layer['activation']})
        # prepare the computational graph for this topology
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

        #self.model.summary()
        self.current_scheme = scheme
        return None

    
    # fit and test the performance of the ANN
    def trainANN(self, epochs, verbose):
        x_train = np.asarray(self.Xt_train)
        x_test = np.asarray(self.Xt_test)
        y_train = np.asarray(self.yt_train)
        y_test = np.asarray(self.yt_test)
        #try: # some combinations are not allowed, so, return zero if necessary
        hist = self.model.fit(x=x_train, y=y_train, epochs=epochs, verbose=verbose, validation_split=0.2, shuffle=True)

        # evaluate the performance of the classifier
        if self.strategy == 'classification_nn' and self.yt_train.shape[1] == 1:
            y_pred = self.model.predict_classes(x_test)
        elif self.strategy == 'classification_nn' and self.yt_train.shape[1] > 1:
            y_pred = self.model.predict(x_test)
            y_test = np.argmax(y_test, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        elif self.strategy == 'regression_nn' and self.yt_train.shape[1] == 1:
            y_pred = self.model.predict(x_test)
        #display(pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_pred)], axis=1).sample(5))
        if self.strategy == 'classification_nn':
            score = 100 * f1_score(y_test, y_pred, average='macro')
            accuracy = accuracy_score(y_test, y_pred, normalize=True)
        else:
            score = self.calc_rmse(y_test, y_pred)
            accuracy = self.calc_r2(y_test, y_pred)
        
        if (score > self.best_score and self.strategy == 'classification_nn') or \
            (score < self.best_score and self.strategy == 'regression_nn'):
            self.best_model = self.model
            self.best_scheme = self.current_scheme
            self.best_score = score
            self.history = hist.history
            self.y_pred = y_pred
            self.y_true = y_test
            #print('score', score)
            #print('accuracy', accuracy)
            #print('training evolution')
            #print('elapsed time')
        #except: 
        #    score = 0
        return score
    
    
    # random search for better model performance
    def searchANN(self, runs=1, max_depth=1, metric='mse', fixed={}, printt=True, graph=False):
        size = self.graph_width
        
        # list of optimization options, if not in the fixed dict
        if 'neurons' not in fixed.keys(): 
            neurons_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            neurons_list = fixed['neurons']
            
        if 'loss' not in fixed.keys(): 
            if self.strategy == 'regression_nn':
                loss_list = ['mean_squared_error', 'mean_absolute_error',
                            'hinge', 'logcosh']
            else:
                if self.y.shape[1] == 1 and len(self.y.ix[:,0].unique()) == 2:
                    loss_list = ['mean_squared_error', 'mean_absolute_error',
                                'binary_crossentropy', 'categorical_hinge', 'hinge',
                                'logcosh']
                elif self.y.shape[1] == 1 and len(self.y.ix[:,0].unique()) > 2:
                    loss_list = ['sparse_categorical_crossentropy', 'categorical_hinge', 'hinge',
                                'logcosh']
                elif self.y.shape[1] > 1:
                    loss_list = ['mean_squared_error', 'mean_absolute_error', 
                                'binary_crossentropy', 'categorical_crossentropy',
                                'categorical_hinge', 'hinge', 'logcosh']
        else:
            loss_list = fixed['loss']
            
        if 'optimizer' not in fixed.keys(): 
            optimizer_list = ['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta']
        else:
            optimizer_list = fixed['optimizer']
            
        if 'epochs' not in fixed.keys(): 
            epoch_list = [5, 100, 500, 1000]
        else:
            epoch_list = fixed['epochs']
            
        if 'type' not in fixed.keys(): 
            type_list = ['Dense', 'Dropout']
        else:
            type_list = fixed['type']
            
        if 'activation' not in fixed.keys(): 
            if self.strategy == 'regression_nn':
                activation_list = ['elu', 'relu', 'linear']
            else:
                if self.y.shape[1] == 1 and len(self.y.ix[:,0].unique()) == 2:
                    activation_list = ['sigmoid', 'softmax', 'elu', 'relu', 'linear', 'tanh']
                elif self.y.shape[1] == 1 and len(self.y.ix[:,0].unique()) > 2:
                    activation_list = ['sigmoid', 'softmax', 'elu', 'relu', 'linear', 'tanh']
                elif self.y.shape[1] > 1:
                    activation_list = ['sigmoid', 'softmax', 'elu', 'relu', 'linear', 'tanh']
        else:
            activation_list = fixed['activation']
            
        self.best_model = None
        self.best_score = 0.0 if self.strategy == 'classification_nn' else +np.inf
        
        # generate the scheme, evaluate and store all relevant results
        names = []
        et = []
        results = []
        model_scheme = []
        losses = []
        optimizers = []
        eps = []
        for run in range(runs):
            start = time.time()
            n_layers = np.random.randint(1, max_depth+1) # random number of layers between 1 and max_depth
            n_in = len(self.y.columns)
            # add a very simple topology to the models initial scheme
            scheme = {}
            for i in range(1, n_layers+1):
                # remove invalid layer types in the first layer
                allowed = [typeof for typeof in type_list if (i != 1 or (typeof != 'Dropout' and i == 1))]
                # remove invalid layer types in the last layer
                allowed = [typeof for typeof in allowed if (i != n_layers or (typeof != 'Dropout' and i == n_layers))]
                layer_type = np.random.choice(allowed)
                if layer_type == 'Dense':
                    scheme.update({'Layer %d'%i: {'type': layer_type, 
                                                  'neurons': np.random.choice(neurons_list),
                                                  'activation': np.random.choice(activation_list)}})
                elif layer_type == 'Dropout':
                    scheme.update({'Layer %d'%i: {'type': layer_type, 
                                                  'perc': np.random.uniform(0.05, 0.5)}})
            
            # prepare the topology for this network
            loss = np.random.choice(loss_list)
            losses.append(loss)
            optimizer = np.random.choice(optimizer_list)
            optimizers.append(optimizer)
            self.topology(metric, scheme, loss, optimizer)
            
            # train and evaluate this topology, compare with previous best and update models if necessary
            epochs = np.random.choice(epoch_list)
            eps.append(epochs)
            score = self.trainANN(epochs, 0)
            results.append(score)
            model_scheme.append(scheme)
            names.append('Experiment #%d'%(run+1))
            elapsed = (time.time() - start)
            et.append(elapsed)
        report = pd.DataFrame({'Experiment': names, 'Elapsed Time': et, 'Score': results, 'Model': model_scheme,
                               'Loss': losses, 'Optimizer': optimizers, 'Epochs': eps})
        if self.strategy == 'classification_nn':
            report.sort_values(by='Score', ascending=False, inplace=True)
            score_type = 'f1-score'
        else:
            report.sort_values(by='Score', inplace=True)
            score_type = 'RMSE'
        report.reset_index(inplace=True, drop=True)
        self.report_performance = report
            
        if printt:
            print('\n')
            print(self.report_width * '*', '\n*')
            print('* NEURAL NETWORK RESULTS - BEFORE PARAMETERS BOOSTING \n*')
            print(self.report_width * '*', '')
            display(report)
            print('\n')

        if graph:
            if 'mse' in self.history.keys():
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Training Evolution')
                plt.plot(self.history['mse'], label='training')
                if 'val_mse' in self.history.keys(): plt.plot(self.history['val_mse'], label='test')
                plt.xlabel('Epochs')
                plt.ylabel('MSE')
                plt.legend()
                plt.subplots_adjust(hspace=0.0, bottom=0.25)
                self.graphs_model.append(fig)
                plt.show()             
            if 'accuracy' in self.history.keys():
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Training Evolution')
                plt.plot(self.history['accuracy'], label='training')
                if 'val_accuracy' in self.history.keys(): plt.plot(self.history['val_accuracy'], label='test')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.subplots_adjust(hspace=0.0, bottom=0.25)
                self.graphs_model.append(fig)
                plt.show()             
            if 'loss' in self.history.keys():
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Training Evolution')
                plt.plot(self.history['loss'], label='training')
                if 'val_loss' in self.history.keys(): plt.plot(self.history['val_loss'], label='test')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.subplots_adjust(hspace=0.0, bottom=0.25)
                self.graphs_model.append(fig)
                plt.show()  
                
        y_true = self.y_true
        y_pred = self.y_pred
        if self.strategy == 'classification_nn':
            report = classification_report(y_true, y_pred, output_dict=True)
            score = f1_score(y_true, y_pred, average='macro')
            if printt:
                print(self.report_width * '*', '\n*')
                print('* HYPER-PARAMETER TUNING REPORT\n*')
                print("* SCORING METHOD: %s" % score_type)
                print('* BEST SCORE: %.1f %%' % (100 * score))
                print('* TEST SIZE: %d' % len(y_true))
                print('* LOSS: %s' % self.report_performance['Loss'].iloc[0])
                print('* OPTIMIZER: %s' % self.report_performance['Optimizer'].iloc[0])
                print('* EPOCHS: %d' % self.report_performance['Epochs'].iloc[0])
                print('* BEST MODEL:', self.best_scheme)
                print('*\n', self.report_width * '*')
                if not graph:
                    display(pd.DataFrame(report).T)
                #self.best_model.summary()

            if graph:
                fig, ax = plt.subplots(figsize=(size, 0.3 * size))
                plt.title('Confusion Matrix')
                sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='YlGn', fmt='d',)
                plt.xlabel('Predicted')
                plt.ylabel('True Class')
                self.graphs_model.append(fig)
                plt.show()
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Classification Report')
                sns.heatmap(pd.DataFrame(report).iloc[0:3].T, annot=True, vmin=0, vmax=1, cmap='BrBG', fmt='.2g')
                plt.xlabel('Score')
                self.graphs_model.append(fig)
                plt.show()

        else:
            if printt:
                print(self.report_width * '*', '\n*')
                print('* HYPER-PARAMETER TUNING REPORT\n*')
                print("* SCORING METHOD: %s" % score_type)
                print('* TEST SIZE: %d' % len(y_true))
                print('* LOSS: %s' % self.report_performance['Loss'].iloc[0])
                print('* OPTIMIZER: %s' % self.report_performance['Optimizer'].iloc[0])
                print('* EPOCHS: %d' % self.report_performance['Epochs'].iloc[0])
                print('* BEST MODEL:', self.best_scheme)
                print('*\n', self.report_width * '*')
            if graph:
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Model Overall Performance')
                plt.scatter(y_true, y_pred, color='g')
                viewer = lr()
                plt.plot(y_true, viewer.fit(y_true, y_pred).predict(y_true), color='k')
                plt.xlabel('Observed')
                plt.ylabel('Predicted')
                self.graphs_model.append(fig)
                plt.show()
            self.residual(y_true, y_pred, self.report_performance['Experiment'].iloc[0], printt, graph)
            
        return None


    # evaluate some models
    # metric can be any from one of those:
    # - grid search score metrics for sklearn classifiers
    # - grid search score metrics for sklearn regressors
    # - silhouette score metrics for sklearn clustering
    def evaluate(self, test_size=0.2, transformX='None', transformY='None', folds=10, alphas=[], xy=(0,0), printt=True, graph=False, metric=''):
        self.graphs_model = []
        if self.strategy == 'regression':
            if metric == '': metric = 'neg_mean_squared_error'
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=True)
            # transform data
            self.transform('X', transformX, graph) #model transf for X_train
            self.transform('y', transformY, graph) #model transf for y_train
            self.regression(metric, folds, alphas, printt, graph)
        if self.strategy == 'regression_nn':
            if metric == '': metric = 'mse'
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=True)
            # transform data
            self.transform('X', transformX, graph) #model transf for X_train
            self.transform('y', transformY, graph) #model transf for y_train
            self.searchANN(runs=10, max_depth=3, metric=metric, fixed={}, printt=printt, graph=graph)
        elif self.strategy == 'classification':
            if metric == '': metric = 'accuracy'
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=True, stratify=self.y)
            # transform data
            self.transform('X', transformX, graph) #model transf for X_train
            self.transform('y', transformY, graph) #model transf for y_train
            self.classification(metric, folds, printt, graph)
        elif self.strategy == 'classification_nn':
            if metric == '': metric = 'accuracy'
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=True, stratify=self.y)
            # transform data
            self.transform('X', transformX, graph) #model transf for X_train
            self.transform('y', transformY, graph) #model transf for y_train
            self.searchANN(runs=10, max_depth=3, metric=metric, fixed={}, printt=printt, graph=graph)
        elif self.strategy == 'clustering':
            if metric == '': metric = 'euclidean'
            self.X_train = self.X
            self.X_test = pd.DataFrame()
            # transform data
            self.transform('X', transformX, graph) #model transf for X_train
            self.clustering(metric, xy, printt, graph)
        return None


    # given a model, do a grid search for parameters optimization
    def boost(self, model, printt=True, fixed={}): 

        # n_tests is used to create linspaced values for some grid parameters
        n_tests = 10
        alphas = 10 ** np.linspace(10, -2, n_tests) * 0.5 
        percs = np.linspace(0, 1, n_tests) 
        groups = [2, 3, 5, 10]
        pot10 = [1, 10, 100, 1000]
        samples = [5, 10, 30]
        quarts = [0.25, 0.5, 0.75]

        params = list(model.get_params().keys())
        grid_params = {}

        # use all processors n_jobs=-1     
        # list parameters range for the more common hyper parameters    
        if 'n_jobs' in params: grid_params.update({'clf__n_jobs':[-1]})
        if 'shrinkage' in params: grid_params.update({'clf__shrinkage':percs})
        if isinstance(model, sklearn.discriminant_analysis.LinearDiscriminantAnalysis):
            if 'solver' in params: grid_params.update({'clf__solver':['svd','lsqr','eigen']})
        if 'n_neighbors' in params: grid_params.update({'clf__n_neighbors':groups})
        if 'weights' in params: grid_params.update({'clf__weights':['uniform','distance']})
        if isinstance(model, sklearn.neighbors.KNeighborsClassifier):
            if 'metric' in params: grid_params.update({'clf__metric':['euclidean','manhattan','cityblock','minkowski']})
        if 'p' in params: grid_params.update({'clf__p':[1, 2]})
        if 'C' in params: grid_params.update({'clf__C':pot10})
        if 'penalty' in params: grid_params.update({'clf__penalty':['l1','l2','elasticnet','none']})
        if 'multi_class' in params: grid_params.update({'clf__multi_class':['auto','ovr','multinomial']})
        if 'estimator__C' in params: grid_params.update({'clf__C':pot10})
        if 'estimator__penalty' in params: grid_params.update({'clf__penalty':['l1','l2','elasticnet','none']})
        if 'estimator__multi_class' in params: grid_params.update({'clf__multi_class':['auto','ovr','multinomial']})
        if 'kernel' in params: grid_params.update({'clf__kernel':['linear','rbf','sigmoid']})
        if self.strategy == 'classification':
            if 'alpha' in params: grid_params.update({'clf__alphas':alphas})
            if not isinstance(model, sklearn.ensemble.GradientBoostingClassifier):
                if 'loss' in params: grid_params.update({'clf__loss':['hinge','log','modified_huber','squared_hinge','perceptron']})
                if 'criterion' in params: grid_params.update({'clf__criterion':['gini','entropy']})
        if self.strategy == 'regression':
            if 'loss' in params: grid_params.update({'clf__loss':['ls','lad','huber']})
        if 'max_depth' in params: grid_params.update({'clf__max_depth':groups[:-1]})
        if 'min_samples_leaf' in params: grid_params.update({'clf__min_samples_leaf':groups})
        if 'n_estimators' in params: grid_params.update({'clf__n_estimators':pot10})
        if 'n_clusters' in params: grid_params.update({'clf__n_clusters':groups})
        if 'n_components' in params: grid_params.update({'clf__n_components':groups})
        if not isinstance(model, sklearn.cluster.DBSCAN) and self.strategy == 'clustering':
            if 'algorithm' in params: grid_params.update({'clf__algorithm':['full','elkan']})
        if 'eps' in params: grid_params.update({'clf__eps':quarts})
        if 'min_samples' in params: grid_params.update({'clf__min_samples':samples})
        if 'linkage' in params: grid_params.update({'clf__linkage':['ward','complete','average','single']})

        if 'k' in fixed.keys():
            if 'n_clusters' in params: grid_params.update({'clf__n_clusters':[fixed['k']]})
            if 'n_components' in params: grid_params.update({'clf__n_components':[fixed['k']]})
            if 'n_neighbors' in params: grid_params.update({'clf__n_neighbors':[fixed['k']]})
        
        # temporary, use this to improve the gridsearch process
        print('Available hyper-parameters', params)
        grid_params = [grid_params]
        print(grid_params, '\n')

        if self.scalerX != 'None':
            pipe = Pipeline([('scl', self.scalerX), ('clf', model)])
        else:
            pipe = Pipeline([('clf', model)])
        
        if self.strategy == 'regression': # and len(grid_params) > 0:
            scores = ['neg_mean_squared_error']
            # scores = ['neg_mean_squared_error']
            for score in scores:
                kfolds = KFold(n_splits=2, shuffle=True)
                cv = kfolds.split(self.X_train, self.y_train)
                gs = GridSearchCV(estimator=pipe, param_grid=grid_params, scoring=score, cv=cv, error_score=0.0)
                gs.fit(self.X_train, self.y_train)
            self.best_model = gs.best_estimator_

        elif self.strategy == 'classification': # and len(grid_params) > 0:
            scores = ['accuracy']
            # scores = ['accuracy', 'recall_macro', 'precision_macro']
            for score in scores:
                kfolds = StratifiedKFold(n_splits=2, shuffle=True)
                cv = kfolds.split(self.X_train, self.y_train)
                gs = GridSearchCV(estimator=pipe, param_grid=grid_params, scoring=score, cv=cv, error_score=0.0)
                gs.fit(self.X_train, self.y_train)
            self.best_model = gs.best_estimator_

        elif self.strategy == 'clustering': # and len(grid_params) > 0:
            best = model
            best_result = 0
            k = 2
            scores = ['euclidean']
            for score in scores:
                # lets create our own GridSearch for clustering problems
                my_dict = grid_params[0]
                allParams = sorted(my_dict)
                combinations = product(*(my_dict[param] for param in allParams))
                for comb in combinations:
                    new_params = {}
                    for i, param in enumerate(allParams):
                        if param == 'n_clusters' or param == 'n_components':
                            k = comb[i]
                        new_params.update({param: comb[i]})

                    # update all tuning parameters and run 
                    pipe.set_params(**new_params)
                    bandwidth = estimate_bandwidth(self.X_train, quantile=0.3)
                    connectivity = kneighbors_graph(self.X_train, n_neighbors=k, include_self=False)
                    connectivity = 0.5 * (connectivity + connectivity.T)
                    pipe.fit(self.X_train)
                    if hasattr(pipe, 'labels_'):
                        y_pred = pipe.labels_.astype(np.int)
                    else:
                        y_pred = pipe.predict(self.X_train)
                    result = silhouette_score(self.X_train, y_pred, metric=score)
                    if result > best_result:
                        best_result = result
                        best = pipe

            self.best_model = best
            self.best_model.best_params_ = new_params

        if printt:
            print(self.report_width * '*', '\n*')
            print('* HYPER-PARAMETER TUNING REPORT\n*')
            print("* SCORING METHOD: %s" % score)
            if self.strategy == 'regression':
                print('* BEST SCORE: %.3f' % (-gs.best_score_))
                print('* BEST PARAMS:', gs.best_params_)
            if self.strategy == 'classification':
                print('* BEST SCORE: %.1f %%' % (100 * gs.best_score_))
                print('* BEST PARAMS:', gs.best_params_)
            if self.strategy == 'clustering':
                print('* BEST SCORE (silhouette): %.2f' % (best_result))
                print('* BEST PARAMS:', self.best_model.best_params_)
            print('*\n', self.report_width * '*')
            print(self.best_model)

        return None 


    # given a model name, evaluate y_hat/y_pred/clusters and the overall performance of such model
    def optimize(self, model_name, printt=True, graph=False, xy=(0,0), fixed={}):
        self.graphs_model = []
        size = self.graph_width
        model = self.models[model_name]
        self.boost(model, printt, fixed=fixed) # grid search hyper parameters for this model
        
        if self.strategy == 'regression':
            X, y = self.X_test, self.y_test # evaluate using the test subset
            y_hat = self.best_model.predict(X)
            # show residual analysis
            result = self.residual(y, y_hat, model_name, printt, graph)
            if graph:
                # show the correlation between y and y_hat
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Model Overall Performance')
                plt.scatter(y, y_hat, color='g')
                viewer = lr()
                plt.plot(y, viewer.fit(y, y_hat).predict(y), color='k')
                plt.xlabel('Observed')
                plt.ylabel('Predicted')
                self.graphs_model.append(fig)
                plt.show()
            return result

        elif self.strategy == 'classification':
            X, y = self.X_test, self.y_test # evaluate using the test subset
            y_pred = self.best_model.predict(X)
            report = classification_report(y, y_pred, output_dict=True)
            sample_size = len(y_pred)
            if printt:
                print('\n')
                print(self.report_width * '*', '\n*')
                print('* MODEL PERFORMANCE \n*')
                print('* MODEL NAME: ', model_name)
                print('* TEST SAMPLE SIZE: ', sample_size)
                print('* ACCURACY: ', round(accuracy_score(y, y_pred)*100, 1), '%')
                print('* ')
                print(self.report_width * '*', '\n')
                if not graph:
                    display(pd.DataFrame(report).T)
            if graph:
                fig, ax = plt.subplots(figsize=(size, 0.3 * size))
                plt.title('Confusion Matrix')
                sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap='YlGn', fmt='d',)
                plt.xlabel('Predicted')
                plt.ylabel('True Class')
                self.graphs_model.append(fig)
                plt.show()
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Classification Report')
                sns.heatmap(pd.DataFrame(report).iloc[0:3].T, annot=True, vmin=0, vmax=1, cmap='BrBG', fmt='.2g')
                plt.xlabel('Score')
                self.graphs_model.append(fig)
                plt.show()
            return 'Accuracy: ' + str(round(accuracy_score(y, y_pred)*100, 1)) + '%'
        
        elif self.strategy == 'clustering':
            X = np.array(self.X_train) # use the same dataset to show the final model
            self.best_model.fit(X)
            if hasattr(self.best_model, 'labels_'):
                y_pred = self.best_model.labels_.astype(np.int)
            else:
                y_pred = self.best_model.predict(X)
            score = silhouette_score(X, y_pred, metric='euclidean')
            sample_size = len(y_pred)
            if printt:
                print('\n')
                print(self.report_width * '*', '\n*')
                print('* MODEL PERFORMANCE \n*')
                print('* MODEL NAME: ', model_name)
                print('* TEST SAMPLE SIZE: ', sample_size)
                print('* SILHOUETTE: ', round(score, 2))
                print('* ')
                print(self.report_width * '*', '\n')
            if graph:
                fig, ax = plt.subplots(figsize=(size, 0.5 * size))
                plt.title('Cluster Segmentation')
                plt.scatter(X[:, xy[0]], X[:, xy[1]], c=y_pred, s=50, cmap='viridis')
                plt.xlabel(self.X_train.columns[xy[0]])
                plt.ylabel(self.X_train.columns[xy[1]])
                self.graphs_model.append(fig)
                plt.show()
            return 'Silhouette: ' + str(round(score, 2))





# Rubia Models demo cases

def selectDemo(id):
    if id == 0:
        data, meta = scipy.io.arff.loadarff('dataset/scene_arff.arff')
        df = pd.DataFrame(data)
        y_cols = ['Beach']
        #y_cols = ['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']
        ignore_cols = ['Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']
    elif id == 1:
        df = pd.read_csv('dataset/Advertising.csv', index_col=0)
        y_cols = ['sales']
        ignore_cols = []
    elif id == 2:
        df = pd.read_csv('dataset/SAheart.csv')
        y_cols = ['chd']
        ignore_cols = []
    elif id == 3:
        df = pd.read_csv('dataset/pima-indians-diabetes.csv')
        y_cols = ['Class']
        ignore_cols = []
    elif id == 4:
        df = pd.read_excel('dataset/sample.xlsx')
        y_cols = ['g1']
        ignore_cols = ['g2', 'y', 'yr']       
    elif id == 5:
        df = pd.read_excel('dataset/sample.xlsx')
        y_cols = ['yr']
        ignore_cols = ['g1', 'g2', 'y']    
    elif id == 6:
        df = pd.read_csv('dataset/iris.csv')
        y_cols = []
        ignore_cols = ['species']    
    elif id == 7:
        df = pd.read_csv('../../../bigdata/jet/full_data.csv')
        y_cols = ['class']
        ignore_cols = ['class']   
    else:
        df = pd.read_csv('dataset/iris.csv')
        y_cols = ['species']
        ignore_cols = []
    return df, y_cols, ignore_cols

run_demo = False
priority = 'NN'
#priority = 'Classical'
id = 1 # -1 classifier multi | 4 binary | 1 regressor | 6 clustering
graph = False
graphm = True
balance_tol = 0.3
order = 1
ncomponents = 2
enc = 'LabelEncoder'
xy = (0, 1)
fixed = {'k': 3}
if run_demo:
    # load data as a pandas.dataframe object and pass it to the class
    df, y_cols, ignore_cols = selectDemo(id)

    # load the class rubia_models and show important info about the dataset
    # flag debug mode to True to show warning messages
    rm = rubia_models(df, debug=False)
    rm.describe(rm.data_raw)

    # columns listed as ignored will be discarded while modeling
    # flag graph to true to show some exploratory and correlation graphs on the dataset
    rm.explore(rm.data_raw, y_cols, ignore_cols, graph=graph) #updates X, y, M

    # encode every column of type object or string to categorical numbers
    if priority == 'Classical':
        rm.encode(encoder='LabelEncoder', who='both') # classical only accepts single output right now
    else:
        if enc == 'LabelEncoder':
            rm.encode(encoder='LabelEncoder', who='both')
        else:
            rm.encode(encoder='LabelEncoder', who='X')
            rm.encode(encoder='OneHotEncoder', who='Y')
        y_cols = list(rm.y.columns) # update y_cols after the OneHotEncoder

    # this method makes an auto balance for each class, using the minority class
    # only applies if the dataset size variation is under tolerance value
    rm.balance(balance_tol, rm.M, y_cols, ignore_cols)

    # add higher level and interaction terms to the model
    # be carefull when using higher level terms and graphs together, less powerfull hardware can bottleneck with higher complexity
    rm.addTerms(rm.X, rm.y, levels=order, interaction=False, root=False)
    rm.explore(rm.M, y_cols, ignore_cols, graph=graph) #updates X, y, M
    y_cols = list(rm.y.columns) # update y_cols after an exploratory that could remove constant columns

    # analyse if this is a regression, classification or clustering problem and evaluate some models
    # when y is float or has more then 10 different classes, the algorithm turns into a regression algorithm automatically
    # else it will perform a classification modeling
    # in multilevel problems, if one of the ys is identified as regression, then the entire process is set to regression mode
    # this routine also drops any column of constant value, if it exists
    rm.analyse(y_cols, priority=priority)

    # dimensionality reduction
    if ncomponents > 0: 
        rm.redux(k=ncomponents, mode='pca', transform='MinMax')
        print('Explained variance (%)', rm.scalerX_pca.explained_variance_ratio_.sum())
    elif len(rm.X.columns) > 10: 
        rm.redux(k=10)

    # evaluate the performance of a mix of models
    alphas = 10 ** np.linspace(10, -2, 100) * 0.5
    rm.evaluate(test_size=0.3, transformX='Standard', transformY='None', folds=10, alphas=alphas, xy=xy, graph=graphm)

    # apply tuning to the best models
    if rm.strategy[-3:] != '_nn':
        rm.optimize(str(rm.report_performance.Model.iloc[0]), graph=graphm, xy=xy, fixed=fixed)





# To get all coefficients for a given model:
#   lassor.coef_, lassocvr.coef_, rr.coef_, rcvr.coef_
#   rfc.feature_importances_
#   logit.classes_, coef_, intercept_, n_iter_
#   nbc.class_count_, class_prior_, classes_, sigma_, theta_
#   ldac.explained_variance_ratio_



# TO DO

# break the searchANN function in smaller functions
# add weights (new column) to NN results (add time, accuracy, f1score, accuracy evolution)
# implement number of generations and genetic to the NN booster

# implement multioutput (not planned)
# adjust redux(k='auto') to calculate the optimal value for k (not planned)
# add help menu with highlights for each model type, pros and cons (not planned)
# weight CV and model constraints while choosing the best model type, for similar performances (not planned)
# add metric parameter to the boost method, spefically for clustering (not planned)

