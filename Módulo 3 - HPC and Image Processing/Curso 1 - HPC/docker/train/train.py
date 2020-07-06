import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, classification_report

from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf


url="https://raw.githubusercontent.com/harpreetSinghGuller/Data-Science-R/master/SAHeart.csv"
df2=pd.read_csv(url)

X2 = df2.drop(['row.names'], axis=1)

X = X2.drop(['chd'], axis=1)
y = df2['chd']

le = preprocessing.LabelEncoder()
X['famhist'] = le.fit_transform(X['famhist'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)

sgd_clf = DecisionTreeClassifier()
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

joblib.dump(sgd_clf, '/models/classifier.pkl', compress=1)


# executar qualquer coisa no tf para confirmar se a GPU está sendo usada
# watch -n 1 nvidia-smi
print(tf.eye(10000,10000))