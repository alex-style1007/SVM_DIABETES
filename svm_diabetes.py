import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets
import pandas as pd
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df= datasets.load_diabetes()

df.feature_names

df.DESCR

df.data

df.target

X = df.data[:, [2, 3]]

y=df.target
print('Class labels:', np.unique(y))

X

X[:,0]

plt.scatter(X[:,0],X[:,1],c=y)
plt.xlabel(df.feature_names[2])
plt.ylabel(df.feature_names[3])

plt.hist(X[:,0])

plt.hist(X[:,1])

#Normalización de datos a datos con media 0 y desviación 1
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

plt.hist(X[:,0])

plt.hist(X[:,1])

#Separando el dataset entre entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

clf = SVC(kernel='rbf', C=1.0, decision_function_shape="ovr",gamma=1, random_state=0)
clf.fit(X_train, y_train)

print ("Train - Confusion matrix : \n",metrics.confusion_matrix(y_train, clf.
predict(X_train)))

print ("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict
(X_train)))

print ("Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict
(X_test)))

print ("Train - classification report : \n", metrics.classification_report
(y_train, clf.predict(X_train)))

print ("Test - classification report : \n", metrics.classification_report
(y_test, clf.predict(X_test)))

from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [10, 1, 0.1, 0.01, 0.001],'kernel': ['linear','rbf','poly'], "degree":[2,3,4]}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose =3,scoring="accuracy",cv=3)

# fitting the model for grid search
grid.fit(X, y)

# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
