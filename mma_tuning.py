'''
Programmer: Aniket Panda
Class: CSC 580
Description: This file attempts to improve upon mma_features.py with
hyperparameter tuning. 10-fold cross-validation is used to search for
the optimal value of a single hyperparameter for each model type. For
those models which benefit from tuning on validation, the optimal
hyperparameter values are used when evaluating with the test set. It is
advised not to rerun cross-validation for the SVM or the neural net as
those models take very long to run. It is recommended to comment those
code blocks out.
'''

# importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras import layers, models, optimizers
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from scikeras.wrappers import KerasClassifier

'''
Import and clean data for use in models
'''

data_df =  pd.read_csv('data.csv')
# additional features added
data_df = data_df[['Winner', 'title_bout','R_wins','R_losses','R_draw',
    'B_wins','B_losses','B_draw','B_current_win_streak',
    'B_current_lose_streak','R_current_win_streak',
    'R_current_lose_streak','B_Stance','B_Height_cms','B_Reach_cms',
    'R_Stance','R_Height_cms','R_Reach_cms',
    'B_avg_SIG_STR_landed','B_avg_opp_SIG_STR_landed',
    'R_avg_SIG_STR_landed','R_avg_opp_SIG_STR_landed',
    'R_avg_TD_landed','R_avg_opp_TD_landed',
    'B_avg_TD_landed','B_avg_opp_TD_landed',
    'R_win_by_KO/TKO','R_win_by_Submission',
    'B_win_by_KO/TKO','B_win_by_Submission','B_age','R_age']]
df = data_df.dropna()


columns=df.select_dtypes(include='object').columns
columns

map_stance = {'Orthodox': 0, 'Switch': 1, 'Southpaw': 2, 'Open Stance': 3}
df['B_Stance'] = df['B_Stance'].replace(map_stance)
df['R_Stance'] = df['R_Stance'].replace(map_stance)

map_winner = {'Red': 0, 'Blue': 1, 'Draw': 2}
df['Winner'] = df['Winner'].replace(map_winner)

df.drop(columns=df.select_dtypes(include='bool').columns, inplace=True)

df['Winner'].unique()
df = df[df['Winner'] != 2]
X = df.drop(columns=['Winner'])
Y = df['Winner']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print("\nTraining size = " + str(X_train.shape[0]))
print("Testing size = " + str(X_test.shape[0]))

'''
Model Training and Evaluation on Data using k-fold cross validation
'''
seed = 404
np.random.seed(seed)

# K-Nearest Neighbors
print('\nKNN ave. K-fold scores:')
pars = [5,10,20,40]
for par in pars:
    knn = KNeighborsClassifier(n_neighbors=par)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_score = cross_val_score(knn, X_train, y_train.values.ravel(), cv=kfold)
    knn_score = cv_score.mean()
    print(par, '\t', knn_score)

# Decision Tree
print('\nDT ave. K-fold scores:')
pars = ['gini', 'entropy']
for par in pars:
    dt = tree.DecisionTreeClassifier(random_state = 1, criterion=par)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_score = cross_val_score(dt, X_train, y_train.values.ravel(), cv=kfold)
    dt_score = cv_score.mean()
    print(par, '\t', dt_score)

# Logistic Regression
print('\nLR ave. K-fold scores:')
pars = ['l1', 'l2', 'elasticnet', 'none']
for par in pars:
    lr = LogisticRegression(max_iter = 10000, penalty=par, solver='saga')
    if par == 'elasticnet':
        lr = LogisticRegression(max_iter = 10000, penalty=par, solver='saga', l1_ratio=0.5)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_score = cross_val_score(lr, X_train, y_train.values.ravel(), cv=kfold)
    lr_score = cv_score.mean()
    print(par, '\t', lr_score)

# Support Vector Machine
print('\nSVM ave. K-fold scores:')
pars = ['linear', 'poly', 'rbf', 'sigmoid']
for par in pars:
    svc = SVC(kernel=par, probability = True)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_score = cross_val_score(svc, X_train, y_train.values.ravel(), cv=kfold)
    svc_score = cv_score.mean()
    print(par, '\t', svc_score)

# model creation method for the neural net
def create_model():
    model = Sequential()
    
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Feedforward Neural Network
print('\nNN ave. K-fold scores:')
pars = [5,10,20,40]
for par in pars:
    model = KerasClassifier(model=create_model, epochs=150, batch_size=par, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    cv_score = cross_val_score(model, X_train, y_train, cv=kfold)
    nn_score = cv_score.mean()
    print(par, '\t', nn_score)

'''
Model Training and Evaluation with test set and best hyperparameters
'''

# Decision Tree
dt = tree.DecisionTreeClassifier(random_state = 1, criterion='entropy')
dt_model = dt.fit(X_train, y_train.values.ravel())
y_pred = dt_model.predict(X_test)
print('Decison Tree Model Accuracy (on testing set): ')
print(accuracy_score(y_test, y_pred))

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=20)
knn_model = knn.fit(X_train, y_train.values.ravel())
y_pred = knn_model.predict(X_test)
print('KNN Model Accuracy (on testing set): ')
print(accuracy_score(y_test, y_pred))

# Support Vector Model
svc = SVC(kernel='linear', probability = True)
svc_model = svc.fit(X_train, y_train.values.ravel())
y_pred = svc_model.predict(X_test)
print('SVM Model Accuracy (on testing set): ')
print(accuracy_score(y_test, y_pred),'\n')

