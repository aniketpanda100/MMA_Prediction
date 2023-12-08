'''
Programmer: Aniket Panda
Class: CSC 580
Description: This is a copy of the original MMA prediction project found at
https://www.kaggle.com/code/dbsimpson/mma-ml-model-fight-predictions-ufc-259.
The only modifications are that the XGBoost and RandomForest models have 
been eschewed. Additionally, a different, but largely backwards compatible
version of KerasClassifier was used that requires the scikeras library, since
the original library was deprecated. Rather than using the small test set 
provided by the original project, I used the test set from the new, updated
file for ease of comparison with the other scripts I created.
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
data_df = data_df[['Winner', 'title_bout','R_wins','R_losses','R_draw',
    'B_wins','B_losses','B_draw','B_current_win_streak',
    'B_current_lose_streak','R_current_win_streak',
    'R_current_lose_streak','B_Stance','B_Height_cms','B_Reach_cms',
    'R_Stance','R_Height_cms','R_Reach_cms']]
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
Model Training and Evaluation 
'''
seed = 404
np.random.seed(seed)

# Gaussian Naive Bayes
GNB = GaussianNB()
GNB_model = GNB.fit(X_train, y_train.values.ravel())
y_pred = GNB_model.predict(X_test)
print('\nGaussian Naive Bayes Model Accuracy (on testing set): ')
print(accuracy_score(y_test, y_pred))

# Decision Tree
dt = tree.DecisionTreeClassifier(random_state = 1)
dt_model = dt.fit(X_train, y_train.values.ravel())
y_pred = dt_model.predict(X_test)
print('Decison Tree Model Accuracy (on testing set): ')
print(accuracy_score(y_test, y_pred))

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train.values.ravel())
y_pred = knn_model.predict(X_test)
print('KNN Model Accuracy (on testing set): ')
print(accuracy_score(y_test, y_pred))

# Logistic Regression
lr = LogisticRegression(max_iter = 10000)
lr_model = lr.fit(X_train, y_train.values.ravel())
y_pred = lr_model.predict(X_test)
print('Logistic Regression Model Accuracy (on testing set): ')
print(accuracy_score(y_test, y_pred))

# Support Vector Model
svc = SVC(probability = True)
svc_model = svc.fit(X_train, y_train.values.ravel())
y_pred = svc_model.predict(X_test)
print('SVM Model Accuracy (on testing set): ')
print(accuracy_score(y_test, y_pred),'\n')

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
nn = KerasClassifier(model=create_model, epochs=150, batch_size=10, verbose=0)
nn_model = nn.fit(X_train, y_train)
y_pred = nn_model.predict(X_test)
print('\nNeural Net Model Accuracy (on testing set): ')
print(accuracy_score(y_test, y_pred), '\n')
