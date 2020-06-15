import numpy as np 
import pandas as pd 
import tensorflow as tf

#importing the dataset 
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

# One Hot Encoding the "Geography" column
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=0)

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier=Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6,activation='relu'))

# Adding the second hidden layer
classifier.add(Dense(units=6,activation='relu'))

# Adding the input layer
classifier.add(Dense(units=1,activation='sigmoid'))

# Compiling the classifier
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# Training the ANN on the Training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 20)

# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

















