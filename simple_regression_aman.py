# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 11:04:30 2017

@author: dell 1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
#splitting the training and test set
from sklearn.cross_validation import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=1/3,random_state=0)
#fitting the simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(train_X,train_Y)
#predicting the test set results
Y_pred=regressor.predict(test_X)
#visualize the training set results
plt.scatter(train_X,train_Y,color='red')
plt.plot(train_X,regressor.predict(train_X),color='blue')
plt.title('salary vs experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
#visualize the test set results
plt.scatter(test_X,test_Y,color='red')
plt.plot(train_X,regressor.predict(train_X),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()