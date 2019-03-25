# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:00 2019

@author: BIFOLA
"""

#importing the libraries
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import os

df = pd.read_csv('50_Startups.csv')

x = df.iloc[:,:-1]
y = df.iloc[:, 4]


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
x.iloc[:,3] = encoder.fit_transform(x.iloc[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

x = x[: , 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

regressor.score(x_test , y_pred)

from sklearn.metrics import r2_score
r2_score(y_test , y_pred)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))





import statsmodels.formula.api as sm

x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[: , [0,1,2,3,4,5]]
regressor.OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor.OLS.summary()

x_opt = x[: , [0,1,3,4,5]]
regressor.OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor.OLS.summary()

x_opt = x[: , [0,3,4,5]]
regressor.OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor.OLS.summary()

x_opt = x[: , [0,3,5]]
regressor.OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor.OLS.summary()

x_opt = x[: , [0,3]]
regressor.OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor.OLS.summary()


