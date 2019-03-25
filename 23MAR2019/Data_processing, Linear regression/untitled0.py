# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:31:50 2019

@author: BIFOLA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


df = pd.read_csv("housing.csv")

x = df.drop(columns = "median_house_value")
y = df.loc[: , "median_house_value"]

df.info()
y.isnull().any()
x.isnull().any()

x["total_bedrooms"].isnull().sum()
q = x[x["total_bedrooms"].isnull()]

x["total_bedrooms"].mean()
x["total_bedrooms"].median()
x["total_bedrooms"].describe()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "median", axis = 0)
#x.iloc[: , 4:5] = imputer.fit_transform(x.iloc[: , 4:5])
x.iloc[: , 4:5] = imputer.fit_transform(x.iloc[: , 4:5])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
x.iloc[:,-1] = encoder.fit_transform(x.iloc[:,-1])
onehotencoder = OneHotEncoder(categorical_features = [-1])
x = onehotencoder.fit_transform(x).toarray()

encoder.classes_

onehotencoder.feature_indices_

#x = x[: , 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

regressor.score(x_test , y_pred)

from sklearn.metrics import r2_score
r2_score(y_test , y_pred)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

model = regressor.predict([[0,0,0,1,0,-122.23, 37.88, 53, 130, 160, 300, 180, 9.63]])

