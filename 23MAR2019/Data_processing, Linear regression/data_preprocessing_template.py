# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:08:16 2017

@author: Techspecialist
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Data.csv')
#Creating Matrix of features for the independent variables (Country,Age,Salary )
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
"""
# Taking care of missing values in a dataset. First import class from a appropriate library
# the class to be imported here is the 'Imputer'
# next u create an object for the class. I am calling the object 'imputer'
# Next is to fit the object 'imputer' into the data we are targeting
# In this case, it'd be x
# The value 'x[:, 1:3]' sets the columns to take the 2nd column (i.e index 1)..
# and the third column which is index 2 that is represented by '3'.
# 3 is imputed instead of '2' because it shows the upper bound of d data that would not be taken.
# So for eg, if ur going to take the 3rd and 5th column in a dataset, then the representation would be:
# 'x[:, 2,6]'
# Next is to replace the missing data by the chosen missing_values parameter (in this case 'mean')
# for axis, 0= column, 1 = row"""

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:,1:3])


#practice: For median
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='median',axis = 0)
imputer = imputer.fit_transform(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#practice: For Categorization data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

#practice: Splitting dataset into Training and test sets
# As always you first import the necesary libary
#this is the final part of this session
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

