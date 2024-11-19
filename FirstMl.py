# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r'C:\Venkat\Python\Practice_Material\11th Nov -ML\Data (1).csv')
print(data)
x=data.iloc[:,0:3].values
y=data.iloc[:,3]
from sklearn.impute import SimpleImputer # SPYDER 4
imputer=SimpleImputer()  # by default strategy is mean, to get median, use stratogy='median'
#imputer=SimpleImputer(strategy='mode')
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
labelencoder_x.fit_transform(x[:,0])
x[:,0]=labelencoder_x.fit_transform(x[:,0])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
