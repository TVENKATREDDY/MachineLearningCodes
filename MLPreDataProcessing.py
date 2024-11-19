# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:08:36 2024

@author: 91807
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r'C:\Venkat\Python\Practice_Material\11th Nov -ML\Data (1).csv')
print(data)
X=data.iloc[:,:-1] # Independent Variable
Y=data.iloc[:,3]  # Dependent Variable
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
#imputer=imputer.fit(x[:,1:3])
iputer=imputer.fit(X[:,1:2])
x[:,1:3]=imputer.transform(x[:,1:3])
#imputer=imputer.fit(X[:,:-1])
#X[:,1:3]=imputer.transform(X[:,1:3])
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
labelencoder_x.fit_transform(x.iloc[:,0])
x.iloc[:,0]=labelencoder_x.fit_transform(x.iloc[:,0])

from sklearn.model_selection import train_test_split
x_tran,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
