# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:16:40 2024

@author: 91807
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
houseData=pd.read_csv(r'C:\Venkat\Python\Practice_Material\13th Nov - SLR\SLR - House price prediction\House_data.csv')

space=houseData['sqft_living']
price=houseData['price']

x=np.array(space).reshape(-1,1)
y=np.array(price)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=.8,random_state=0)

regressor=LinearRegression()
regressor.fit(xtrain,ytrain)

pred=regressor.predict(xtest)

comparison=pd.DataFrame({'Actual':ytest,'Predicted':pred})

bias=regressor.score(xtrain,ytrain)

variance=regressor.score(xtest,ytest)

print(f'Bias is: {bias}')
print(f'variance is: {variance}')
