# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:05:18 2024

@author: 91807
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
HouseData=pd.read_csv(r'C:\Venkat\Python\Practice_Material\18th Nov - mlr\MLR\House_data.csv')

HouseData=HouseData.drop(['id','date'],axis=1)
print(HouseData.head(2))

x=HouseData.iloc[:,1:].values
y=HouseData.iloc[:,0].values

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

regressor=LinearRegression()

regressor.fit(xtrain,ytrain)

y_pred=regressor.predict(xtest)


x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]

regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
