# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:50:54 2024

@author: 91807
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

investmentData=pd.read_csv(r'C:\Venkat\Python\Practice_Material\18th- mlr\MLR\Investment.csv')
x=investmentData.iloc[:,:-1]  #Rest of them are independent variables
y=investmentData.iloc[:,4]  #Profit is dependent variable

x=pd.get_dummies(x,dtype=int)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

regressor=LinearRegression()
regressor.fit(xtrain,ytrain)

y_pred=regressor.predict(xtest)

m_slope=regressor.coef_  #Since we have 6 independent vars so 6 m_slopes will generate

print(m_slope)

c_intrecept=regressor.intercept_

print(c_intrecept)

x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)  #Adding constant to x variable
import statsmodels.api  as sm
x_opt=x[:,[0,1,2,3,4,5]]
regressr_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressr_OLS.summary()  #Prints Regression table, check P>|t| value, x4 has 0.990, if the 
# P value > 0.05 then reject the nul hypothesis, means we have to remove x4 from medel

# Now removed x4 from model and re run the model
x_opt=x[:,[0,1,2,3,5]]  # 4 is removed as it refers to x4
regressr_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressr_OLS.summary()  #check P value, x4 has .994 so remove x4 means '5' in next iteration

#this removing P>|t| is called backward elimination method using P vlaue

x_opt=x[:,[0,1,2,3]]  # 4 is removed as it refers to x4
regressr_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressr_OLS.summary() #Check x2 has 0.602, so remove x2 means '2'

x_opt=x[:,[0,1,3]]  # 4 is removed as it refers to x4
regressr_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressr_OLS.summary()  #check x2 value 0.060 remove that mean '3'

x_opt=x[:,[0,1]]  # 4 is removed as it refers to x4
regressr_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressr_OLS.summary()


#P value is '0', so x1 attribute we can choose from multiple attributes


