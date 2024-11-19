# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:24:04 2024

@author: 91807
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
eSalary=pd.read_csv(r'C:\Venkat\Python\Practice_Material\11th Nov -ML\DataSets\Salary_Data.csv')
eSalary.head(10)

#Split data into 2, Dependent Variable 'Y' and Independent variable 'X'
x=eSalary.iloc[:,:-1].values
y=eSalary.iloc[:,1].values


#Split the data into training and tetsing
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)


#Train the model
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)


#Predcit the test
y_pred=regressor.predict(xtest)

#Comparison

comparison=pd.DataFrame({'Actual':ytest,'Predicted':y_pred})

m_slope=regressor.coef_

c_intercept=regressor.intercept_

#Predict the salary of 12 and 20 years of Experience
y_12=regressor.predict([[12]])
y_20=regressor.predict([[20]])

ym_12=m_slope*12+c_intercept
print(f'Manual Prediction of 12 years experience is: ${ym_12}')

print(f'Predcited Salary of 12 years of Experience is: ${y_12[0]:,.2f}')
print(f'Predicted salary of 20 years of Experience is : ${y_20[0]:,.2f}')


#Check model peeformace
bias=regressor.score(xtrain,ytrain)
variance=r

print(eSalary.mean())
print(eSalary['Salary'].mean())
print(eSalary.mod()[0])
print(eSalary.median())
print(eSalary.std())
print(eSalary['Salary'].std())
print(eSalary['Salary'].mean())
print(eSalary['Salary'].std()/eSalary['Salary'].mean())  #Variation S/X(bar)
from scipy.stats import variation
print(variation(eSalary))
print(eSalary.var())
print(variation(eSalary['Salary']))  #S/x{bar}
print(eSalary['Salary'].var())

print(eSalary.corr())

print(eSalary['Salary'].corr(eSalary['YearsExperience']))


print(eSalary.skew())

print(eSalary.sem())

import scipy,stats as stats
print(stats.zscore(eSalary['Salary']))

a=eSalary.shape[0]
b=eSalary.shape[1]

degree_freedom=a-b

#SSR Sum of Square of Regressor
y_mean=np.mean(y[0:6])
SSR=sum((y_pred-y_mean)**2)
print(SSR)

#SSE Sum of Square of Error
Y=y[0:6]
SSE=np.sum((Y-y_pred)**2)
print(SSE)

SST=SSE+SSR
print(SST)

mean_total=np.mean(eSalary[0:6].values)
print(mean_total)
SST1=np.sum((eSalary[0:6].values-mean_total)**2)
print(SST1)
print(SST)

r_square=SSR/SST
print(r_square)

print(regressor)

bias=regressor.score(xtrain,ytrain)
variance=regressor.score(xtest,ytest)
print(bias)
print(variance)

#Visualise Training data
plt.scatter(xtrain,ytrain,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title('Salary vs Experience (Train Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visuallize Test Data
plt.scatter(xtest,ytest,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title('Salary vs Experience (Test Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#if bias 94% and variance is 98%   good model

#if the bias 94% and variance is 40% high bias and low variance -- underfit model
#if the bias is 40 and variance is 94% thenn  low bias and high variance then overfitting