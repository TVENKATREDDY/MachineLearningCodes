# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:40:53 2024

@author: 91807
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
salary=pd.read_csv(r'C:\Venkat\Python\Practice_Material\11th Nov -ML\DataSets\Salary_Data.csv')
#salary
X=salary.iloc[:,:-1] # ind ependent Variable(Years)
Y=salary.iloc[:,-1] # Dependent variable (Salary)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experince ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

print(regressor)

m_slope=regressor.coef_
print(m_slope)

c_intercept=regressor.intercept_
print(c_intercept)


y_12=m_slope * 12 + c_intercept
print(y_12)


print(salary.mean())
print(salary['Salary'].median())

print(salary['Salary'].mode())

print(salary.var())

print(salary.std())

#
from scipy.stats import variation
variation(salary.values)

variation(salary['Salary'].values)
print(salary.corr())


print(salary['Salary'].corr(salary['YearsExperience']))


salary.skew()

salary['Salalry'].skew()

print(salary.sem())


# Inferential Stats
#Z-score
import scipy.stats as stats

print(salary.apply(stats.zscore))

print(stats.zscore(salary['Salary']))

a=salary.shape[0]
b=salary.shape[1]

degree_freedom=a-b

#Sum of Squares regressor SSR
y_mean=np.mean(Y)

SSR=np.sum((Y_pred-y_mean)**2)
print(SSR)

y=Y[0:6]

SSE=np.sum((y-Y_pred)**2)
print(SSE)

#SST=SSR+SSE
#print(SST)

mean_total=np.mean(salary.values)

SST1=np.sum((salary.values-mean_total)**2)

print(SST1)


#R Square
r_square = SSR/SST

print(r_square)

print(regressor)

bias=regressor.score(X_train,Y_train)
print(bias)


variance=regressor.score(X_test,Y_test)
print(variance)

#if bias 94% and variance is 98%   good model

#if the bias 94% and variance is 40% high bias and low variance -- underfit model
#if the bias is 40 and variance is 94% thenn  low bias and high variance then overfitting