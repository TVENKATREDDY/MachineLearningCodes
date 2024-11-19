# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:08:10 2024

@author: 91807
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
house=pd.read_csv(r'C:\Venkat\Python\Practice_Material\13th Nov - SLR\SLR - House price prediction\House_data.csv')
print(house.head(10))
space=house['sqft_living']
price=house['price']
x=np.array(space).reshape(-1,1)
y=np.array(price)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state=0)

#from sklearn.cross_decomposition import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)


pred=regressor.predict(xtest)

m_slope=regressor.coef_
print(m_slope)

c_intercept=regressor.intercept_
print(c_intercept)

plt.scatter(xtrain, ytrain,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title('Visuals for tarinining Dataset')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()


#Visuals for Test records
plt.scatter(xtest, ytest, color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title('Visuals for test Data')
plt.xlabel("space")
plt.ylabel("Prices")
plt.show()

print(regressor.score(xtrain,ytrain))
print()