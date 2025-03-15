# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.


```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Jayaseelan U
RegisterNumber:  212223220039
*/
```
## Program:
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores (1).csv')
print(dataset.head())
dataset=pd.read_csv('student_scores (1).csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,reg.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```


## Output:
![Screenshot 2025-03-15 161502](https://github.com/user-attachments/assets/28017f8f-3c6a-4f11-bb69-2df1acea9841)
![Screenshot 2025-03-15 161538](https://github.com/user-attachments/assets/279a0bb7-b3af-44af-82b0-212c8ca2af02)
![Screenshot 2025-03-15 161552](https://github.com/user-attachments/assets/f7b1c0e7-7930-4345-b2c2-0e88bedcd6c0)
![Screenshot 2025-03-15 161605](https://github.com/user-attachments/assets/cdfc5b43-47b2-45e9-b804-00d026d77db9)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
