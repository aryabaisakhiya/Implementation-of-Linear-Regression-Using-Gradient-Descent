# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate
4.Plot the Cost function using Gradient Descent and generate the required graph.
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Arya Baisakhiya
RegisterNumber:  212222040019
*/


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1],color="cadetblue")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000) ")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  """
  take in a numpy array X,y theta and generate the cost function of using the
  in a linear regression model
  """
  m=len(y) #length of the training data
  h=x.dot(theta) #hypothesis
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err) #returning ]

data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta) #call the function

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    j_history.append(computeCost(x,y,theta))
  return theta,j_history

theta,j_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1],color="cadetblue")
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value)
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000,we predict a profit of $" +str(round(predict2,0)))
```

## Output:
1.Profit Prediction:
![271824429-364a60fa-689d-420c-8182-8fbb7c7b7325](https://github.com/aryabaisakhiya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393645/e12140db-3426-4487-b124-029e2e001b8e)
2.Function Output:
![271824465-75ed7cb1-4236-423e-9d87-bef66c87047d](https://github.com/aryabaisakhiya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393645/590b18ec-bc00-4ef1-9627-10aa4fc3dcf8)
3.Gradient Decent:
![271824504-91192bba-083a-46ee-bc65-3b74d27f99c7](https://github.com/aryabaisakhiya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393645/7baa0843-4be8-4714-bf91-3823e52001fe)
4.Cost Function Using Gradient Decent:
![271824577-0cea68d5-88ba-40de-a393-e28a378ac6af](https://github.com/aryabaisakhiya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393645/2fd11855-64bd-4421-bf69-5065bfd52aea)
5.Linear Regresion using Profit Prediction:
![271824636-a65c2b53-24ae-4ebd-8eff-9a381f804de6](https://github.com/aryabaisakhiya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393645/03c1257c-c10a-42c9-a8e0-7babee47c8f0)
6.Profit prediction for a population of 35,000:
![271824715-3e9d257a-8d61-4a6f-bd67-79f70a02cfb2](https://github.com/aryabaisakhiya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393645/eced6da3-d310-4e54-9e06-452f0e56c0f9)
7.Profit prediction for a population of 70,000:
![271824776-28d92d14-7909-4770-80ee-884c5813b2d8](https://github.com/aryabaisakhiya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393645/2d5e5a44-a25c-4611-936a-95266223351f)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
