# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required Python libraries and create the datasets with study hours and marks. 2.Divide the datasets into training and testing sets.
2. Create a simple Linear Regression model and train it using the training data.
3. Use the trained model to predict marks on the testing data and display the predicted output. 

## Program:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("student_scores.csv")


print("First 5 rows of the dataset:")
print(df.head())
print("\nLast 5 rows of the dataset:")
print(df.tail())


X = df[["Hours"]].values   
Y = df["Scores"].values     


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0
)


regressor = LinearRegression()
regressor.fit(X_train, Y_train)


Y_pred = regressor.predict(X_test)


print("\nPredicted values:")
print(Y_pred)
print("\nActual values:")
print(Y_test)


plt.scatter(X_train, Y_train, label="Actual Scores")
plt.plot(X_train, regressor.predict(X_train), label="Regression Line")
plt.title("Hours Studied vs Marks (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.legend()
plt.grid(True)
plt.show()


plt.scatter(X_test, Y_test, label="Actual Scores")
plt.plot(X_test, regressor.predict(X_test), label="Regression Line")
plt.title("Hours Studied vs Marks (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.legend()
plt.grid(True)
plt.show()


mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("\nError Metrics:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

```
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: YOGESH.S
RegisterNumber:  212224230311

```

## Output:

<img width="411" height="336" alt="image" src="https://github.com/user-attachments/assets/924f5d6f-1ef1-4403-afe1-3b3613e41320" />

<img width="850" height="150" alt="image" src="https://github.com/user-attachments/assets/48b34dc7-248c-431c-951a-9b7d76902693" />

<img width="863" height="568" alt="image" src="https://github.com/user-attachments/assets/640ded27-77d8-4431-8b84-0a30eeee2d3e" />

<img width="923" height="583" alt="image" src="https://github.com/user-attachments/assets/7d09c966-2dde-4f35-8336-dd5e3dbff3d6" />

<img width="600" height="102" alt="image" src="https://github.com/user-attachments/assets/e04d7255-27c4-457b-9820-0d397584dc83" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
