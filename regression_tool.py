# -*- coding: utf-8 -*-

"""
Created on Tue Aug 27 10:00:34 2019

This is a program fits an XGBoost gradient boosting regression model, and then creates a windowed application to insert values that return predictions.

The program takes values for LSTAT, the % lower status of the population, and RM, the average number of rooms, to predict MEDV, the median value of the home.

@author: jsteven.raquel
"""
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor

import tkinter as tk
from functools import partial


# Reading in the data
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = boston.target

# Setting response and target variables
y = df['MEDV']
X = pd.DataFrame(boston.data, columns = boston.feature_names)
X = X[['RM', 'LSTAT']]

# split between training and testing sets, (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

# scaling the data
scaler = StandardScaler().fit(X_train)

# scaling the training and test splits and adding the column names back
X_train = scaler.transform(X_train)
X_train = pd.DataFrame(X_train, columns = X.columns)
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns = X.columns)

regr_XGB = XGBRegressor(random_state = 10)

# Hyper-parameter tuning
parameters_XGB = {'n_estimators': [10, 20, 50, 100, 400], 'max_depth': [2,5,7,9]}
grid_XGB = GridSearchCV(estimator = regr_XGB, param_grid = parameters_XGB, cv = 3)
model_XGB = grid_XGB.fit(X_train, y_train)

# Testing
# Creating new data
new_data = pd.DataFrame({'RM': [6], 'LSTAT': [14]})
# scaling new data
new_data = scaler.transform(new_data)
new_data = pd.DataFrame(new_data, columns = ['RM', 'LSTAT'])
# Prediction
model_XGB.predict(new_data)[0]

# Building the GUI application

def call_result(label_result, n1, n2):
    num1 = (n1.get())
    num2 = (n2.get())

    new_data = pd.DataFrame({'RM': [num1], 'LSTAT': [num2]})
    new_data = scaler.transform(new_data)
    new_data = pd.DataFrame(new_data, columns = ['RM', 'LSTAT'])

    result = model_XGB.predict(new_data)[0]
    label_result.config(text="Result is %d" % result)
    return

root = tk.Tk()
root.geometry('400x200+100+200')
root.title('Prediction with Regression Model')

number1 = tk.StringVar()
number2 = tk.StringVar()

label_Title = tk.Label(root, text = 'Prediction with Regression Model').grid(row = 0, column = 2)
label_Num1 = tk.Label(root, text = 'Enter the value for RM').grid(row = 1, column = 0)
label_Num2 = tk.Label(root, text = 'Enter the value for LSTAT').grid(row = 2, column = 0)
label_Result = tk.Label(root)
label_Result.grid(row = 7, column = 2)


entry_Num1 = tk.Entry(root, textvariable = number1).grid(row=1, column=2)
entry_Num2 = tk.Entry(root, textvariable = number2).grid(row=2, column=2)

call_result = partial(call_result, label_Result, number1, number2)
buttonCal = tk.Button(root, text = "Calculate", command = call_result).grid(row = 3, column = 0)
root.mainloop()