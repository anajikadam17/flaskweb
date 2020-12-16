# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:09:07 2020

@author: Anaji
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle

def clean(df):
    print(df.shape)
    df.dropna(inplace = True)
    # print(df.isnull().sum())

    # EDA
    
    # Date_of_Journey
    df["Journey_day"] = pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.day
    df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format = "%d/%m/%Y").dt.month
    df.drop(["Date_of_Journey"], axis = 1, inplace = True)
    
    # Dep_Time
    df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
    df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
    df.drop(["Dep_Time"], axis = 1, inplace = True)
    
    # Arrival_Time
    df["Arrival_hour"] = pd.to_datetime(df.Arrival_Time).dt.hour
    df["Arrival_min"] = pd.to_datetime(df.Arrival_Time).dt.minute
    df.drop(["Arrival_Time"], axis = 1, inplace = True)
    
    # Duration
    duration = list(df["Duration"])
    
    for i in range(len(duration)):
        if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
            if "h" in duration[i]:
                duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
            else:
                duration[i] = "0h " + duration[i]           # Adds 0 hour
    
    duration_hours = []
    duration_mins = []
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
        duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
    
    # Adding Duration column to test set
    df["Duration_hours"] = duration_hours
    df["Duration_mins"] = duration_mins
    df.drop(["Duration"], axis = 1, inplace = True)
    # print("Airline")
    # print("-"*75)
    # print(df["Airline"].value_counts())
    Airline = pd.get_dummies(df["Airline"], drop_first= True)
    
    # print("Source")
    # print("-"*75)
    # print(df["Source"].value_counts())
    Source = pd.get_dummies(df["Source"], drop_first= True)
    
    # print("Destination")
    # print("-"*75)
    # print(df["Destination"].value_counts())
    Destination = pd.get_dummies(df["Destination"], drop_first = True)
    
    # Route and Total_Stops are related to each other
    df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
    
    # Replacing Total_Stops
    df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
    
    # Concatenate dataframe --> test_data + Airline + Source + Destination
    df.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
    data = pd.concat([df, Airline, Source, Destination], axis = 1)

    print("Shape of data : ", data.shape)
    return data


train_df=pd.read_excel('data/Data_Train.xlsx')
train = clean(train_df)
test_df=pd.read_excel('data/Test_set.xlsx')
test = clean(test_df)
# print(train.head())
X = train.drop(["Price"], axis = 1)
y = train.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
rf_random = RandomForestRegressor(n_estimators= 700, min_samples_split= 15, min_samples_leaf = 1, max_features= 'auto', max_depth= 20)
rf_random.fit(X_train, y_train)
prediction = rf_random.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
# open a file, where you ant to store the data
file = open('model/flight_price_rf1.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)
model = open('model/flight_price_rf1.pkl','rb')
forest = pickle.load(model)
y_prediction = forest.predict(X_test)
print(metrics.r2_score(y_test, y_prediction))


# =============================================================================
# (10683, 11)
# Shape of data :  (10682, 30)
# (2671, 10)
# Shape of data :  (2671, 28)
# MAE: 1164.5822360545721
# MSE: 4050861.4385031047
# RMSE: 2012.6751944869557
# 0.8121300966103204
# =============================================================================
