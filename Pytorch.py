import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Data (preparing and loading)

# Data can be almost anything 

# linear regression 

# Known Parameters

weight = 0.7
bias = 0.3

# Create
start = 0 
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

#Generalisation : the ability to adapt to a unseen situation

#Create a training/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

def Plot_predictions(train_data = X_train,
                     train_label = y_train,
                     test_data = X_test,
                     test_label = y_test,
                     predictions = None):
    plt.figure(figsize=(10, 7)) # plots Data test Data and compares predictions


    # Plot Training Data in blue
    plt.scatter(train_data, train_label, c="b", s=4,label="training data")

    #Plot test Data in green
    plt.scatter(test_data, test_label, c="g", s=4,label="testing data")

    #Are there predictions 
    if predictions is not None:
        #Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label = predictions)
    
    # Show the legend
    plt.legend(prop={"size" : 14})


Plot_predictions(X_train, y_train, X_test, y_test)
plt.show()