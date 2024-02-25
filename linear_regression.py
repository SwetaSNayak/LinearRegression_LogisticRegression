import numpy as np
import pandas as pd
import math
import sys
import os
import random
    

#Todo : define necessary functions


def linear_regression(data,lr=0.00001,epoch=100000):
    #splitting the target and the context values
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values
    #y.reshape(y.size,1)# Target variable

    n_samples, n_features = X.shape
    #adding a extra column of 0's to the weights for calculation
    weights = np.zeros(n_features)
    bias = 0
    #print (X.shape)
    #print(y.shape)
    #print(y)

    for _ in range(epoch):

        y_pred = np.dot(X, weights) + bias
#derivatives of the weights
        d_weights = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        d_bias = (1 / n_samples) * np.sum(y_pred - y)
#   Updating the weights
        weights -= lr * d_weights
        bias -= lr * d_bias


    print(weights)
    print(bias)
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse
    """
    data: input data matrix
    return: rmse value
    """
    #Todo : fill code here



# do not modify this function
def load_data():

    filename = sys.argv[1]
    feature_matrix = pd.read_csv(filename)
    feature_matrix = feature_matrix.dropna()
    features = sys.argv[2:]
    #print(feature_matrix[features])
    #print(feature_matrix)
    return feature_matrix


if __name__ == "__main__":
    data = load_data()
   # print(data)
    #linear_regression(data)
    RMSE_SCORE = linear_regression(data)
    print("RMSE score is : ", RMSE_SCORE)
