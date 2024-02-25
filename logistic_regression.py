import numpy as np
import pandas as pd
import math
import sys
import os
    

#todo define necessary functions

def sigmoid(x):
    return 1/(1+np.exp(-x))
def logistic_regression(xtrain, ytrain, xtest, ytest,lr=0.0015,iter=5000):


    xtrain = xtrain.values
    ytrain = ytrain.values
    xtest = xtest.values
    ytest = ytest.values

# Reshaping the data for broadcasting
    xtrain = xtrain.T
    ytrain = ytrain.reshape(1, xtrain.shape[1])
    xtest = xtest.T
    ytest = ytest.reshape(1, xtest.shape[1])

    #print(xtrain.shape)
    #print(ytrain.shape)
    #print(xtest.shape)
    #print(ytest.shape)

    m=xtrain.shape[1]
    n=xtrain.shape[0]
# W=weights, B=bias
    W=np.zeros((n,1))
    B=0

    for i in range(iter):
        z=np.dot(W.T,xtrain)+B
        A=sigmoid(z)

# cost function calculation
        cost= -(1 / m) * np.sum(ytrain * np.log(A) + (1 - ytrain) * np.log(1 - A))
#finding the derivatives of the weights

        dW= (1/m) * np.dot(A-ytrain,xtrain.T)
        dB= (1/m) * np.sum(A-ytrain)

# Updating the weights

        W= W- lr*dW.T
        B= B- lr *dB
# accuracy rate calculation
        test_predictions = sigmoid(np.dot(W.T,xtest) + B)
        accuracy_score=(1-np.sum(np.absolute(test_predictions-ytest))/ytest.shape[1])*100



    return accuracy_score

# do not modify this function
def load_data():
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    train_feature_matrix = pd.read_csv(train_filename) 
    test_feature_matrix = pd.read_csv(test_filename)
    train_feature_matrix = train_feature_matrix.dropna()
    test_feature_matrix = test_feature_matrix.dropna()
    X_TRAIN = train_feature_matrix.iloc[:, :len(train_feature_matrix.columns)-1] 
    Y_TRAIN = train_feature_matrix.iloc[:, -1]
    X_TEST = test_feature_matrix.iloc[:, :len(test_feature_matrix.columns)-1] 
    Y_TEST = test_feature_matrix.iloc[:, -1]
    return X_TRAIN, Y_TRAIN, X_TEST, Y_TEST


if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest = load_data()
    ACCURACY_SCORE = logistic_regression(xtrain, ytrain, xtest, ytest)
    print("ACCURACY score is : ", ACCURACY_SCORE)
