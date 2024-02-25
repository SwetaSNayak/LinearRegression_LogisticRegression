import numpy as np
import argparse
import pandas as pd

def sigmoid(z):
    return 1/(1+np.exp(-z))

def logistic_regression_newton(X, y, learning_rate=0.001, n_iterations=1000, tol=1e-6):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for i in range(n_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)

        # Compute gradient
        gradient = np.dot(X.T,(y_predicted-y))/n_samples

        # Compute Hessian matrix
        W = np.diag(y_predicted * (1 - y_predicted))
        hessian = np.dot(X.T, np.dot(W, X)) / n_samples
        #hessian = np.dot(X.T, (np.dot(X, weights) + bias - y_predicted.reshape(-1,1)) * y_predicted.reshape(-1,1) * (1 - y_predicted.reshape(-1,1))) / n_samples

        # Update parameters using Newton's method
    weights-=learning_rate* np.dot(np.linalg.inv(hessian),gradient)
    bias-= learning_rate * np.sum((y_predicted-y))/n_samples

    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression with Newton's Method")
    parser.add_argument("--data", type=str, help="Path to data file (CSV format)")
    args = parser.parse_args()

    if args.data:
        data = np.genfromtxt(args.data, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]
        print(y)

        predictions = logistic_regression_newton(X, y)
        print("Predictions:", predictions)
    else:
        print("Please provide the path to the data file using the '--data' argument.")
