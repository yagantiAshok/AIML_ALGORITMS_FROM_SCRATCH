
import os
import numpy as np
import pandas as pd
from  logger import logger


def relu_derivative(value):
    return (value>0).astype(float)


def sigmoid_derivative(output):
    return output * (1 - output)


def relu(value):
    return np.maximum(0,value)

def sigmoid(z_value):
    return 1 / (1 + np.exp(-z_value))




def ANN(path):

    data = pd.read_csv(path)
    data.drop("sample",axis=1,inplace=True)
    data["Class"] = data["Class"].replace({2: 0, 4: 1})
    x = data.drop("Class",axis = 1)
    y = data["Class"]
    x = x.to_numpy()
    y = y.to_numpy().reshape(-1,1)
    w1  = np.array([0.1,0.4,0.8,0.4,0.2,0.4,0.9,0.4,0.5]).reshape(-1,1)
    w2 = np.array([0.3,0.5,0.3,0.4,0.9,0.3,0.4,0.5,0.1]).reshape(-1,1)
    w19 = 0.3
    w20 = 0.1
    b1 = 0
    b2 = 0
    b3 = 0
    lr = 0.01
    epsilon = 1e-8
    n_samples = len(data)
    for epoch in range(1):
        z1 = np.dot(x,w1)  + b1
        A1 = relu(z1)
        z2 = np.dot(x,w2) + b2
        A2 = relu(z2)
        z3 = A1 * w19 + A2 * w20 + b3
        y_hat = sigmoid(z3)
        loss = - (y * (np.log(y_hat+epsilon)) + ((1-y) * np.log(1-y_hat+epsilon)))
        total_error = np.mean(loss)
        if epoch%25==0:
            logger.info(f"For EPOCH {epoch} total error {total_error}")
            logger.info("\n\n\n")
        dLoss = -y/(y_hat+epsilon) + (1-y)/(1-y_hat+epsilon)
        dloss_with_z3 = dLoss * sigmoid_derivative(y_hat)
        # back propagation 
        d_w19 = np.sum(dloss_with_z3 * A1)/n_samples
        d_w20 = np.sum(dloss_with_z3*A2)/n_samples
        d_b3 = np.sum(dloss_with_z3)/n_samples
        # weight updation for w2
        dloss_with_A2 = dloss_with_z3 * w20 * relu_derivative(z2)
        d_w2 = np.dot(x.T,dloss_with_A2)/n_samples
        d_b2 = np.sum(dloss_with_A2)/n_samples
        # weight updation for w1
        dloss_with_A1 = dloss_with_z3 * w19 * relu_derivative(z1)
        d_w1 = np.dot(x.T,dloss_with_A1)/n_samples
        d_b1 = np.sum(dloss_with_A1)/n_samples
        # gradients updation 
        w1 -= lr*d_w1
        w2 -= lr*d_w2
        w19 -= lr*d_w19
        w20 -= lr*d_w20
        b1 -= lr*d_b1
        b2 -= lr*d_b2
        b3 -= lr*d_b3
    return total_error

print(ANN(os.path.join("notebooks","Data_cls.csv")))
