import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist


data = mnist.load_data()
(X_train, y_train), (X_test, y_test) = data
x_train = X_train.reshape(X_train.shape[0], -1) / 255.0
x_test = X_test.reshape(X_test.shape[0], -1) / 255.0

def init_params():
    #Hidden Layer parameters
    W1 = np.random.randn(10, 784) * np.sqrt(2/784)
    b1 = np.zeros((10, 1))

    #Output layer parameters
    W2 = np.random.randn(10, 10) * np.sqrt(2/10)
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2


def ReLU(x):
    return np.maximum(0, x)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis = 0)

def forward_prop(W1, b1, W2, b2, x):
    #Hidden Layer
    Z1 = W1.dot(x.T) + b1  #Matrix Multiplication with the input layer
    a1 = ReLU(Z1)  #ReLU Activation
    # print(Z1.shape, a1.shape)
    # print(a1)
    #Output Layer
    Z2 = W2.dot(a1) + b2  #Matrix Multiplication with the hidden layer
    a2 = softmax(Z2)  #Softmax Activation
    return Z1, a1, Z2, a2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def deriv_ReLU(Z):
    return Z>0


# def back_prop(Z1,A1,Z2,A2,W2,Y):
#     one_hot_Y = one_hot(Y)
#     m= Y.size
#     dZ2 = A2-one_hot_Y
#     dW2= 1/m*dZ2.dot(A1.T)
#     db2= 1/m*np.sum(dZ2,axis=1,keepdims=True)
#     dZ1= W2.T.dot(dZ2)*deriv_ReLU(Z1)
#     dW1=1/m* dZ1.dot(X.T)
#     db1=1/m* np.sum(dZ1,axis=1,keepdims=True)
#     return dW1,db1,dW2,db2


def back_prop(Z1, a1, Z2, a2, W1, W2, x, y):
    m = x.shape[0]
    one_hot_y = one_hot(y)  #Encoding the labels

    #Output Layer derivatives
    dZ2 = 2*(a2 - one_hot_y)  #Derivative of the softmax function
    dW2 = dZ2.dot(a1.T) / m  #Derivative of output weigths
    dB2 = np.sum(dZ2, 1) / m  #Derivative of output biases

    #Hidden Layer derivatives
    dZ1 = W2.dot(dZ2) * (Z1 > 0)  #Derivative of the hidden layer and ReLU function
    dW1 = dZ1.dot(x) / m  #Derivative of the hidden layer weights
    dB1 = np.sum(dZ1, 1) / m  #Derivative of the hidden layer biases
    return dW1, dB1, dW2, dB2


def update_params(W1, b1, W2, b2, dW1, dB1, dW2, dB2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(dB1,(10, 1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(dB2, (10, 1))
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2,axis=1)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions==Y)/Y.size

def loss_and_accuracy(y_pred, y_true):
    #Cross-Entropy Loss
    epsilon = 1e-10  #Constant
    m = y_true.shape[0]
    pred_loss = np.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = -np.sum(y_true * np.log(pred_loss)) / m

    #Accuracy
    accuracy = np.mean(y_pred == y_true)
    return loss, accuracy

def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = -1/m * np.sum(Y * np.log(A2 + 1e-8))
    return cost

def gradient_descent(X,Y,iterations,alpha):
    W1,b1,W2,b2=init_params()
    for i in range (iterations):
        Z1,A1,Z2,A2=forward_prop(W1,b1,W2,b2,X)
        dW1,db1,dW2,db2=back_prop(Z1,A1,Z2,A2,W1,W2,X,Y)
        W1,b1,W2,b2 = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)
        if i % 100 == 0:
            print('Epoch: ', i)
            pred = np.argmax(A2, 0)
            loss, accuracy = loss_and_accuracy(pred, Y)
            print(f'Loss: {loss:.4f}\t Accuracy: {accuracy*100:.2f} %')
    return W1, b1, W2, b2


W1,b1,W2,b2= gradient_descent(x_train,y_train,2001,0.5)