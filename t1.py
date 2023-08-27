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
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.random.randn(10, 1) * 0.01
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.random.randn(10, 1) * 0.01
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / np.sum(e_Z, axis=0, keepdims=True)
    return A

def feed_forward(W1, b1, W2, b2, x):
    #Hidden Layer
    Z1 = W1.dot(x.T) + b1  #Matrix Multiplication with the input layer
    a1 = ReLU(Z1)  #ReLU Activation

    #Output Layer
    Z2 = W2.dot(a1) + b2  #Matrix Multiplication with the hidden layer
    a2 = softmax(Z2)  #Softmax Activation

    return Z1, a1, Z2, a2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max()+1,Y.size))
    one_hot_Y[Y,np.arange(Y.size)]=1
    # one_hot_Y=one_hot_Y.T
    return one_hot_Y

def back_propagation(Z1, a1, Z2, a2, W1, W2, x, y):
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

def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1,(10, 1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10, 1))
    return W1, b1, W2, b2

def loss_and_accuracy(y_pred, y_true):
    #Cross-Entropy Loss
    epsilon = 1e-10  #Constant
    m = y_true.shape[0]
    pred_loss = np.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = -np.sum(y_true * np.log(pred_loss)) / m

    #Accuracy
    accuracy = np.mean(y_pred == y_true)
    return loss, accuracy

def neural_network(x, y, epochs, alpha):
    W1, b1, W2, b2 = init_params()
    #epoch = no. of iterations
    for epoch in range(epochs):
        Z1, a1, Z2, a2 = feed_forward(W1, b1, W2, b2, x)
        dW1, dB1, dW2, dB2 = back_propagation(Z1, a1, Z2, a2, W1, W2, x, y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, dB1, dW2, dB2, alpha)
        if epoch % 100 == 0:
            print('Epoch: ', epoch)
            pred = np.argmax(a2, 0)
            loss, accuracy = loss_and_accuracy(pred, y)
            print(f'Loss: {loss:.4f}\t Accuracy: {accuracy*100:.2f} %')
    return W1, b1, W2, b2
print(y_train.shape)
W1, b1, W2, b2 = neural_network(x_train, y_train, 2001, 0.1)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = feed_forward(W1, b1, W2, b2, X)
    pred = np.argmax(A2, 0)
    return pred