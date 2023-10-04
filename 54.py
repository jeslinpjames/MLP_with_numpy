import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist
 
data = mnist.load_data()
(x_train, y_train), (x_test, y_test) = data
x_train = x_train.reshape(x_train.shape[0], -1) /255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
# print(x_train[0].shape, x_test[0].shape)

def init_params():
    W1 = np.random.randn(10, 784) * np.sqrt(2/784)
    b1 = np.random.randn(10, 1) 
    W2 = np.random.randn(10, 10) * np.sqrt(2/10)
    b2 = np.random.randn(10, 1) 
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z))
    A = e_Z / e_Z.sum( axis=0)
    return A


def forward_prop(W1,b1,W2,b2,X):
    Z1=W1.dot(X.T)+b1
    A1=ReLU(Z1)
    Z2=W2.dot(A1 )+b2
    A2=softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max()+1,Y.size))
    one_hot_Y[Y,np.arange(Y.size)]=1
    # one_hot_Y=one_hot_Y.T
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


def back_prop(Z1, A1, Z2, A2, W2, Y, X):
    one_hot_Y = one_hot(Y)
    #m = Y.size
    m = X.shape[0]
    dZ2 = 2*(A2 - one_hot_Y)
    dW2 = dZ2.dot(A1.T) / m
    db2 = np.sum(dZ2, axis=1) / m

    dZ1 = W2.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = dZ1.dot(X) / m
    db1 = np.sum(dZ1, axis=1) / m
    return dW1, db1, dW2, db2


def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha):
    W1= W1-alpha*dW1
    b1= b1- alpha*np.reshape(db1,(10,1))
    W2 = W2 - alpha*dW2
    b2= b2- alpha*np.reshape(db2,(10,1))
    return W1,b1,W2,b2

def get_predictions(A2):
    return np.argmax(A2,axis=1)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions==Y)/Y.size

def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = -1/m * np.sum(Y * np.log(A2 + 1e-8))
    return cost

def loss_and_accuracy(y_pred, y_true):
    #Cross-Entropy Loss
    epsilon = 1e-10  #Constant
    m = y_true.shape[0]
    pred_loss = np.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = -np.sum(y_true * np.log(pred_loss)) / m

    #Accuracy
    accuracy = np.mean(y_pred == y_true)
    return loss, accuracy

def gradient_descent(X,Y,iterations,alpha):
    W1,b1,W2,b2=init_params()
    for i in range (iterations):
        Z1,A1,Z2,A2=forward_prop(W1,b1,W2,b2,X)
        dW1,db1,dW2,db2=back_prop(Z1,A1,Z2,A2,W2,Y,X)
        W1,b1,W2,b2 = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)
        if i%30 ==0:
            #cost = compute_cost(A2, one_hot(Y))
            pred = np.argmax(A2, 0)
            cost, accuracy = loss_and_accuracy(pred, Y)
            print("Iteration : ",i)
            #print("Accuracy : ",get_accuracy(get_predictions(A2),Y))
            print("Accuracy : ",accuracy)
            print("Cost:", cost)
    return W1,b1,W2,b2


W1,b1,W2,b2= gradient_descent(x_train,y_train,2001,0.1)

"""
# Assuming you have initialized the required variables
Z1 = np.array([[-3.03058198],
               [ 0.4370953 ],
               [ 0.76729269],
               [ 1.21276302],
               [ 0.43938869],
               [ 0.9985658 ],
               [ 0.58333937],
               [-0.10489132],
               [-0.03171531],
               [ 0.66560378]])

A1 = np.array([[0.        ],
               [0.4370953 ],
               [0.76729269],
               [1.21276302],
               [0.43938869],
               [0.9985658 ],
               [0.58333937],
               [0.        ],
               [0.        ],
               [0.66560378]])

Z2 = np.array([[ 1.96321386],
               [-0.98340892],
               [-0.7805494 ],
               [-2.00948379],
               [-0.30459328],
               [-0.77193105],
               [ 1.49162348],
               [ 1.29016978],
               [ 1.76314639],
               [-0.5324132 ]])

A2 = np.array([[0.29945738],
               [0.01572653],
               [0.01926344],
               [0.00563656],
               [0.03100551],
               [0.01943017],
               [0.18686412],
               [0.15276916],
               [0.24515843],
               [0.02468869]])

W2 = np.array([[ 0.04, -0.24,  0.15, -0.19, -0.16, -0.22,  0.01, -0.14, -0.18,  0.08],
               [ 0.12,  0.07, -0.12, -0.08,  0.18, -0.11,  0.08, -0.01,  0.15, -0.15],
               [ 0.16,  0.07,  0.12, -0.07, -0.03,  0.08,  0.05, -0.11, -0.11, -0.19],
               [-0.19,  0.18,  0.16,  0.01,  0.13,  0.1 ,  0.11, -0.15,  0.16,  0.19]])
Y = np.array([2, 0, 1, 0, 2, 1, 2, 0, 1, 2])  # Updated to (10, 1)

X = np.array([[0.45973695, 0.19223454, 0.44736714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # Updated to (10, 1)

# Calling the back_prop function
dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, Y, X)

# Printing the results
print("dW1:")
print(dW1)
print("db1:")
print(db1)
print("dW2:")
print(dW2)
print("db2:")
print(db2)
"""