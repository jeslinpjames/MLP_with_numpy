import numpy as np
def ReLU(Z):
    return np.maximum(0,Z)

def softmax(Z):
    return exp(Z)/np.sum(exp(Z))