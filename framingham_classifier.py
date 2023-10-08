from acti import ReLU, softmax
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


#Reading the data
data = pd.read_csv('framingham.csv')


#Visualizing the data
fig = plt.figure(figsize = (8,8))
ax = fig.gca()
data.hist(ax=ax)
plt.show()

X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values

#Handling missing values
si = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X = si.fit_transform(X)

np.isnan(X).sum()
np.isnan(y).sum()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

np.isnan(X_train).sum()
np.isnan(y_train).sum()


input_size = 15  # Number of features in your dataset
hidden_layer_size = 128  
output_size = 2  # For binary classification


def init_params(input_size, hidden_layer_size, output_size):
    # Hidden Layer parameters
    W1 = np.random.randn(hidden_layer_size, input_size) * np.sqrt(2 / input_size)
    b1 = np.zeros((hidden_layer_size, 1))

    # Output layer parameters
    W2 = np.random.randn(output_size, hidden_layer_size) * np.sqrt(2 / hidden_layer_size)
    b2 = np.zeros((output_size, 1))

    return W1, b1, W2, b2


def feed_forward(W1, b1, W2, b2, x):
    #Hidden Layer
    Z1 = W1.dot(x.T) + b1  #Matrix Multiplication with the input layer
    a1 = ReLU(Z1)  #ReLU Activation
    #print(Z1.shape, a1.shape)
    #print(a1)
    #Output Layer
    Z2 = W2.dot(a1) + b2  #Matrix Multiplication with the hidden layer
    a2 = softmax(Z2)  #Softmax Activation
    return Z1, a1, Z2, a2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def back_propagation(Z1, a1, Z2, a2, W1, W2, x, y):
    m = x.shape[0]
    one_hot_y = one_hot(y)  #Encoding the labels
    #Output Layer derivatives
    dZ2 = 2*(a2 - one_hot_y)  #Derivative of the softmax function
    dW2 = dZ2.dot(a1.T) / m  #Derivative of output weigths
    dB2 = np.sum(dZ2, 1) / m  #Derivative of output biases
    #Hidden Layer derivatives
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)  #Derivative of the hidden layer and ReLU function
    dW1 = dZ1.dot(x) / m  #Derivative of the hidden layer weights
    dB1 = np.sum(dZ1, 1) / m  #Derivative of the hidden layer biases
    return dW1, dB1, dW2, dB2

def update_params(W1, b1, W2, b2, dW1, dB1, dW2, dB2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(dB1,(b1.shape[0], 1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(dB2, (b2.shape[0], 1))
    return W1, b1, W2, b2

def loss_and_accuracy(y_pred, y_true):
    #Cross-Entropy Loss
    epsilon = 1e-10  #Constant
    m = y_true.shape[0]
    # pred_loss = np.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = - (1 / m) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

    #Accuracy
    accuracy = np.mean(y_pred == y_true)
    return loss, accuracy

def neural_network(x, y, epochs, alpha):
    W1, b1, W2, b2 = init_params(input_size, hidden_layer_size, output_size)
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


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = feed_forward(W1, b1, W2, b2, X)
    pred = np.argmax(A2, 0)
    return pred

def calculate_test_accuracy(X_test, y_test, W1, b1, W2, b2):
    # Make predictions using the trained model
    predictions = make_predictions(X_test, W1, b1, W2, b2)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    
    return accuracy

if __name__ == '__main__':
    W1, b1, W2, b2 = neural_network(X_train, y_train, 5001, 0.25)
    test_accuracy = calculate_test_accuracy(X_test, y_test, W1, b1, W2, b2)
    print(f"Test Accuracy: {test_accuracy * 100:.2f} %")


    # Save the trained weights and biases to a file
    np.save('framingham_model_weights.npy', {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2})
