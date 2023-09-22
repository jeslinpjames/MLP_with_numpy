from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved weights and biases
loaded_weights = np.load('model_weights.npy', allow_pickle=True).item()
W1_loaded = loaded_weights['W1']
b1_loaded = loaded_weights['b1']
W2_loaded = loaded_weights['W2']
b2_loaded = loaded_weights['b2']

# Define a function to preprocess a PNG image for recognition
def preprocess_image(image_path):
    # Open and resize the image to 28x28 pixels (MNIST image size)
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Normalize pixel values to the range [0, 1] and flatten the image
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, -1)  # Flatten to a 1D array
    
    return img_array

def display_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def ReLU(x):
    return np.maximum(0, x)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis = 0)


def feed_forward(W1, b1, W2, b2, x):
    #Hidden Layer
    Z1 = W1.dot(x.T) + b1  #Matrix Multiplication with the input layer
    a1 = ReLU(Z1)  #ReLU Activation

    #Output Layer
    Z2 = W2.dot(a1) + b2  #Matrix Multiplication with the hidden layer
    a2 = softmax(Z2)  #Softmax Activation
    return Z1, a1, Z2, a2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = feed_forward(W1, b1, W2, b2, X)
    pred = np.argmax(A2, 0)
    return pred
# Load and preprocess a PNG image for recognition
image_path = 'nine.png'  # Replace with the path to your PNG image
preprocessed_image = preprocess_image(image_path)

# Use the loaded weights to make predictions
predicted_digit = make_predictions(preprocessed_image, W1_loaded, b1_loaded, W2_loaded, b2_loaded)[0]

# Display the predicted digit
display_image(image_path)
print(f"Predicted Digit: {predicted_digit}")
