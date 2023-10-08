from PIL import Image
from mnist_model import ReLU, softmax, feed_forward, make_predictions
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

# Load and preprocess a PNG image for recognition
image_path = 'Eight.png'  
preprocessed_image = preprocess_image(image_path)

# Use the loaded weights to make predictions
predicted_digit = make_predictions(preprocessed_image, W1_loaded, b1_loaded, W2_loaded, b2_loaded)[0]

# Display the predicted digit
display_image(image_path)
print(f"Predicted Digit: {predicted_digit}")
