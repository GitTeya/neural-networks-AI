import numpy as np
from PIL import Image
import os

# Part 1: Preprocessing
def load_images(folder, size=(20, 20), is_flatten=True):
    images = []
    for filename in os.listdir(folder):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(size)
                if is_flatten:
                    images.append(np.array(img).flatten())
                else:
                    images.append(np.array(img))
            except IOError as e:
                print(f"Could not read image: {img_path}. Error: {e}")
    return np.array(images)


# Test images are in 'images' folder

# Load and label train images
train_images = load_images(r'D:\images') 
train_labels = np.repeat(np.arange(10), 10)  # Adjust based on your dataset

# Load test images
test_images = load_images(r'D:\test') 

# Part 2: Perceptron Implementation
class Perceptron:
    def __init__(self, input_size, lr=0.01):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = np.random.rand()
        self.lr = lr

    def activation(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # Forward pass
        return self.activation(np.dot(x, self.weights) + self.bias)

    def train(self, x, y, epochs):
        # Training function
        for epoch in range(epochs):
            for i in range(len(x)):
                y_pred = self.forward(x[i])
                error = y[i] - y_pred
                # Update weights and bias
                self.weights += self.lr * error * x[i]
                self.bias += self.lr * error

# Part 3: Training the Perceptron
perceptron = Perceptron(input_size=400)  # 20x20 pixels flattened
perceptron.train(train_images, train_labels, epochs=1000)

# Part 4: Testing the Perceptron
def test_perceptron(model, test_images):
    predictions = []
    for img in test_images:
        pred = model.forward(img)
        predictions.append(pred)
    return predictions

predictions = test_perceptron(perceptron, test_images)

print(predictions)
