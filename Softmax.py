import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# Example usage
x = np.array([1.0, 2.0, 0.1])
print(softmax(x))
