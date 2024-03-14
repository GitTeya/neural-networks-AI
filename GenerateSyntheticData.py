import numpy as np

def generate_synthetic_data(samples=1000, features=10):

    np.random.seed(42)

    X = np.random.randn(samples, features)

    y = np.random.randint(0, 2, size=samples)
    
    return X, y


X_train, y_train = generate_synthetic_data(samples=1000, features=10)

print("Feature sample:", X_train[0])
print("Label sample:", y_train[0])
