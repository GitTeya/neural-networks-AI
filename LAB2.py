import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_model():
    inputs = Input(shape=(10,))
    x = Dense(10, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(8, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(8, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(4, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def generate_dataset(num_samples=1000, num_features=10):
    np.random.seed(30)
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(2, size=num_samples)
    return X, y

def generate_test_dataset(num_samples=200, num_features=10):
    np.random.seed(31) 
    X_test = np.random.rand(num_samples, num_features)
    y_test = np.random.randint(2, size=num_samples)
    return X_test, y_test

if __name__ == "__main__":
    model = create_model()
    
    X_train, y_train = generate_dataset()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

    model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping, lr_reducer, model_checkpoint])
    
    model.summary()
 
    X_test, y_test = generate_test_dataset()
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")
