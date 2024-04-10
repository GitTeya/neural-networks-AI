import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from kerastuner import RandomSearch

def create_model(hp):
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    l2_rate = hp.Float('l2_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    
    inputs = Input(shape=(10,))
    x = Dense(10, activation='relu', kernel_regularizer=l2(l2_rate))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    for i in range(hp.Int('num_layers', 2, 4)):  
        x = Dense(hp.Int(f'neurons_{i}', min_value=8, max_value=64, step=8),
                  activation='relu',
                  kernel_regularizer=l2(l2_rate))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def generate_dataset(num_samples=1000, num_features=10):
    np.random.seed(30)
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(2, size=num_samples)
    return X, y

early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')


X, y = generate_dataset()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

tuner = RandomSearch(
    create_model,
    objective='val_accuracy',
    max_trials=10,  
    executions_per_trial=1, 
    directory='my_dir',
    project_name='lab2_tuning'
)

tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping, lr_reducer, model_checkpoint])

best_model = tuner.get_best_models(num_models=1)[0]

test_loss, test_acc = best_model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {test_acc}, Validation Loss: {test_loss}")
