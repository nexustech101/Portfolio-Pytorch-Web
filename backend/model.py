import os
import tensorflow as tf
from keras import layers, models
from keras.models import Sequential
import numpy as np

x = np.array([
    [1,0,1], [0,0,0], [0,0,0], [0,0,1], [0,1,0], [0,1,1],
    [1,0,1], [0,0,0], [1,1,1], [1,1,0], [1,0,0], [0,1,1],
    [1,1,1], [1,1,1], [0,0,0], [0,0,1], [0,1,0], [0,1,1],
    [1,0,1], [0,0,0], [1,1,1], [1,1,0], [1,0,0], [0,1,1],
    [1,1,1], [0,0,0], [1,1,1], [1,1,1], [1,0,1], [0,0,0],
], dtype=float)

y = np.array([
    [1], [0], [0], [1], [1], [1],
    [1], [0], [0], [1], [1], [1],
    [0], [0], [0], [1], [1], [1],
    [1], [0], [0], [1], [1], [1],
    [0], [0], [0], [0], [1], [0],
], dtype=float)

def train_XOR_model():
    model = Sequential()
    model.add(layers.Dense(3, input_dim=3, use_bias=True))
    model.add(layers.Dense(3, activation='relu', use_bias=True))
    model.add(layers.Dense(3, activation='relu', use_bias=True))
    model.add(layers.Dense(1, activation='sigmoid', use_bias=True))
    
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.fit(x, y, epochs=400, validation_data=(x, y))
    model.save(model_path)
    
    
model_path = 'XOR.keras'
model = None

def load_XOR_model():
    global model
    model = models.load_model(model_path)

def predict(data):
    if model is None:
        raise Exception("Model is not loaded.")
    
    prediction = model.predict(data)
    return prediction
    
    
# if __name__ == '__main__':
#     if not model_path in os.listdir():
#         train_XOR_model()
        
#     data = np.array([
#         [0,0,0], [1,0,1],
#         [0,0,1], [1,1,1],
#     ], dtype=float)
    
#     print(predict(data))