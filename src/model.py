from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_model(input_shape=(28, 28, 1), num_classes=10):
    """Construire le modèle CNN."""
    model = Sequential([
        Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model
