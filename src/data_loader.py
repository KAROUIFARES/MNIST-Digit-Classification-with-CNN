from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    """Charger et prétraiter les données MNIST."""
    # Charger les données MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Redimensionner pour CNN (ajouter canal) et normaliser
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # One-hot encoding des labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    return (X_train, y_train, y_train_cat), (X_test, y_test, y_test_cat)
