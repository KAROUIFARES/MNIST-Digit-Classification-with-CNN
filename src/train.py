def train_model(model, X_train, y_train_cat, epochs=5, batch_size=64, validation_split=0.1):
    """Compiler et entraîner le modèle."""
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train_cat, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=validation_split
    )
    return history
