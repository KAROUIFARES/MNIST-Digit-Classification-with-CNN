from src.data_loader import load_and_preprocess_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("Step 1: Loading and preprocessing data...")
    (X_train, y_train, y_train_cat), (X_test, y_test, y_test_cat) = load_and_preprocess_data()

    print("Step 2: Building model...")
    model = build_model()
    model.summary()

    print("Step 3: Training model...")
    history = train_model(model, X_train, y_train_cat)

    print("Step 4: Evaluating model...")
    evaluate_model(model, X_test, y_test, y_test_cat, history)

    print("\nProject execution completed successfully.")

if __name__ == "__main__":
    main()
