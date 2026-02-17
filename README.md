# MNIST Digit Classification with CNN

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using TensorFlow/Keras.

## Project Structure

```text
.
├── main.py                 # Entry point of the application
├── src/
│   ├── data_loader.py      # Data loading and preprocessing logic
│   ├── model.py            # CNN architecture definition
│   ├── train.py            # Training logic
│   └── evaluate.py         # Evaluation and plotting logic
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Features

- **Modular Architecture**: Code is separated into logical modules for better maintainability.
- **CNN Implementation**: A classic CNN architecture with Convolutional, Max Pooling, and Dense layers.
- **Automated Preprocessing**: Includes normalization and one-hot encoding.
- **Visualization**: Automatically generates training curves and a confusion matrix.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to start the training and evaluation pipeline:

```bash
python main.py
```

After execution, the following files will be generated:
- `training_curves.png`: Plots for accuracy and loss over epochs.
- `confusion_matrix.png`: Heatmap of the model's performance on the test set.

## Results

The model achieves high accuracy (>98%) on the MNIST test dataset.
