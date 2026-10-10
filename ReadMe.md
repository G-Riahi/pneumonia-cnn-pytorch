

# Pneumonia Classification with PyTorch and Grad-CAM

## Overview

This project was developed as a simple refresher on **PyTorch** and **Convolutional Neural Networks (CNNs)**, while also providing an introduction to **Explainable AI (XAI)**.

The project consists of building a simple CNN to classify chest X-ray images as either **Normal** or **Pneumonia** using the **PneumoniaMNIST** dataset.

Several model parameters and architectural choices are explored in order to observe their influence on training and validation performance.

The project then introduces Explainable AI using **Grad-CAM**, which is used to visualize the image regions that contribute the most to the model's predictions.

The main objectives of the project are to:

- Review the basic PyTorch workflow for building, training, validating, and testing a CNN.
- Explore the effect of selected model parameters and regularization techniques.
- Evaluate the trained model using common classification metrics.
- Introduce model interpretability through Grad-CAM visualizations.

> **Note:** This project is intended for educational purposes only and is not designed for clinical use.


## Creating a Virtual Environment and Installing Dependencies

The project was developed using a Python virtual environment on Windows with PowerShell.

### 1. Create the virtual environment

```powershell
py -m venv .venv
```

### 2. Allow PowerShell script execution for the current session

If PowerShell prevents the activation script from running, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install the general dependencies

The following libraries are used for loading the PneumoniaMNIST dataset, data analysis, model evaluation, plotting, and Jupyter notebooks:

```powershell
pip install medmnist matplotlib scikit-learn jupyter ipykernel
```

### 5. Install PyTorch and torchvision

For systems with a compatible NVIDIA GPU and driver, PyTorch can be installed with CUDA support using:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

For systems without a compatible NVIDIA GPU, the standard version can be installed instead:

```powershell
pip install torch torchvision
```

### 6. Install Grad-CAM

The following package is used to generate Grad-CAM visualizations:

```powershell
pip install grad-cam
```


## Project Structure

```text
PneumoniaCNN/
│
├── Notebooks/
│   └── PneumoniaCNN_Experiments.ipynb
│
├── Plots/
│   ├── Grad-CAM/
│   └── Training/
│
├── Data_Loader.py
├── Grad_CAM.py
├── Model_Trainer.py
├── PneumoniaCNN.py
├── main.py
├── README.md
└── .gitignore
```

### Main files

- `PneumoniaCNN.py` contains the CNN architecture.
- `Data_Loader.py` handles dataset loading, normalization, DataLoaders, and class distribution.
- `Model_Trainer.py` handles training, validation, best-model selection, and testing.
- `Grad_CAM.py` contains the Grad-CAM visualization implementation.
- `main.py` runs the complete training, evaluation, and Grad-CAM pipeline.
- `Notebooks/PneumoniaCNN_Experiments.ipynb` contains the exploratory experiments used during model development.
- `Plots/` contains selected training/validation curves and Grad-CAM visualizations.


## Dataset

The project uses the **PneumoniaMNIST** dataset from the MedMNIST collection.

The dataset contains grayscale chest X-ray images with two classes:

- `0` — Normal
- `1` — Pneumonia

The images are resized to `128 × 128` pixels before being passed to the CNN.


## Model

The model is a small convolutional neural network designed for binary classification.

The architecture includes:

- Convolutional layers
- ReLU activation functions
- Max-pooling layers
- Adaptive average pooling
- Dropout regularization
- Fully connected layers

The final layer produces a single logit, which is converted into a probability using the sigmoid function.


## Training and Evaluation

The model is trained using:

- Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`)
- Class weighting to account for class imbalance
- Stochastic Gradient Descent (`SGD`)
- Training and validation monitoring
- Best-model selection based on validation loss

The final model is evaluated on the test dataset using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix


## Model Experiments

Several small experiments were performed during development, including:

- Comparing different adaptive pooling sizes.
- Studying the effect of the number of training epochs.
- Comparing different dropout rates.
- Monitoring training and validation loss to identify overfitting.

These experiments are available in:

```text
Notebooks/PneumoniaCNN_experiments.ipynb
```

The selected final model uses an adaptive pooling size of `3 × 3` and a dropout rate of `0.25`.


## Explainable AI with Grad-CAM

Grad-CAM is used to visualize which regions of the chest X-ray contribute most strongly to the model's prediction.

For each selected test image, the project displays:

- The original chest X-ray.
- The Grad-CAM activation map.
- The Grad-CAM heatmap overlaid on the original image.

The Grad-CAM analysis also highlights an important limitation of neural networks: the model does not always focus exclusively on clinically relevant lung regions. In some examples, strong activations may appear around the thorax or outside the lung fields.

Therefore, good classification performance does not necessarily imply that the model has learned clinically meaningful features.


## Running the Project

After activating the virtual environment and installing the required libraries, run:

```powershell
python main.py
```

The script will:

1. Load and normalize the PneumoniaMNIST dataset.
2. Create the training, validation, and test DataLoaders.
3. Train the CNN.
4. Select the best model based on validation loss.
5. Evaluate the model on the test dataset.
6. Generate Grad-CAM visualizations.


## Disclaimer

This project was created for educational purposes to practice PyTorch, CNNs, model evaluation, regularization, and Explainable AI.

It is **not intended for medical diagnosis or clinical use**.