# Classification using ResNet and Vision Transformer
## Project Overview
This project utilizes a hybrid deep learning model combining ResNet-50 and a Vision Transformer to classify eye diseases from images. The model architecture leverages the feature extraction capabilities of ResNet-50 and the relational reasoning of transformers to enhance the classification accuracy.


## Structure
01. main.py: Contains the entire code including model definition, training, validation, and test evaluation.
02. Train Dataset Path: Directory containing training data categorized by class folders.
03. Validation Dataset Path: Directory containing validation data categorized by class folders.
04. Test Dataset Path: Directory containing test data categorized by class folders.

## Dependencies
01. torch
02. torchvision
03. matplotlib
04. sklearn

## Model Details
→ Feature extraction: Modified ResNet-50 to output feature maps.
→ Dimension projection: A linear layer to reduce feature dimensions suitable for the transformer.
→ Transformer encoder: Processes the projected features to capture global dependencies.
→ Classification layer: A fully connected layer for binary classification.

## Usage
→ Adjust the dataset paths in the script to point to your train, validation, and test directories.
→ The training process will output logs for loss and accuracy and save plots for training/validation loss and accuracy.

## Model Training and Evaluation
The model is trained using SGD with specified hyperparameters. The training function includes both training and validation phases, and outputs performance metrics after each epoch. After training, the model is evaluated on a separate test dataset to report accuracy, precision, recall, and F1 score.

## Additional Notes
→ The test evaluation section loads the test dataset, runs model inference, and calculates various performance metrics.
→ Ensure CUDA-compatible hardware is available for GPU acceleration. Adjust the device variable as needed to switch between CPU and GPU.


## Installation
To run the code, you need to install few basic dependencies.
>most of the commands can be installed using pip.
>To install Pip follow this [link](https://pip.pypa.io/en/stable/installation/).

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
> For statistical modelling we have used sklearn.
```
pip install -U scikit-learn
```
> with all these dependencies you can run the following code. The model has been trained for about 50 epochs. 
















 


