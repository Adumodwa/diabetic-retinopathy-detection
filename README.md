# Diabetic Retinopathy Detection with CNNs
A deep learning project using Convolutional Neural Networks (CNNs) to detect diabetic retinopathy from retinal images. This project aims to improve early diagnosis accuracy in medical diagnostics by automating the identification of diabetic retinopathy, a severe complication of diabetes.

## Overview
This repository contains code and resources to train a CNN model that can classify retinal images as either diabetic retinopathy positive or negative. The project leverages deep learning techniques to achieve a high level of accuracy and provides interpretability methods to make the model's decisions transparent.

## Key Features
Data Augmentation: Techniques like rotation, zooming, and flipping improve the model’s generalization.
Model Evaluation: Metrics including accuracy, precision, recall, and AUC assess model performance.
Interpretability: Tools like SHAP and Grad-CAM offer insight into model decision-making.

## Data
The dataset consists of labeled retinal images, sourced from Kaggle. It includes:

Training set: For model learning.
Validation set: For tuning hyperparameters.
Test set: For final model evaluation.
Preprocessing steps include resizing, normalization, and data augmentation.

## Model Architecture
The model is built using a Convolutional Neural Network with three primary layers:

Convolutional Layers: Extract features from the images.
Pooling Layers: Reduce spatial dimensions to improve efficiency.
Fully Connected Layer: Classifies images as DR (Diabetic Retinopathy) or No_DR.
Training and Evaluation
The model was trained using the Adam optimizer and binary cross-entropy loss, with callbacks for early stopping and learning rate reduction.

## Metrics
Accuracy: Measures overall correct predictions.
Precision & Recall: Provide insights into model sensitivity.
AUC: Indicates the model's ability to differentiate between classes.

## Interpretability
To make the model's predictions understandable:
SHAP: Shows feature contributions to each prediction.
Grad-CAM: Highlights image regions most influential in the decision.

## Results
The CNN model achieved high accuracy and recall, making it suitable for detecting diabetic retinopathy from retinal images. Interpretability methods confirm that the model focuses on medically relevant areas in images.

## Contributors
Mncwili Adumodwa - Primary Researcher
Mr. S. Ngwenya - Supervisor

## License
This project is licensed under the MIT License.


