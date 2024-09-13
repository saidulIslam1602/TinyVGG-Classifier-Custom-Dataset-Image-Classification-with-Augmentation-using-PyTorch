# TinyVGG-Classifier-Custom-Dataset-Image-Classification-with-Augmentation-using-PyTorch

Project Overview
This project implements an image classification pipeline using a custom dataset of pizza, steak, and sushi images. It leverages the TinyVGG model architecture and is built using PyTorch. The project showcases how to handle custom datasets, apply various image augmentations, and visualize model performance across training and validation phases.

Key features include:

Custom Dataset Handling: The project demonstrates how to load and process custom image datasets.
Data Augmentation: Data augmentation techniques such as resizing, random flipping, and trivial augmentations are applied to improve model generalization.
TinyVGG Model Architecture: A convolutional neural network based on TinyVGG is used to perform multi-class classification.
Training and Evaluation: The model is trained on the custom dataset, and performance metrics (accuracy and loss) are visualized after each epoch.
Inference on Custom Images: The project includes functionality to make predictions on custom images and visualize results, with predicted class probabilities.
Project Structure
Dataset: Custom image dataset (pizza, steak, sushi) with train and test splits.
Model: TinyVGG architecture built using PyTorch for image classification.
Training: Training pipeline with data augmentation, cross-entropy loss, and Adam optimizer.
Evaluation: Metrics tracking and visualization of loss and accuracy for training and test sets.
Inference: Prediction on new, custom images with detailed visual output of classification probabilities.
