# Brain Tumor Classification using CNN and MobileNet

This repository contains code for brain tumor classification using Convolutional Neural Networks (CNNs) and a pre-trained MobileNet model. The project focuses on classifying brain MRI images into four categories: glioma, meningioma, pituitary, and no tumor.

## Dataset

The dataset consists of brain MRI images categorized into four classes:

-   glioma
-   meningioma
-   notumor
-   pituitary

The original dataset is located in the `Training` directory. The data is then split into training, testing, and validation sets.

## Project Overview

The project aims to:

-   Preprocess and split the brain MRI image dataset into training, testing, and validation sets.
-   Build and train a custom CNN model for brain tumor classification.
-   Utilize a pre-trained MobileNet model for transfer learning to improve classification accuracy.
-   Evaluate the performance of both models and compare their results.

## Files

-   `brain_tumor_classification.ipynb`: Jupyter Notebook containing the Python code for data preprocessing, model building, training, and evaluation.
-   `Training/`: Directory containing the original brain MRI images.
-   `train/`: Directory containing the training dataset.
-   `test/`: Directory containing the testing dataset.
-   `validation/`: Directory containing the validation dataset.
-   `bestmodel.h5`: Saved model file for the custom CNN.
-   `bestmodelmobile.h5`: Saved model file for the MobileNet model.

## Dependencies

-   numpy
-   matplotlib
-   os
-   shutil
-   glob
-   keras
-   tensorflow

## Data Preprocessing
The script performs the following data preprocessing steps:

1. **Counting images:** Counts the number of images in each class within the Training directory.
2. **Data splitting:** Splits the dataset into training (70%), testing (15%), and validation (15%) sets.
3. **Image data generation:** Uses ImageDataGenerator to preprocess images, apply data augmentation, and generate batches of image data for training and evaluation.

## Model Building and Training
The notebook includes the following model building and training steps:

1. **Custom CNN model:** Builds a sequential CNN model with convolutional, pooling, and dense layers.
2. **MobileNet model:** Utilizes a pre-trained MobileNet model, freezes its layers, and adds a custom output layer for classification.
3. **Model compilation:** Compiles both models with appropriate optimizers, loss functions, and metrics.
4. **Model training:** Trains both models using the generated training data and validates their performance using the validation data.
5. **Early stopping and model checkpoint:** Implements early stopping and model checkpointing to prevent overfitting and save the best model.

## Model Evaluation
The script evaluates the performance of the trained models using the testing dataset:

## Model loading
Loads the best saved model from the model checkpoint.

Testing with a single image: Demonstrates how to load and predict the class of a single image using the trained model.
## Results
1. **Custom CNN:** Achieved an accuracy of approximately 67.65% on the testing dataset.
2. **MobileNet:** Achieved an accuracy of approximately 83.47% on the testing dataset.
The MobileNet model outperforms the custom CNN due to transfer learning and the pre-trained model's ability to extract more robust features.

