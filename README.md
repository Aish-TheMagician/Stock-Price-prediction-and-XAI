# Stock Price Prediction and Explainable AI

This project aims to predict stock prices using machine learning techniques and provide explanations for the model's predictions using Explainable AI (XAI) methods like LIME and SHAP. The project uses historical stock data to predict the closing prices and analyzes the contribution of features to the model's predictions.

## Table of Contents

1. [Overview](#overview)
2. [Data Collection](#data-collection)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training and Evaluation](#training-and-evaluation)
6. [Explainable AI (XAI)](#explainable-ai-xai)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Results](#results)
10. [License](#license)

## Overview

This project focuses on predicting stock prices using an LSTM (Long Short-Term Memory) model and explaining the model’s predictions using LIME and SHAP. Stock data is fetched from an online CSV, which includes historical stock prices and other related features. The model predicts the closing price, and the explanation methods are used to interpret the model's decision-making process.

## Data Collection

The data is sourced from a Google Sheets link, which provides historical stock prices. This data includes:

- Date of the stock price
- Closing price of the stock

The data is then cleaned, reshaped, and prepared for model training.

## Data Preprocessing

The preprocessing steps include:
- Extracting the relevant columns from the dataset (`date` and `close` prices).
- Converting the `date` column into a datetime object and retaining only the date part.
- Creating lag features (`x1`, `x2`) to predict the stock price using past data.
- Splitting the data into training, validation, and test datasets.

## Model Architecture

The stock price prediction model is built using the following architecture:

- **LSTM Layer:** To capture the temporal dependencies in the stock price data.
- **Dense Layers:** To add complexity and enable the model to learn patterns from the features.
- **Output Layer:** To predict the stock closing price.

## Training and Evaluation
The model is trained on the training dataset for 300 epochs.
The performance is evaluated using the Mean Absolute Error (MAE) on the test dataset.
The loss and MAE are logged and printed to evaluate the model's accuracy.

## Explainable AI (XAI)
Explainable AI methods are used to interpret the predictions made by the LSTM model, helping to understand which features contributed the most to each prediction. This project utilizes two key XAI techniques: LIME (Local Interpretable Model-Agnostic Explanations) and SHAP (SHapley Additive exPlanations).

### LIME
LIME is an interpretable machine learning method that explains individual predictions of a model by approximating it with an interpretable model in the local neighborhood of the prediction. In this project, LIME is used to explain the predictions of the stock price model.

LIME works by creating a surrogate model that mimics the behavior of the complex model (LSTM) for specific predictions. It uses the LimeTabularExplainer class, which takes the training data, features, and predicted values to generate explanations for specific instances.

The generated explanations list the features (like lag values of stock prices) and their contributions to the prediction. This helps to understand how each feature affects the prediction, which is crucial for model interpretability.

### SHAP
SHAP values provide a unified measure of feature importance by assigning each feature an importance value for each prediction. SHAP values are based on cooperative game theory and calculate the average contribution of each feature across all possible combinations of features.

In this project, SHAP is used to calculate the Shapley values for each feature in the stock price prediction task. The KernelExplainer is used to explain the model's predictions by approximating the complex model (LSTM) with a simpler model.
