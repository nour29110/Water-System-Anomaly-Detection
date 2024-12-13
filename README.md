# Anomaly Detection in Water Systems

This repository contains the implementation of various machine learning models used to detect anomalies (attacks) in water systems. The project aims to identify whether there is an attack on the water system using multiple neural networks and machine learning algorithms, comparing their performance in terms of accuracy, precision, recall, and F1-score. 

## Models Implemented
The following models were developed and evaluated in this project:

- **Convolutional Neural Networks (CNN):**
  Convolutional Neural Networks are designed to process data with grid-like topology, such as images or time-series data. In this project, the CNN struggled with detecting anomalies due to its limited ability to capture long-term dependencies in time-series data.

  ![CNN Model Image](https://github.com/nour29110/Water-System-Anomaly-Detection-/blob/main/Images/CNN.png)

- **Long Short-Term Memory Networks (LSTM):**
  LSTMs are a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies, making them ideal for time-series data. This model achieved the best performance in the project by leveraging the temporal structure of the dataset.

  ![LSTM Model Image](#)

- **Autoencoders:**
  Autoencoders are unsupervised neural networks used to learn efficient representations of input data. They are often employed in anomaly detection by comparing reconstructed inputs to the original ones. The performance was average in this project.

  ![Autoencoder Model Image](#)

- **Random Forest:**
  Random Forest is an ensemble learning method that operates by constructing multiple decision trees and outputting the majority class prediction. It showed moderate performance in detecting anomalies.

  ![Random Forest Model Image](#)

- **Support Vector Machines (SVM):**
  SVM is a supervised learning algorithm used for classification by finding the hyperplane that best separates the classes. While effective in some scenarios, SVM struggled due to the high dimensionality and class imbalance of the dataset.

  ![SVM Model Image](#)

- **Transformers:**
  Transformers, originally designed for natural language processing, use attention mechanisms to weigh input features dynamically. While innovative, their performance in this anomaly detection task was not as strong as the LSTM.

  ![Transformer Model Image](#)

## Key Features
### 1. Dataset Characteristics
- **Dataset Size:** ~3,200 samples
  - 200 attack samples (minority class)
  - 3,000 non-attack samples (majority class)
- **Features:** Includes time-series data with a datetime column, sensor readings, and system metrics.

### 2. Challenges Faced
#### Overfitting
- **Problem:** Models initially performed well on training data but poorly on test data due to overfitting.
- **Solutions:**
  - Early Stopping: Halted training when validation performance degraded.
  - Regularization: Used Dropout layers and Batch Normalization to generalize better.

#### Class Imbalance
- **Problem:** The dataset had a significant imbalance between attack and non-attack samples.
- **Solutions:**
  - **SMOTE (Synthetic Minority Over-sampling Technique):** Generated synthetic samples for the minority class.
  - **Class Weights:** Adjusted the model loss function to prioritize the minority class.

#### Datetime Column
- **Problem:** Models like Random Forest and SVM do not natively handle datetime features.
- **Solution:** Dropped the datetime column for most models except LSTM, which leverages time-series data.

#### Missing Labels in Testing Dataset
- **Problem:** The testing dataset lacked labels.
- **Solution:** Manually added labels using reference materials.

### 3. Evaluation Metrics
To evaluate model performance, the following metrics were used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### 4. Model Performance Summary
- **LSTM:** Achieved the highest accuracy and best overall performance by leveraging time-series data.
- **CNN:** Struggled the most in capturing patterns, likely due to its architectural limitations for time-series analysis.

## Results
### Performance Comparison
| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| LSTM           | 0.96  | 0.95      | 0.90   | 0.92     |
| Transformers   | 0.67     | 0.26      | 0.35   | 0.30     |
| Autoencoders   | 0.26  | 0.17   | 0.73| 0.28  |
| Random Forest  | 0.49  | 0.18   | 0.47| 0.26  |
| SVM            | 0.29      | 0.18       | 0.72    | 0.28      |
| CNN            | 0.19   | 0.19    | 1.0 | 0.33   |

## Acknowledgments
- Special thanks to the research team for providing guidance and the dataset.
- This project is part of an ongoing effort to improve the security of critical infrastructure systems.
