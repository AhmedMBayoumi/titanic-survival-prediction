# Titanic Survival Prediction

This project utilizes the Titanic dataset to gather insights, apply feature engineering, split and scale the data, and compare various machine learning models to predict passenger survival. The project includes data visualization, feature engineering, model training, and evaluation.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Visualization](#data-visualization)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
The goal of this project is to apply various machine-learning techniques to predict the survival of passengers on the Titanic. The project involves:
1. Data visualization to understand the dataset.
2. Feature engineering to create new features and prepare the data.
3. Splitting and scaling the data.
4. Training and evaluating multiple machine learning models.

## Project Structure
The project is structured as follows:
- `data/`: Contains the Titanic dataset files (`train.csv` and `test.csv`).
- `notebooks/`: Jupyter notebooks for data exploration and model training.
- `README.md`: Project documentation.

## Data Visualization
Data visualization helps in understanding the distribution and relationships within the dataset. The visualizations include:
1. Pie chart of Survived vs. Dead.
2. Donut chart of Male vs. Female.
3. Count plot of Survived and Dead by Gender.
4. Histogram of Age distribution.
5. Count plot of Passenger Class distribution.
6. Bar plot of Survival rate by Family Size.

## Feature Engineering
Feature engineering steps include:
1. Creating a new feature 'Family Size' by combining 'SibSp' and 'Parch'.
2. Dropping irrelevant columns such as 'PassengerId', 'Name', and 'Ticket'.
3. Encoding categorical features like 'Sex' and 'Embarked'.
4. Handling missing values in 'Cabin' and 'Embarked'.
5. Creating and encoding an 'AgeGroup' feature.

## Model Training and Evaluation
Several machine learning models are trained and evaluated, including:
1. Decision Tree
2. Random Forest
3. Extra Trees
4. Support Vector Machine (SVM)
5. Naive Bayes
6. K-Nearest Neighbors (KNN)
7. Logistic Regression
8. Linear Regression
9. Gradient Boosting
10. AdaBoost
11. Lasso and Ridge Regression
12. ElasticNet
13. Simple Neural Network (Perceptron and MLP)
14. Clustering (KMeans, Hierarchical, Gaussian Mixture)

Each model is trained using the training set and evaluated on the test set. Evaluation metrics include accuracy for classification models and mean squared error for regression models.

## Results
The models' performance is compared to determine the best model for predicting Titanic survival. Key results include:
- Decision Tree Accuracy:  0.95%
- Random Forest Accuracy: 0.95%
- SVM Accuracy: 0.80%
- Gradient Boosting Accuracy: 0.86%
- AdaBoost: 0.80%

## Usage
To train and evaluate the models, run the Jupyter notebooks in the `notebooks/` directory. Ensure you have the Titanic dataset files (`train.csv` and `test.csv`) in the `data/` directory.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.
