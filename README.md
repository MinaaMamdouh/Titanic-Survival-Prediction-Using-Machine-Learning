# Titanic Survival Prediction Using Machine Learning

## Project Overview

This project focuses on predicting the survival of passengers aboard the Titanic using machine learning techniques. The Titanic dataset contains information about the passengers, such as their age, sex, class, and other personal features, which can be used to predict whether they survived or not. 

The primary goal of the project is to implement various machine learning algorithms, evaluate their performance, and compare their results to determine the best approach for predicting Titanic passenger survival.

## Dataset

The dataset used for this project is the famous Titanic dataset from Kaggle. It contains historical data about the passengers who were on board the ill-fated Titanic voyage. The dataset includes both numerical and categorical variables, such as:

- **Survived**: Whether the passenger survived (0 = No, 1 = Yes)
- **Pclass**: Passenger class (1st, 2nd, or 3rd)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger (male, female)
- **Age**: Age of the passenger
- **SibSp**: Number of siblings or spouses aboard the Titanic
- **Parch**: Number of parents or children aboard the Titanic
- **Ticket**: Ticket number
- **Fare**: The fare the passenger paid
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

You can find the dataset [here](https://github.com/MinaaMamdouh/Titanic-Survival-Prediction-Using-Machine-Learning/blob/main/Titanic-Dataset%20(2).csv).

## Objective

The objective of this project is to:

- Preprocess and clean the Titanic dataset
- Apply different machine learning models to predict survival
- Use techniques like imputation, feature engineering, and balancing the dataset
- Evaluate and compare the performance of different models
- Perform hyperparameter tuning for improved model performance

## Steps Involved

1. **Data Preprocessing**:
   - Handling missing values with imputation strategies
   - Encoding categorical variables (e.g., Sex, Embarked)
   - Extracting useful features (e.g., extracting titles from passenger names, creating family size feature)
   - Handling outliers in numerical features

2. **Feature Engineering**:
   - Creating new features such as "Title" (e.g., Mr., Mrs., etc.)
   - Engineering the "FamilySize" feature
   - Handling outliers in the "Fare" and "Cabin" features

3. **Model Building**:
   - Applying machine learning models including Random Forest, XGBoost, and Logistic Regression
   - Using SMOTE to balance the class distribution (as the dataset is imbalanced)

4. **Model Evaluation**:
   - Evaluating models using accuracy, precision, recall, and F1-score metrics
   - Visualizing the performance of each model
   - Comparing results from different classifiers (Random Forest, XGBoost, Voting Classifier)

5. **Hyperparameter Tuning**:
   - Using GridSearchCV to fine-tune the hyperparameters of Random Forest and XGBoost models for better performance.

6. **Ensemble Method (Voting Classifier)**:
   - Combining multiple models (Logistic Regression, Random Forest, and XGBoost) using a Voting Classifier to improve prediction performance.

## Models Used

- **Random Forest Classifier**: A versatile ensemble method based on decision trees.
- **XGBoost**: A gradient boosting method known for its performance in classification tasks.
- **Logistic Regression**: A baseline model for binary classification tasks.
- **Voting Classifier**: An ensemble method that combines the predictions of multiple models to improve accuracy.

## Performance Evaluation

The models were evaluated on the following metrics:
- **Accuracy**: The overall correctness of the model.
- **Precision**: The proportion of true positives among all positive predictions.
- **Recall**: The proportion of true positives among all actual positives.
- **F1 Score**: The harmonic mean of precision and recall, which balances both metrics.

## Results

The comparison of different models' performance is visualized, and the final accuracy scores are as follows:

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 78.9%    | 0.681     | 0.889  | 0.771    |
| Random Forest         | 80.0%    | 0.725     | 0.806  | 0.763    |
| XGBoost               | 82.2%    | 0.727     | 0.889  | 0.800    |
| Voting Classifier     | 83.3%    | 0.733     | 0.917  | 0.815    |

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git
   cd titanic-survival-prediction

