# END TO END DATA SCIENCE PROJECT

## Student Performance Prediction Pipeline

This project implements a complete machine learning pipeline to predict student math scores based on demographic and test preparation data. It includes data ingestion, data transformation, model training, evaluation, and prediction.

## 📌 Project Description
This project predicts student performance using various regression models based on input features like gender, parental education, lunch status, test preparation, and more.

## 🚀 Features
- Multiple regression model comparisons  
- Hyperparameter tuning using GridSearchCV  
- Evaluation using R² score  
- Data preprocessing with imputation, scaling, and encoding  
- Prediction pipeline with saved model and preprocessor  
- (Optional) Flask app for interactive prediction

## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, seaborn  
- CatBoost  
- XGBoost  
- Dill  
- Flask  

**For full dependencies, see** `requirements.txt`

---

## About the Dataset

The dataset contains student demographic information and their scores in three subjects. The goal is to predict the math score (target variable) based on other features.

Features include:  
- `gender` (Male/Female)  
- `race_ethnicity` (Group A, B, C, D, E)  
- `parental_level_of_education` (e.g., bachelor's degree, some college, master's degree)  
- `lunch` (standard or free/reduced)  
- `test_preparation_course` (completed or not completed)  
- `reading_score`  
- `writing_score`  

**Target variable:**  
- `math_score`

---

## Project Structure

- **Data Ingestion**  
  Reads the raw dataset (`notebook/data.csv`), splits it into training and test sets, and saves them as CSV files in the `artifacts/` directory.

- **Data Transformation**  
  Applies preprocessing including imputing missing values, scaling numerical features, and one-hot encoding categorical features using sklearn Pipelines.

- **Model Training and Evaluation**  
  Trains multiple regression models on the transformed data, performs hyperparameter tuning using GridSearchCV, evaluates models using R² score, and selects the best model.  
  Models used:  
  - Random Forest Regressor  
  - Decision Tree Regressor  
  - Gradient Boosting Regressor  
  - Linear Regression  
  - XGBoost Regressor  
  - CatBoost Regressor  
  - AdaBoost Regressor  

- **Prediction Pipeline**  
  Loads the saved preprocessor and best model to predict math scores from new input data.

- **Flask App Creation**  
  A Flask web application to predict student scores via a user interface (if implemented).

---

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt


2. **Run Data Ingestion**  
   This reads `notebook\data.csv`, splits into train and test sets, and saves them:
   ```bash
   python data_ingestion.py

2. **Run Flask App**  
  
   ```bash
   python app.py  
   -- OR --
   python -m app


# Exploratory Data Analysis Notebook
 [Exploratory Data Analysis Notebook](https://github.com/Mohitkumart/mlproject/blob/main/notebook/1.EDA%20STUDENT%20PERFORMENCE.ipynb)

# Model Training Approach Notebook
 [Model Training Approach Notebook](https://github.com/Mohitkumart/mlproject/blob/main/notebook/2.%20MODEL%20TRAINING.ipynb)

# DATA SET LINK
 [DATA SET LINK](https://github.com/Mohitkumart/mlproject/blob/main/notebook/data.csv)


# Author

Mohit Kumar 
[Mohit Kumar GitHub Profile](https://github.com/Mohitkumart)



