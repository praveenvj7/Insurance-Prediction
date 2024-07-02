# Insurance Cost Prediction
A Python program for insurance price prediction using linear regression

This project aims to predict the insurance cost per person based on various demographic and health-related factors. The dataset used for this analysis contains information about age, sex, BMI, number of children, smoking status, and region.

## Business Problem

Insurance companies need to predict the cost of insurance for individuals accurately to set premiums fairly and manage risks effectively. By using machine learning models to predict insurance charges, companies can better estimate costs, optimize pricing strategies, and improve customer satisfaction.

## Project Overview

The project follows these steps:
1. **Data Loading and Exploration**: Load the dataset and explore its structure.
2. **Data Cleaning and Preprocessing**: Check for missing values and encode categorical variables.
3. **Feature Selection**: Define features and target variable.
4. **Model Training**: Split the data into training and testing sets, and train a Linear Regression model.
5. **Model Evaluation**: Evaluate the model's performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
6. **Visualization**: Visualize the actual vs predicted charges and display the model coefficients.

## Dataset

The dataset `insurance.csv` contains the following columns:
- `age`: Age of the individual
- `sex`: Gender of the individual
- `bmi`: Body Mass Index
- `children`: Number of children/dependents covered by health insurance
- `smoker`: Smoking status of the individual
- `region`: Residential area of the individual
- `charges`: Medical costs billed by health insurance

## Dependencies

The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation

To install the required libraries, run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

##To run the project, execute the script insurance_cost_per_person_prediction.py:
##python insurance_cost_per_person_prediction.py

