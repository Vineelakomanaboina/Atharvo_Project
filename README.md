# Random Forest Regression for Temperature Prediction

# Overview
This project uses a Random Forest Regressor to predict actual temperatures based on historical weather data. The dataset is processed to remove labels, perform one-hot encoding, and split into training and testing sets. The Random Forest model is trained, and its performance is evaluated based on Mean Absolute Error (MAE) and accuracy.

# Project Structure
Data Loading: A CSV file containing temperature data is loaded into a Pandas DataFrame.
Feature Engineering: One-hot encoding is applied to categorical variables, and the target variable (actual temperature) is separated from the feature set.
Train-Test Split: Data is split into training and testing sets using an 80-20 ratio.
Model Training: A RandomForestRegressor model is trained on the training data.
Prediction and Evaluation: The model is used to make predictions on the test set, and the accuracy is calculated based on the Mean Absolute Percentage Error (MAPE).
Tree Visualization: A single decision tree from the Random Forest is exported and visualized using pydot.

# Requirements
To run this project, you will need the following Python libraries:
pandas
numpy
matplotlib
scikit-learn
pydot

# Install these dependencies using:
bash
Copy code
pip install pandas numpy matplotlib scikit-learn pydot

# How to Run
Load the Data: Load the CSV file containing the weather data using Pandas.
Feature Engineering: Perform one-hot encoding on categorical features and separate the label (actual temperature) from the features.
Split the Data: Split the dataset into training and testing sets using train_test_split.
Train the Model: Instantiate and train a RandomForestRegressor model on the training data.
Make Predictions: Use the trained model to make predictions on the test data.
Evaluate the Model: Calculate the Mean Absolute Error (MAE) and accuracy of the model.
Visualize a Tree: Export a single decision tree from the trained Random Forest and save it as a PNG image.
Example Results
Mean Absolute Error (MAE): The model achieves a mean absolute error of approximately X degrees, where X is the temperature difference between predicted and actual values.
Accuracy: The model's prediction accuracy is approximately Y%.

# Example output:
plaintext
Copy code
Mean Absolute Error: 3.4 degrees.
Accuracy: 94.5 %.
Data Description
The dataset used in this project contains historical temperature data. The actual temperature is the target variable, while other columns represent different features such as year, month, week, and other weather-related information.

# Model Overview
The RandomForestRegressor is an ensemble learning method that constructs multiple decision trees during training and outputs the mean prediction of individual trees. This model helps to improve prediction accuracy by reducing overfitting that may occur with a single decision tree.

# Visualizing the Decision Tree
One tree from the Random Forest model is visualized using pydot. The graph is exported as a PNG file, providing insights into the decision process of one of the trees within the Random Forest.

# Future Work
Test different hyperparameters using Grid Search or Random Search for optimization.
Experiment with other regression models, such as Gradient Boosting Regressor or XGBoost.
Use additional feature engineering techniques to improve model performance
