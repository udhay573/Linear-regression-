import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load CSV file into a pandas DataFrame
data = pd.read_csv('/content/Salary_Data.csv')

# Display the first few rows of the DataFrame and check for null values
print(data.head())
print("\nNull values in the dataset:")
print(data.isnull().sum())

# Handle null values if any (for demonstration purpose)
# Replace null values with median value of YearsExperience
data['YearsExperience'].fillna(data['YearsExperience'].median(), inplace=True)

# Separate features and target
X = data[['YearsExperience']]  # Feature(s)
y = data['Salary']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Display intercept and coefficients
intercept = linear_model.intercept_
coefficients = linear_model.coef_

print(f"Intercept: {intercept:.2f}")
print(f"Coefficient: {coefficients[0]:.2f}")

# Predictions
y_pred_train = linear_model.predict(X_train)
y_pred_test = linear_model.predict(X_test)

# Calculate metrics
linear_mae_train = mean_absolute_error(y_train, y_pred_train)
linear_mse_train = mean_squared_error(y_train, y_pred_train)
linear_rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
linear_r2_train = r2_score(y_train, y_pred_train)

linear_mae_test = mean_absolute_error(y_test, y_pred_test)
linear_mse_test = mean_squared_error(y_test, y_pred_test)
linear_rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
linear_r2_test = r2_score(y_test, y_pred_test)

print("\nTraining Set Metrics:")
print(f"MAE: {linear_mae_train:.2f}")
print(f"MSE: {linear_mse_train:.2f}")
print(f"RMSE: {linear_rmse_train:.2f}")
print(f"R2 Score: {linear_r2_train:.2f}")

print("\nTesting Set Metrics:")
print(f"MAE: {linear_mae_test:.2f}")
print(f"MSE: {linear_mse_test:.2f}")
print(f"RMSE: {linear_rmse_test:.2f}")
print(f"R2 Score: {linear_r2_test:.2f}")

# Plotting function
def plot_regression(X, y, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting for training set
plot_regression(X_train, y_train, y_pred_train, 'Linear Regression: Training Set')

# Plotting for testing set
plot_regression(X_test, y_test, y_pred_test, 'Linear Regression: Testing Set')
