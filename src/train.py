import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

print("=== Student Exam Score Prediction ===")
df = pd.read_csv("data/student_exam_scores.csv")

print(df.columns)
y = df["exam_score"]
X = df.drop(["exam_score", "student_id"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

model = LinearRegression()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = model.predict(X_test_scaled)
y_pred_tree = tree_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
rf_model.fit(X_train, y_train)
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
print("\n--- Model Results ---")
# Predictions on training data
y_train_pred_tree = tree_model.predict(X_train)

# Training error
mse_train_tree = mean_squared_error(y_train, y_train_pred_tree)

mse = mean_squared_error(y_test, y_pred)

mse_tree = mean_squared_error(y_test, y_pred_tree)

mse_rf = mean_squared_error(y_test, y_pred_rf)

importances = rf_model.feature_importances_
feature_names = X.columns

for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance}")

print("Linear Regression MSE:", mse)
print("Decision Tree MSE:", mse_tree)
print("Random Forest MSE:", mse_rf)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Exam Scores")
plt.show()