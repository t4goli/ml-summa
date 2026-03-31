import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("student_exam_scores.csv", sep=",")

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


y_pred = model.predict(X_test_scaled)
print(y_pred[:5])

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print(results.head(10))

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

print(model.coef_)
print(model.intercept_)
