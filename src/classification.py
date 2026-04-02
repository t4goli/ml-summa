import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("=== Student Exam Score Prediction ===")
df = pd.read_csv("data/student_exam_scores.csv")
df["pass"] = df["exam_score"] >= 35

print(df.columns)
y = df["pass"]
X = df.drop(["exam_score", "student_id", "pass"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
print(results.head())
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)