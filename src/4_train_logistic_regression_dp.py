import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load DP-protected data
df_dp = pd.read_csv("D:\Winter Term 2025\ELG5295-W00(Ethics,AI & Robotics)\DP_Logistic_Regression_Project\data\dp_protected_cancer_data.csv")

# Define features & target
X_dp = df_dp.drop(columns=["Diagnosis"])
y_dp = df_dp["Diagnosis"]

# Split data
X_train_dp, X_test_dp, y_train_dp, y_test_dp = train_test_split(X_dp, y_dp, test_size=0.2, random_state=42)

# Train Logistic Regression
model_dp = LogisticRegression()
model_dp.fit(X_train_dp, y_train_dp)

# Save DP model
joblib.dump(model_dp, "D:\Winter Term 2025\ELG5295-W00(Ethics,AI & Robotics)\DP_Logistic_Regression_Project\models\logistic_regression_dp.pkl")

# Evaluate model
y_pred_dp = model_dp.predict(X_test_dp)
print("📊 Logistic Regression with DP Data Performance:")
print(classification_report(y_test_dp, y_pred_dp))
