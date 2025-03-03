import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# Load cleaned data
df = pd.read_csv(r"D:\Winter Term 2025\ELG5295-W00(Ethics,AI & Robotics)\DP_Logistic_Regression_Project\data\cleaned_cancer_data.csv")

# Check column names
print("Columns in dataset:", df.columns)

# Update target column name
target_col = "Diagnosis"  # Change from "Target" to "Diagnosis"

if target_col not in df.columns:
    raise KeyError(f"Column '{target_col}' not found. Available columns: {df.columns}")

# Define features & target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Ensure 'models/' folder exists before saving
os.makedirs(r"D:\Winter Term 2025\ELG5295-W00(Ethics,AI & Robotics)\DP_Logistic_Regression_Project\models", exist_ok=True)

# Save model
joblib.dump(model, r"D:\Winter Term 2025\ELG5295-W00(Ethics,AI & Robotics)\DP_Logistic_Regression_Project\models\logistic_regression.pkl")

# Evaluate model
y_pred = model.predict(X_test)
print("📊 Logistic Regression Performance:")
print(classification_report(y_test, y_pred))
