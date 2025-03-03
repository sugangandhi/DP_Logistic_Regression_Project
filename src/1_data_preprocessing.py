import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("D:\Winter Term 2025\ELG5295-W00(Ethics,AI & Robotics)\DP_Logistic_Regression_Project\data\The_Cancer_data_1500_V2.csv")

# Handle missing values
df.fillna(df.median(), inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# Scale numerical features
scaler = StandardScaler()
num_cols = ["Age", "BMI", "PhysicalActivity"]
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save cleaned dataset
df.to_csv("D:\Winter Term 2025\ELG5295-W00(Ethics,AI & Robotics)\DP_Logistic_Regression_Project\data\cleaned_cancer_data.csv", index=False)
print("✅ Cleaned dataset saved successfully.")
