import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv("D:\Winter Term 2025\ELG5295-W00(Ethics,AI & Robotics)\DP_Logistic_Regression_Project\data\The_Cancer_data_1500_V2.csv")

# Define noise scale (adjustable)
noise_scale = 0.1  

# Add Gaussian noise to sensitive features
sensitive_features = ["Age", "BMI", "PhysicalActivity"]
df_dp = df.copy()
for col in sensitive_features:
    df_dp[col] += np.random.normal(0, noise_scale, df_dp.shape[0])

# Save DP-protected dataset
df_dp.to_csv("D:\Winter Term 2025\ELG5295-W00(Ethics,AI & Robotics)\DP_Logistic_Regression_Project\data\dp_protected_cancer_data.csv", index=False)
print("✅ DP-protected dataset saved successfully.")
