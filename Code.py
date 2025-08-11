import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("E:\\.OASIS. INTERNSHIP PROJECT\\Task_1\\Iris.csv")

# Show first 5 rows
print(df.head())

# Dataset summary
print(df.info())

# Basic statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check class distribution
print(df['Species'].value_counts())

# Pairplot to see relation between features
sns.pairplot(df, hue="Species")
plt.show()

# Heatmap of correlations
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
