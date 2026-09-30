import pandas as pd

# Load the dataset
df = pd.read_csv("titanic.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
df.info()

# Number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Drop deck column because it has many missing values
df = df.drop(columns=["deck"])

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Basic insights
print("\nEDA Insights:")

print("Average Age:", round(df["age"].mean(), 2))
print("Average Fare:", round(df["fare"].mean(), 2))
print("Survival Rate:", round(df["survived"].mean() * 100, 2), "%")
print("Male Passengers:", (df["sex"] == "male").sum())
print("Female Passengers:", (df["sex"] == "female").sum())

# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")

import matplotlib.pyplot as plt
import seaborn as sns

# Survival count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="survived")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.savefig("survival_count.png")
plt.show()

# Age distribution
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="age", bins=20)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.savefig("age_distribution.png")
plt.show()