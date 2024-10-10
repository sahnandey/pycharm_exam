import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Load the dataset
# Note: In a real scenario, you would use: df = pd.read_csv('customer_data.csv')
# Creating sample data for demonstration
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'age': np.random.normal(35, 12, n_samples).astype(int),
    'gender': np.random.binomial(1, 0.5, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'loyalty_status': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_samples),
    'purchase_frequency': np.random.poisson(5, n_samples),
    'purchase_amount': np.random.normal(200, 100, n_samples),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_samples),
    'promotion_usage': np.random.binomial(1, 0.3, n_samples),
    'satisfaction_score': np.random.normal(7, 1.5, n_samples).clip(0, 10)
})

# Basic EDA
print("First few rows of the dataset:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Age and Income distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['age'], bins=30, ax=ax1)
ax1.set_title('Age Distribution')
sns.histplot(df['income'], bins=30, ax=ax2)
ax2.set_title('Income Distribution')
plt.tight_layout()
plt.show()

# Purchase amount by product category
plt.figure(figsize=(10, 6))
sns.boxplot(x='product_category', y='purchase_amount', data=df)
plt.xticks(rotation=45)
plt.title('Purchase Amount Distribution by Product Category')
plt.tight_layout()
plt.show()

# Gender proportion pie chart
plt.figure(figsize=(8, 8))
gender_counts = df['gender'].value_counts()
plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%')
plt.title('Customer Gender Distribution')
plt.show()

# Average purchase amount by education
edu_purchase = df.groupby('education')['purchase_amount'].mean().sort_values(ascending=False)
print("\nAverage purchase amount by education level:")
print(edu_purchase)

# Average satisfaction score by loyalty status
loyalty_satisfaction = df.groupby('loyalty_status')['satisfaction_score'].mean().sort_values(ascending=False)
print("\nAverage satisfaction score by loyalty status:")
print(loyalty_satisfaction)

# Purchase frequency by region
plt.figure(figsize=(10, 6))
sns.barplot(x='region', y='purchase_frequency', data=df)
plt.title('Average Purchase Frequency by Region')
plt.show()

# Promotion usage percentage
promo_percentage = (df['promotion_usage'].mean() * 100)
print(f"\nPercentage of customers using promotions: {promo_percentage:.2f}%")

# Predictive Modeling
# Prepare data for regression
categorical_cols = ['education', 'region', 'loyalty_status', 'product_category']
X = pd.get_dummies(df.drop(['purchase_amount', 'promotion_usage'], axis=1))
y_regression = df['purchase_amount']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Train regression model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Evaluate regression model
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nRegression Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Feature importance for regression
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_regressor.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 3 features for purchase amount prediction:")
print(feature_importance.head(3))

# Classification Model
X_class = X
y_class = df['promotion_usage']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_class, y_train_class)

# Evaluate classification model
y_pred_class = rf_classifier.predict(X_test_class)
print("\nClassification Model Performance:")
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class))

# Feature importance for classification
feature_importance_class = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 3 features for promotion usage prediction:")
print(feature_importance_class.head(3))
