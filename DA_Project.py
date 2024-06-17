import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

# Load the dataset
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError as e:
    print("File not found. Please ensure 'data.csv' is in the correct directory.")
    raise e

# Data Preprocessing
# Handling missing values if any
data.fillna(method='ffill', inplace=True)

# Splitting features and target variables
X = data.drop(['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'], axis=1)
y = data[['xyz_vaccine', 'seasonal_vaccine']]

# Identifying categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Preprocessing pipeline for numerical data
numerical_transformer = StandardScaler()

# Preprocessing pipeline for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestClassifier(random_state=42)

# Create and evaluate the pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', MultiOutputClassifier(model))])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning (optional but recommended)
param_grid = {
    'classifier__estimator__n_estimators': [100, 200],
    'classifier__estimator__max_depth': [None, 10, 20]
}
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Make predictions
y_pred_prob = grid_search.predict_proba(X_test)

# Extract probabilities for each class
xyz_pred_prob = y_pred_prob[0][:, 1]
seasonal_pred_prob = y_pred_prob[1][:, 1]

# Calculate ROC AUC scores
roc_auc_xyz = roc_auc_score(y_test['xyz_vaccine'], xyz_pred_prob)
roc_auc_seasonal = roc_auc_score(y_test['seasonal_vaccine'], seasonal_pred_prob)

# Print the scores
print(f'ROC AUC Score for xyz_vaccine: {roc_auc_xyz}')
print(f'ROC AUC Score for seasonal_vaccine: {roc_auc_seasonal}')
print(f'Mean ROC AUC Score: {(roc_auc_xyz + roc_auc_seasonal) / 2}')

# Prepare submission file
submission = pd.DataFrame({
    'respondent_id': data['respondent_id'],
    'xyz_vaccine': xyz_pred_prob,
    'seasonal_vaccine': seasonal_pred_prob
})

submission.to_csv('submission.csv', index=False)
