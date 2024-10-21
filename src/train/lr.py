import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from utils.constants import PREPROCESSED_TRAIN_DATASET, TEST_DATASET

# Load the preprocessed dataset
data = pd.read_csv(PREPROCESSED_TRAIN_DATASET)

# Define the features and target variable
X = data.drop(
    columns=["SalePrice", "SalePriceScaled", "RemodAge", "HouseAge", "NeighborhoodFull"]
)
y = data["SalePrice"]

# Preprocessing pipeline for numerical features
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# Preprocessing pipeline for categorical features
categorical_features = X.select_dtypes(include=["object"]).columns
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Define the model
model = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# Preview the evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Load the test dataset
test_data = pd.read_csv(TEST_DATASET)

# Make predictions on the test set
test_predictions = model.predict(test_data)

# Save the predictions to a CSV file
submission = pd.DataFrame({"Id": test_data["Id"], "SalePrice": test_predictions})
submission.to_csv("datasets/submission.csv", index=False)
print("Predictions saved to submission.csv")
