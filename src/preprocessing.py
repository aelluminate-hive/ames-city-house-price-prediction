import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(data, drop_columns):
    # Drop specified columns
    X = data.drop(columns=drop_columns)
    y = data["SalePrice"]

    # Drop columns with all missing values
    X = X.dropna(axis=1, how="all")

    # Preprocessing pipeline for numerical features
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
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

    return X, y, preprocessor
