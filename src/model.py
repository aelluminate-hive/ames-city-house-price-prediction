import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_model(X, y, preprocessor, model_type="linear"):
    if model_type == "linear":
        regressor = LinearRegression()
    elif model_type == "decision_tree":
        regressor = DecisionTreeRegressor(random_state=42)
    elif model_type == "random_forest":
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "gradient_boosting":
        regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == "xgboost":
        regressor = xgb.XGBRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type")

    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    return model
