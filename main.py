import inquirer
from src.preprocessing import load_data, preprocess_data
from src.model import train_model
from src.predict import make_predictions
from utils.constants import PREPROCESSED_TRAIN_DATASET, TEST_DATASET

# Prompt the user to select the regression model
questions = [
    inquirer.List(
        "model_type",
        message="Which regression model would you like to use?",
        choices=[
            "Linear",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
        ],
    )
]
answers = inquirer.prompt(questions)
model_type = answers["model_type"].lower().replace(" ", "_")

# Load and preprocess the training data
data = load_data(PREPROCESSED_TRAIN_DATASET)
drop_columns = [
    "SalePrice",
    "SalePriceScaled",
    "RemodAge",
    "HouseAge",
    "NeighborhoodFull",
]
X, y, preprocessor = preprocess_data(data, drop_columns)

# Train the model with the selected model type
model = train_model(X, y, preprocessor, model_type=model_type)

# Make predictions on the test set and save to CSV
make_predictions(model, TEST_DATASET, "datasets/submission.csv")
