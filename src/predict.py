import pandas as pd


def make_predictions(model, test_data_path, output_path):
    test_data = pd.read_csv(test_data_path)
    test_predictions = model.predict(test_data)

    # Save the predictions to a CSV file
    submission = pd.DataFrame({"Id": test_data["Id"], "SalePrice": test_predictions})
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
