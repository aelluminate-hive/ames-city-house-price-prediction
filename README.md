# Ames City House Price Prediction

This project aims to leverage the Ames Housing Price Dataset to develop an accurate predictive model for estimating residential property sale prices in Ames, Iowa. By analyzing various property features and their relationships to sale prices, the project seeks to uncover insights that can inform both buyers and sellers in the housing market.

## Usage

To generate `preprocessed.csv` dataset, run the notebook `report.ipynb` in the `notebooks` directory. You need to have the required libraries installed. To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```
### Why Do You Need to Preprocess the Data?

The dataset contains a large number of features, many of which are categorical. To make the data suitable for machine learning models, we need to preprocess the data by encoding categorical features and handling missing values. The `preprocessed.csv` dataset is the result of this preprocessing.

### How to Use the Application?

Run the `main.py` script to start the application. 

```bash
python main.py
```
or

```bash
py -m main
```

The application will prompt you on what machine learning model you want to use to predict house prices. You can choose from the following models:

1. Linear Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

### Example Usage

```bash
[?] Which regression model would you like to use?:                                                               
 > Linear
   Decision Tree
   Random Forest
   Gradient Boosting
   XGBoost
```

After choosing a model, the application will train the model on the preprocessed dataset and display the model's performance metrics such as MAE and R2 score. Then inside the `constants.py` file inside the `utils` directory, there's a `SAMPLE_SUBMISSION` variable that contains a sample submission. You can use this sample submission to test the model's prediction on the test dataset and the application will generate a `submission.csv` file in the `datasets` directory containing the predicted house prices.

## Introduction

The Ames Housing Price Dataset is a popular dataset in the field of machine learning and data science. It contains 79 features that describe various aspects of residential properties in Ames, Iowa. The dataset is often used to develop predictive models for estimating house prices based on these features. The dataset is divided into two parts: a training set and a test set. The training set contains 1460 observations, while the test set contains 1459 observations. The goal of this project is to develop an accurate predictive model for estimating house prices in Ames, Iowa.

## Project Goal

The goal of this project is to develop an accurate predictive model for estimating house prices in Ames, Iowa. By analyzing the dataset and building a predictive model, we aim to provide insights that can inform both buyers and sellers in the housing market. 

## The Data

The Ames Housing Price Dataset contains 79 features that describe various aspects of residential properties in Ames, Iowa. The dataset is divided into two parts: a training set and a test set. The training set contains 1460 observations, while the test set contains 1459 observations. The dataset contains a mix of numerical and categorical features, as well as missing values that need to be handled.

##### For more information, please refer to the [Ames City Real Estate](https://gitlab.com/aelluminate/databank/2024-10/ames-city-real-estate) repository.

## Methodology

- **Data Preprocessing**: The dataset contains a mix of numerical and categorical features, as well as missing values that need to be handled. We preprocess the data by encoding categorical features and handling missing values.
- **Exploratory Data Analysis (EDA)**: We analyze the relationships between various features and the target variable (sale price) to uncover insights that can inform our predictive model.
- **Visualizations**: We create visualizations to better understand the relationships between features and the target variable.
- **Feature Engineering**: We create new features by combining existing features to improve the predictive power of our model.
- **Model Building**: We build machine learning models using various regression algorithms such as Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
- **Model Evaluation**: We evaluate the performance of our models using metrics such as Mean Absolute Error (MAE) and R2 Score.
- **Model Selection**: We select the best-performing model based on the evaluation metrics.
- **Prediction**: We use the selected model to predict house prices on the test dataset.
- **Submission**: We generate a submission file containing the predicted house prices.
- **Model Deployment**: We deploy the model as an standalone application that allows users to select a regression model and predict house prices.

## Tools

The project is implemented using the following tools: **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Scikit-learn**, and **XGBoost**.

## Visualization

### Correlation of Numerical Features

![Correlation of Numerical Features](https://i.imgur.com/nifPGnV.png)

> - `SalePrice` has strong positive correlations with `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, and `1stFlrSF`, suggesting that these features are significant drivers of the sale price.    
> - `TotRmsAbvGrd` and `GrLivArea` have a strong positive correlation, indicating that houses with more rooms above grade tend to have larger living areas.  
> - `GarageCars` and `GarageArea` have a strong positive correlation, as expected, since houses with more garage bays typically have larger garage areas.
> 
> Overall, the correlation matrix reveals that the sale price is primarily influenced by factors related to the house's overall quality, size, and living space, as well as the presence of certain amenities like a garage. However, it's important to note that correlation does not imply causation, and further analysis would be needed to establish definitive relationships between these features and the sale price.

### House Age vs. Sales Price

![House Age vs. Sales Price](https://i.imgur.com/uWQDxNK.png)

> - The overall trend in the scatter plot indicates a negative correlation between house age and sale price. This suggests that, generally, as houses get older, their sale prices tend to decrease.  
> - There are noticeable clusters of data points, particularly in the lower age ranges. This might indicate that certain factors, such as location, neighborhood characteristics, or specific features, influence sale prices more than age in these particular clusters.
> - A few data points appear to be outliers, deviating significantly from the general trend. These could be due to unique properties (e.g., historical significance, exceptional views), renovations, or other factors that significantly affect the sale price.
> - The red regression line visually represents the linear relationship between house age and sale price. The downward slope confirms the negative correlation. 

### Remodeling Impact on Sale Price

![Remodeling Impact on Sale Price](https://i.imgur.com/r5t2TiP.png)

> - The overall trend in the scatter plot indicates a negative correlation between remodeling age and sale price. This suggests that, generally, as the time since remodeling increases, the sale price tends to decrease.
> - There are noticeable clusters of data points, particularly in the lower remodeling age ranges. This might indicate that other factors, such as location, neighborhood characteristics, or specific features, influence sale prices more than remodeling age in these particular clusters.
> - A few data points appear to be outliers, deviating significantly from the general trend. These could be due to unique properties (e.g., historical significance, exceptional views), extensive renovations, or other factors that significantly affect the sale price.
> - The red regression line visually represents the linear relationship between remodeling age and sale price. The downward slope confirms the negative correlation. 

### House Age Distribution

![House Age Distribution](https://i.imgur.com/vwehprN.png)

> - The distribution is right-skewed, meaning there is a longer tail on the right side. This indicates that there are a few older houses (with higher ages) that pull the average age to the right.
> - The distribution appears to have a mode between 0 and 10 years. This suggests that a significant number of houses in the dataset are relatively new.
> - The distribution is leptokurtic, meaning it has a heavier tail than a normal distribution. This indicates that there are more extreme values (either very young or very old houses) than would be expected in a normal distribution.

### Remodeling Age Distribution

![Remodeling Age Distribution](https://i.imgur.com/EExIwdz.png)

> - The distribution is right-skewed, meaning there is a longer tail on the right side. This indicates that there are a few houses that were remodeled a long time ago (with higher remodeling ages) that pull the average remodeling age to the right.
> The distribution appears to have a mode between 0 and 5 years. This suggests that a significant number of houses in the dataset were either newly built or recently remodeled.
> - The distribution is leptokurtic, meaning it has a heavier tail than a normal distribution. This indicates that there are more extreme values (either very recent or very old remodels) than would be expected in a normal distribution.

### House with Pool vs. Sale Price

![House with Pool vs. Sale Price](https://i.imgur.com/jSk99Bj.png)

> - The majority of data points for houses with pools are clustered at higher sale prices compared to those without pools. This suggests that having a pool generally increases the value of a house.
> - There are a few outliers, especially among houses without pools, that have relatively high sale prices. These could be due to other factors, such as location, size, or unique features, that outweigh the absence of a pool.
> - The sale price ranges for houses with and without pools overlap to some extent. This indicates that while having a pool can generally increase the value, other factors also play a significant role in determining the final sale price.

###### For more visualizations, please refer to the [notebook](/notebooks/report.ipynb).

## Model Evaluation

The following table summarizes the performance metrics of the regression models:

| Model | MAE | R2 Score |
|----|----|----|
| Linear Regression      | 18221.624     | 0.886     |
| Decision Tree          | 26651.479    | 0.795    |
| Random Forest          | 17593.370    | 0.888    |
| **Gradient Boosting**      | **17205.792**   | **0.903**    |
| **XGBoost**               | **17313.087**   | **0.902**   |

## Conclusion

The Gradient Boosting and XGBoost models outperformed the other models in terms of Mean Absolute Error (MAE) and R2 Score. These models demonstrated strong predictive power in estimating house prices based on the dataset's features. The insights gained from the analysis and visualizations can help inform buyers and sellers in the housing market by highlighting the key factors that influence house prices in Ames, Iowa.