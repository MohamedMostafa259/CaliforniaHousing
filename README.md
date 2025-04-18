# California Housing Price Prediction

This repository contains a hands-on end-to-end machine learning project based on Chapter 2 of the book **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron. The goal of this project is to build a regression model to predict California housing prices based on various features using real-world data.

## Dataset

The dataset used in this project is the **California Housing Dataset**, which contains information collected from the 1990 California census. The dataset includes features such as population, housing median age, total rooms, bedrooms, and more.

The dataset is programmatically downloaded from [this source](https://github.com/ageron/data/raw/main/housing.tgz) for reproducibility.

### Data Preprocessing

- **Handling missing values:** Imputed missing values using median strategy.
- **Categorical encoding:** Applied one-hot encoding on the `ocean_proximity` feature.
- **Log transformation:** Applied log transformation on skewed features to reduce the effect of outliers.
- **Feature scaling:** Standardized all numerical features.
- **Capping removal:** Removed rows with maximum values in the target variable to avoid bias.

### Feature Engineering

- Created new ratio-based features such as:
  - `rooms_per_house`
  - `bedrooms_ratio`
  - `people_per_house`
- Applied **KMeans clustering** on geographical coordinates to compute location-based similarity features using **RBF Kernel**.
- Modeled the multimodal distribution of `housing_median_age` using similarity scores to key values (e.g., 16, 26, 35).

### Visualization & Correlation Analysis

- Visualized distributions of features before and after transformations.


- Created scatter plots to understand geographical trends.

  ![Geographical Distribution of Housing Prices in California](https://github.com/MohamedMostafa259/CaliforniaHousing/blob/main/visualizations/Geographical%20Distribution%20of%20Housing%20Prices%20in%20California.png)

    - Housing prices are significantly higher near the coast, especially in the Bay Area (San Francisco) and around Los Angeles.
    - Larger populations are clustered around the coastal regions, indicating urban centers.

- Used heatmaps and bar charts to explore relationships among features.

- Analyzed correlation with the target variable `median_house_value`.

### Pipelines

- Built **preprocessing pipelines** for:
  - Numerical features
  - Categorical features
  - Log-transformed features
  - Ratio features
  - Location similarity features
  - Age similarity features
- Custom transformers were created using Scikit-learn's `BaseEstimator` and `TransformerMixin`.

### Modeling

- Split the dataset using **stratified sampling** based on income categories.
- Used ensemble model: **XGBoost Regressor**.
- Evaluated models using RMSE and R² metrics.
