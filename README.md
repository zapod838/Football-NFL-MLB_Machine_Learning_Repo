# Football-NFL-MLB Machine Learning Repo

![AI-In-Sports](https://github.com/user-attachments/assets/19665445-6469-4a4f-8753-370da56e995e)

This repository contains machine learning models designed to predict game outcomes for Football (NFL) and MLB games. The primary objectives of the models are:

- Predict the **winner** of the game.
- Provide the **confidence** percentage for the prediction.
- Predict the **Over/Under Number**.
- Determine whether the prediction is **Over** or **Under**.
- Provide the **confidence** percentage for the Over/Under prediction.

## Data Processing Steps

1. **Null Value Handling**:
   - Identified null values in all columns.
   - Dropped columns with more than **50% null values**.
   - For the remaining columns with null values, identified symmetric and skewed columns:
     - Imputed null values with **mean** for symmetric columns.
     - Imputed null values with **median** for skewed columns.
     - Applied imputation based on `home_name` and `away_name` teams.
   - Saved the imputed dataframe to a CSV file for modeling.

## Feature Selection & Modeling Approach

1. **Feature Selection**:
   - Applied **Recursive Feature Elimination (RFE)** using a RandomForestClassifier to identify the most relevant features.
   - Selected features with a ranking of **5 or less**.
   - Created a new dataset (`df_new`) with the selected features.
   - Split the data into **training**, **validation**, and **testing** sets.

2. **Model Training**:
   - **Scaling**: Used **RobustScaler** normalization to scale the data.
   - **Handling Imbalanced Data**: Applied a **focal loss function** to address imbalanced datasets.
   - **Model Structure**: 
     - To prevent overfitting, incorporated techniques like **dropout**, **batch normalization**, and **regularization**.
   - The model predicts both the game winner and the over/under results.

## Model Architecture

The model is a **Deep Neural Network (DNN)** with the following key components:

1. **Input Layer**:
   - A Dense layer with **64 neurons** using the **ReLU activation** function.
   - **L2 regularization** to penalize large weights and prevent overfitting.
   - **Batch Normalization** for stabilizing and speeding up training.
   - **Dropout (0.4)** to randomly drop neurons and prevent overfitting.

2. **Hidden Layers**:
   - Second Dense layer with **32 neurons**, **L2 regularization**, Batch Normalization, and Dropout (0.4).
   - Third Dense layer with **16 neurons**, **L2 regularization**, Batch Normalization, and Dropout (0.4).

3. **Output Layer**:
   - A Dense layer with **3 neurons** using **softmax activation** for multi-class classification (predicting outcomes like Home Win, Away Win, or Draw).

4. **Loss Function**:
   - The model uses **Focal Loss** to handle class imbalance in the sports dataset. 
     - **Gamma (γ=2)** focuses learning on hard-to-classify examples.
     - **Alpha (α=0.25)** balances the class weights.

5. **Training**:
   - Optimized with the **Adam optimizer**.
   - **Metrics**: Tracks **accuracy** during training.

## Inference Pipeline

1. **retrieve_historical_data**:
   - This function retrieves historical data for the given combination of `home_team` and `away_team`, with a lookback period of **6 games**.

2. **create_features_vector**:
   - Converts the historical data into a feature vector suitable for model input.
   - Scales the features using the previously trained scaler.

3. **predict_game_outcome**:
   - Utilizes the trained model to predict the game outcome from the feature vector.

## Predictions & Outputs

For each prediction, the model provides:
- **Winner of the Game**.
- **Confidence Percentage** in the winner prediction.
- **Over/Under Number**.
- Whether the prediction is **Over** or **Under**.
- **Confidence Percentage** for the Over/Under prediction.
