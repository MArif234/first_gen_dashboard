import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_ml_models(df):
  """
  Going to train two different models to predict mid-career salary:
  - First model: Linear Regression
  - Second model: Random Forest
  - df: Dataframe the features Wage_Early, Grad_Degree_Share, Unemployment, and Underemployment and what we are trying to find (Mid Career Wage)
  """
  
  # X = features, y = target
  X = df[["Wage_Early", "Grad_Degree_Share", "Unemployment", "Underemployment"]]
  y = df["Wage_Mid"]

  # Splitting up the data so 20% can be used for testing and 80% of it can be used for training
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Linear Regression Model
  lin_model = LinearRegression()
  lin_model.fit(X_train, y_train)
  lin_preds = lin_model.predict(X_test)
  lin_r2 = r2_score(y_test, lin_preds)

  # Random Forest Regression

  rand_model = RandomForestRegressor(n_estimators=100, random_state=42)
  rand_model.fit(X_train, y_train)
  rand_preds = rand_model.predict(X_test)
  rand_r2 = r2_score(y_test, rand_preds)

  # Finding the important features
  import_features = pd.Series(rand_model.feature_importances_, index=X.columns)

  return {
    "Linear Regression Model": lin_model,
    "Random Forest Model": rand_model,
    "Test Features": X_test,
    "Test Target": y_test,
    "Linear Regression Predictions": lin_preds,
    "Random Forest Predictions": rand_preds,
    "Linear Regression Correlation": lin_r2,
    "Random Forest Correlation": rand_r2
  }