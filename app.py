# career comparison project
import streamlit as st
# need streamlit to build an interactive web app
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from data_files import load_salary_data
from visuals import (
  show_majors_based_on_category,
  plot_grad_deg_vs_mid_career_salary,
  show_low_return_majors,
  plot_early_vs_growth,
  plot_lin_reg_predictions,
  plot_rand_for_predictions,
  show_features_importance
)
from ml import train_ml_models

# Loading up the data
df = load_salary_data("majors_data.csv")

# Title and Description of Project
st.title("Learn About College Degree Outcomes")
st.write("Explore majors in regards to wage, unemployment, and underemployment.")

# Creating dropdown menu for category of choice

# Question: What are the top 10 college majors with the lowest [metric]?

# the 4 metrics are Unemployment Rate, Underemployment Rate, Median Wage Early Career, Median Wage Mid-Career

show_majors_based_on_category(df)


# What majors have the highest wage growth from early career to mid-career?


top_growth = df.sort_values("Wage_Growth", ascending=False).head(10)
st.subheader("Top Majors by Wage Growth")
st.table(top_growth[["Major", "Wage_Growth"]])

# Is there a correlation between percentage of people who have completed a graduate degree and mid-career wage?

st.subheader("Relationship Between Graduate Degree Share and Mid-Career Salary")

plot_grad_deg_vs_mid_career_salary(df)

correlation = df["Grad_Degree_Share"].corr(df["Wage_Mid"])
st.write(f"Correlation between Grad Degree Share and Mid-Career Wage: {correlation:.2f}")

show_low_return_majors(df)

# Which majors offer low employment rates and high mid-career wages (best stability + pay combo)?



st.subheader("Majors with Low Unemployment and High Mid-Career Wages")

# Add ways for the user to determine desired unemployment rate and mid-career wage

unemployment_threshold = st.slider("Maximum Unemployment Rate (%)", min_value = 0.0, max_value=10.0, value=3.0, step=0.1)

wage_threshold = st.slider("Minimum Mid-Career Wage ($)", min_value=40000, max_value=150000, value=90000, step=1000)

filtered_df = df[(df["Unemployment"] < unemployment_threshold) & (df["Wage_Mid"] > wage_threshold)]

st.write(f"Majors with Unemployment < {unemployment_threshold}% and Mid-Career Wage > ${wage_threshold}")

st.table(filtered_df[["Major", "Unemployment", "Wage_Mid"]].sort_values(by="Wage_Mid", ascending=False))

# Which majors have high early career salaries but poor mid-career salary growth?

st.subheader("Explore Majors with High Early Salary but Low Wage Growth")

# Adjust the Percentiles to Your Liking

early_salary_percentile = st.slider("Minimum Early Career Salary Percentile", 50, 100, 75)

wage_growth_percentile = st.slider("Maximum Wage Growth Percentile", 0, 50, 25)

# Calculate the thresholds

early_salary_threshold = df["Wage_Early"].quantile(early_salary_percentile / 100)

wage_growth_threshold = df["Wage_Growth"].quantile(wage_growth_percentile / 100)

# Filter for low-growth majors with high early salaries

low_growth_high_early = df[(df["Wage_Early"] >= early_salary_threshold) & (df["Wage_Growth"] <= wage_growth_threshold)]

st.write(f"Filtering for majors in the **top {100 - early_salary_percentile}%** of early salaries and **bottom {wage_growth_percentile}%** of wage growth.")

st.table(low_growth_high_early[["Major", "Wage_Early", "Wage_Mid", "Wage_Growth"]])

# Create a plot for the majors

plot_early_vs_growth(df, low_growth_high_early)


# Goal: Predict Mid-Career Salary for College Majors


st.header("Predicting Mid-Career Salary Using Early Career Data")

st.write("""
This dashboard uses **machine learning models** to predict the median mid-career salary for different college majors. We use features like early-career salary, graduate degree share, unemployment, and underemployment to train: 

- A **Linear Regression** model
- A **Random Forest** model

Then we compare their performance and explore which features were the most influential. 
         """)

ml_results = train_ml_models(df)

st.subheader("Model Performance (Correlation Coefficient)")

st.write("**Linear Regression Correlation Coefficient**", round(ml_results["Linear Regression Correlation"], 2))

st.write("**Random Forest Correlation Coefficient Score:**", round(ml_results["Random Forest Correlation"], 2))

# Show Prediction vs Actual Results

st.subheader("Compare Predictions vs. Actual Mid-Career Salaries")

comparison_df = pd.DataFrame({
  "Actual": ml_results["Test Target"],
  "Linear Regression": ml_results["Linear Regression Predictions"],
  "Random Forest": ml_results["Random Forest Predictions"]
})
st.dataframe(comparison_df.head(10))