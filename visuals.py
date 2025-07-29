import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# sorting majors based on the metric category
def show_majors_based_on_category(df):

  metric = st.selectbox("Choose a category to sort by:", ["Unemployment", "Underemployment", "Wage_Early", "Wage_Mid"])

  # if category not a salary, make sure you are 
  # sorting from lowest to highest 
  # ex: lower employment better so sort lowest to highest
  ascending = metric not in ["Wage_Early", "Wage_Mid"]
  sorted_df = df.sort_values(by=metric, ascending=ascending)

  st.subheader(f"Top 10 majors by {'lowest' if ascending else 'highest'} {metric}")
  st.table(sorted_df[["Major", metric]].head(10))

  # Plotting grad degree vs mid-career salary

  # Grad Degree Share = % of ppl that completed grad degree
def plot_grad_deg_vs_mid_career_salary(df):

  

  fig, ax = plt.subplots()

  ax.scatter(df["Grad_Degree_Share"], df["Wage_Mid"], alpha=0.7)
    # points mostly solid color with alpha=0.7

  ax.set_xlabel("Share with Graduate Degree")
  ax.set_ylabel("Median Mid-Career Salary ($)")
  ax.set_title("Grad Degree Share vs. Mid-Career Salary")
  st.pyplot(fig)

# identifies careers where grad school may not pay off
# trying to highlight high grad school rates but pay still low
def show_low_return_majors(df):
  st.subheader("Majors with High Grad Share but Low Mid-Career Wages")
  low_return = df[(df["Grad_Degree_Share"] > 0.5) & (df["Wage_Mid"] < 70000)]
  st.table(low_return[["Major", "Grad_Degree_Share", "Wage_Mid"]])

# comparing early career salary to long-term growth
# Wage_Early = wage when college students graduate
# Wage_Growth = how much wage grows over time
def plot_early_vs_growth(df, filtered_df):
  st.subheader("Early Career Salary vs. Wage Growth")
  fig, ax = plt.subplots()

  ax.scatter(df["Wage_Early"], df["Wage_Growth"], alpha=0.6, label="All Majors", color="gray")

  ax.scatter(filtered_df["Wage_Early"], filtered_df["Wage_Growth"], color='red', label="Filtered Majors")

  ax.set_xlabel("Early Career Salary ($)")
  ax.set_ylabel("Wage Growth (Mid - Early Career)")
  ax.set_title("Early Career Salary vs. Wage Growth")
  ax.legend()
  st.pyplot(fig)

# Visualizing how well the linear regression model performs
def plot_lin_reg_predictions(y_test, predictions):

  st.subheader("Linear Regression: Actual vs Predicted Mid-Career Salaries")

  fig, ax = plt.subplots()
  ax.scatter(y_test, predictions, alpha=0.6)
  ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
  ax.set_xlabel("Actuial Mid-Career Salary")
  ax.set_ylabel("Predicted Mid-Career Salary")
  ax.set_ylabel("Actual vs Predicted (Linear Regression)")
  st.pyplot(fig)

# Visualizing how well the Random Forest model performs

def plot_rand_for_predictions(y_test, predictions):

  st.subheader("Random Forest: Actual vs Predicted Mid-Career Salaries")

  fig, ax = plt.subplots()
  ax.scatter(y_test, predictions, alpha=0.6, color='green')
  ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
  ax.set_xlabel("Actual Mid-Career Salary")
  ax.set_ylabel("Predicted Salary (RF)")
  ax.set_title("Actual vs Predicted (Random Forest Model)")
  st.pyplot(fig)

# show which features (columns in df) the Random Forest Model Used the Most

def show_features_importance(important_features):
  st.subheader("Feature Importances (Random Forest)")
  st.bar_chart(important_features)