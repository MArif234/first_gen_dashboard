import pandas as pd
import streamlit as st

# streamlit feature that remember the function
# makes app faster bc don't have to reload data
@st.cache_data

def load_salary_data(file_path):
  df = pd.read_csv(file_path)
  df.rename(columns={
    "Median Wage Early Career": "Wage_Early",
    "Median Wage Mid-Career": "Wage_Mid",
    "Unemployment Rate": "Unemployment",
    "Underemployment Rate": "Underemployment",
    "Share with Graduate Degree": "Grad_Degree_Share"
  }, inplace=True)
  # regex=True looks at phrase simulataneously and then takes out all $ signs and commas
  df["Wage_Early"] = pd.to_numeric(df["Wage_Early"].replace(r'[\$,]', '', regex=True), errors="coerce")

  df["Wage_Mid"] = pd.to_numeric(df["Wage_Mid"].replace(r'[\$,]', '', regex=True), errors="coerce")

  df["Wage_Growth"] = df["Wage_Mid"] - df["Wage_Early"]

  return df
