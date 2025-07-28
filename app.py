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


# Loading up the data
df = load_salary_data("majors_data.csv")

# Title and Description of Project
st.title("Learn About College Degree Outcomes")
st.write("Explore majors in regards to wage, unemployment, and underemployment.")

# Creating dropdown menu for category of choice

# Question: What are the top 10 college majors with the lowest [metric]?

# the 4 metrics are Unemployment Rate, Underemployment Rate, Median Wage Early Career, Median Wage Mid-Career

metric_options = {
  "Unemployment Rate": "Unemployment",
  "Underemployment Rate": "Underemployment",
  "Median Wage Early Career": "Wage_Early",
  "Median Wage Mid-Career": "Wage_Mid"
}

choice = st.selectbox("Choose a category you would like to sort by:", list(metric_options.keys()))
metric = metric_options[choice]

# sorts df by column chosen in the metric
# sorts column from smallest to largest
sorted_df = df.sort_values(by=metric, ascending=True)

st.subheader(f"Top 10 majors by lowest {metric}")
st.table(sorted_df[["Major", metric]].head(10))



# What majors have the highest wage growth from early career to mid-career?


top_growth = df.sort_values("Wage_Growth", ascending=False).head(10)
st.subheader("Top Majors by Wage Growth")
st.table(top_growth[["Major", "Wage_Growth"]])

# Is there a correlation between percentage of people who have completed a graduate degree and mid-career wage?

st.subheader("Relationship Between Graduate Degree Share and Mid-Career Salary")

fig, ax = plt.subplots()
ax.scatter(df["Grad_Degree_Share"], df["Wage_Mid"], alpha=0.7)
ax.set_xlabel("Share with Graduate Degree")
ax.set_ylabel("Median Mid-Career Salary ($)")
ax.set_title("Grad Degree Share vs. Mid-Career Salary")

st.pyplot(fig)

correlation = df["Grad_Degree_Share"].corr(df["Wage_Mid"])
st.write(f"Correlation between Grad Degree Share and Mid-Career Wage: {correlation:.2f}")

low_return = df[(df["Grad_Degree_Share"] > 0.5) & (df["Wage_Mid"] < 70000)]
st.subheader("Major with High Grad Share but Low Mid-Career Wages")
st.table(low_return[["Major", "Grad_Degree_Share", "Wage_Mid"]])

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

fig, ax = plt.subplots()
ax.scatter(df["Wage_Early"], df["Wage_Growth"], alpha=0.6, label="All Majors", color="gray")

# Highlight Selected Majors

ax.scatter(low_growth_high_early["Wage_Early"], low_growth_high_early["Wage_Growth"], color='red', label="Filtered Majors")

ax.set_xlabel("Early Career Salary ($)")
ax.set_ylabel("Wage Growth (Mid - Early Career)")
ax.set_title("Early Career Salary vs. Wage Growth")
ax.legend()

st.pyplot(fig)

# Goal: Predict Mid-Career Salary for College Majors


st.header("Predicting Mid-Career Salary Using Early Career Data")

st.write("""
This dashboard uses **machine learning models** to predict the median mid-career salary for different college majors. We use features like early-career salary, graduate degree share, unemployment, and underemployment to train: 

- A **Linear Regression** model
- A **Random Forest** model

Then we compare their performance and explore which features were the most influential. 
         """)



# input variables
X = df[["Wage_Early", "Grad_Degree_Share", "Unemployment", "Underemployment"]]

# variable I want to predict
y = df["Wage_Mid"]

# split up data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("Linear Regression Model Results")
# Training the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)


# creating a DataFrame showing actual vs predicted

results_df = pd.DataFrame({
  "Major": df.loc[y_test.index, "Major"],
  "Actual Mid-Career Salary": y_test,
  "Predicted Mid-Career Salary": lr_predictions
}) 

st.subheader("Actual vs Predicted Mid-Career Salary (Linear Regression)")
st.dataframe(results_df.reset_index(drop=True))

fig, ax = plt.subplots()
ax.scatter(y_test, lr_predictions, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') 
ax.set_xlabel("Actual Mid-Career Salary")
ax.set_ylabel("Predicted Mid-Career Salary")
ax.set_title("Linear Regression: Actual vs Predicted Mid-Career Salaries")
st.pyplot(fig)

lr_score = r2_score(y_test, lr_predictions)
st.write(f"Linear Regression R^2 Score: {lr_score:.2f}")

if lr_score > 0.7:
  st.write("This means the model does a good job explaining differences in mid-career salaries based on the selected factors.")
elif lr_score > 0.4:
  st.write("The model has some predictive power, but it misses other factors that are likely playing a big role in mid-career salaries.")
else:
  st.write("The model doesn't explain much of the variation in mid-career salaries. It suggests the input variables aren't strong predictors on their own.")

# Random Forest Regression

st.subheader("Random Forest Regression Model Results")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_results_df = pd.DataFrame({
  "Major": df.loc[y_test.index, "Major"],
  "Actual Mid-Career Salary": y_test,
  "Predicted (Random Forest)": rf_predictions
})

# Display Table and Scatter Plot
st.markdown("** Table: Actual vs Predicted (Random Forest)**")
st.dataframe(rf_results_df.reset_index(drop=True))

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, rf_predictions, alpha=0.6, color='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax2.set_xlabel("Actual Mid-Career Salary")
ax2.set_ylabel("Predicted Salary (RF)")
ax2.set_title("Actual vs Predicted Mid-Career Salaries (Random Forest)")
st.pyplot(fig2)

# R^2 Score
rf_score = r2_score(y_test, rf_predictions)
st.markdown(f"**Random Forest R^2 Score:**  {rf_score:.2f}")

# Feature Importance
st.subheader("Feature Importance (Random Forest)")
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
st.bar_chart(importances)

# Model Comparison
st.subheader("Model Comparison")

st.markdown(f"""
- **Linear Regression R^2**: `{lr_score:.2f}`
- **Random Forest R^2**: `{rf_score:.2f}`
            """)

if rf_score > lr_score:
  st.write(f"Random Forst performed better than Linear Regression (RF: {rf_score:.2f}) vs LR: {lr_score:.2f}")
elif rf_score < lr_score:
  st.write(f"Linear Regression performed slightly better (LR: {lr_score:.2f} vs RF: {rf_score:.2f})")
else:
  st.write("Both models performed equally based on R^2 score.")