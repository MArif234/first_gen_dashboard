# Learn About College Degree Outcomes

## Goal of this Project

Many first-generation college students and college students with immigrant parents feel pressured to pursue prestigious career like law or medicine without completely understanding alternative career choices. This dashboard aims to **demystify career prospects based on college majors** by exploring employment rates, wages, and growth opportunties. The goal is to make students feel more empowered and to make more informed, authentic career choices.

## Dataset Used

I used a simple dataset from the Federal Reserve Bank of New York titled "Labor Market Outcomes of College Graduates by Major". I understand that this is a broad dataset and mostly focuses on popular majors, and that what a person majors in is only one factor in determining how their career progresses (or in some cases, not a factor at all). However, the purpose of this project is to encourage college students to be more open minded about their career choices and debunk myths about the success of certain careers. The dataset can be found at this link: [https://www.newyorkfed.org/research/college-labor-market#--:explore:outcomes-by-major](https://www.newyorkfed.org/research/college-labor-market#--:explore:outcomes-by-major)

## Live Dashboard

Check out the interactive dashboard here:

[https://firstgendashboard-9gbpg92vwpvi9fuxtm5qfd.streamlit.app/](https://firstgendashboard-9gbpg92vwpvi9fuxtm5qfd.streamlit.app/)

## Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- Scikit-Learn

## Project Overview

This project looks into the outcomes of college majors by exploring the following factors:
- Unemployment and underemployment rates
- Median wages (early and mid-career)
- Wage growth potential
- Correlation between graduate degree attainment and mid-career salary

### Features

- Filtering majors by unemployment rate and wage thresholds
- Identifying majors with high early salaries but low wage growth
- Visualize relationships with scatter plots
- Predict mid-career salaries using two machine learning models and compare their performance
- Display feature importance to understand which factors influence salary predictions most

## Detailed Research Questions and Methods

1. What are the top 10 majors with the lowest unemployment, underemployment, or highest wage?
2. What majors show the highest wage growth from early career to mid-career?
3. Is there a correlation between graduate degree share and mid-career wage?
4. What majors combine low unemployment with high mid-career wages?
5. Which majors have high early-career salaries but low wage growth?
6. Can we predict mid-career salaries using early-career data and unemployment metrics?


