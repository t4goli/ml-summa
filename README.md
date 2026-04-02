# Student Exam Score Prediction

## Overview

This project predicts student exam scores using machine learning models. The goal is to understand how different factors such as study time, sleep, attendance, and previous performance affect exam outcomes.

## Dataset

The dataset includes:

- Hours studied
- Sleep hours
- Attendance percentage
- Previous exam scores

Target variable:

- Exam score

## Approach

1. Loaded and explored dataset using pandas
2. Cleaned data and selected relevant features
3. Split data into training and testing sets (80/20)
4. Applied feature scaling using StandardScaler
5. Trained multiple models:
   - Linear Regression
   - Decision Tree
   - Random Forest

6. Evaluated models using Mean Squared Error (MSE)

## Results

- Linear Regression: MSE ≈ 7.76 (best)
- Random Forest: MSE ≈ 11
- Decision Tree: MSE ≈ 20+

## Key Insights

- Hours studied had the strongest impact on exam performance
- Simpler models performed better due to mostly linear relationships
- Decision trees overfit without constraints

## Tech Stack

- Python
- pandas
- scikit-learn

## Visualization

The model's performance was visualized by plotting predicted vs actual exam scores.

The plot shows a strong linear correlation, indicating that the model accurately captures the relationship between input features and exam performance.

Key Insights
Hours studied was the most influential feature (~65% importance in Random Forest)
Previous scores also contributed significantly to predictions
Linear Regression outperformed more complex models due to the dataset’s linear relationships
Decision Trees overfit the training data, which was mitigated by limiting tree depth
