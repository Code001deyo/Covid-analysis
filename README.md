# COVID-19 Data Analysis and Prediction

![COVID-19 Analysis](https://img.shields.io/badge/Project-COVID--19%20Data%20Analysis-blue) 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive workflow for exploring, cleaning, visualizing, and modeling COVID-19 datasets. This project focuses on detecting correlations between symptoms, medical conditions, and test results while addressing imbalanced data using machine learning techniques.

## Key Features

### Data Processing
- Inspection of dataset dimensions, column types, and null values
- Mapping boolean symptom values (True/False) to numerical representations (1/0)
- Target encoding (Negative: 0, Positive: 1, Other: 2)

### Visualization
- Histograms for numerical feature distributions
- Heatmaps for correlation analysis (symptoms, demographics, vital signs)
- Test result distribution plots
- Feature importance visualization

### Machine Learning
- Train/test splitting with stratified sampling
- Median imputation for missing values
- Handling imbalanced data with SMOTE
- Predictive models:
  - Random Forest
  - Gradient Boosting
  - Decision Trees

### Evaluation
- Classification metrics: accuracy, precision, recall, F1-score
- ROC curves and precision-recall curves

## Setup Instructions

1. **Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/COVID19-DataAnalysis.git
cd COVID19-DataAnalysis
