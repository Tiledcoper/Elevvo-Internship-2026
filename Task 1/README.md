# Student Score Prediction

A regression project that predicts students' **Performance Index** using study habits and related academic factors.

## Project Overview

This notebook covers:

- Data loading and understanding
- Data cleaning and quality checks
- Exploratory data visualization
- Train/test split
- Linear Regression using `Hours Studied` as the required baseline
- Regression evaluation using R², MAE, RMSE, and MAPE
- Actual vs. Predicted visualization
- Polynomial Regression (bonus)
- Feature-combination experiments (bonus)
- Residual analysis
- Final conclusions

## Dataset

**Recommended dataset:** Student Performance Factors (Kaggle)

The notebook expects the dataset file to be named:

`StudentPerformance.csv`

and placed in the same folder as the notebook.

## Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Main Result

The final Linear Regression model using all available features achieved approximately:

- **R²:** 0.988
- **MAE:** 1.647
- **RMSE:** 2.075
- **MAPE:** 3.51%

The notebook also compares the baseline using `Hours Studied` alone with additional feature combinations.

## How to Run

1. Download/clone this repository.
2. Put `StudentPerformance.csv` beside the notebook.
3. Open `Student_Score_Prediction.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
4. Run the cells from top to bottom.

## Project Structure

```text
Student-Score-Prediction/
├── Student_Score_Prediction.ipynb
├── requirements.txt
└── README.md
```
