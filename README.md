# CollabByte
# Credit Risk Predictor

## Overview
Credit Risk Predictor is a machine learning project that predicts whether a loan applicant is a high-risk or low-risk borrower using a Random Forest Classifier.

## Features
- Data Preprocessing (Handling missing values, scaling features)
- Train-Test Splitting
- Random Forest Classification
- Accuracy and Classification Report Evaluation
- Custom Prediction Function

## Installation
```sh
pip install pandas numpy scikit-learn
```

## Usage
Run the script in Python:
```sh
python credit_risk_predictor.py
```

## Example Prediction
```python
example_features = [30, 50000, 10000]  # age, income, loan_amount
print("Predicted Credit Risk:", predict_credit_risk(example_features))
```

## License
MIT License
