import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_dataset(file_path):
    """Load dataset and handle missing values."""
    try:
        data = pd.read_csv(file_path)
        data.dropna(inplace=True)
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        exit(1)

def preprocess_data(data, target_column='default'):
    """Split dataset into features and target, then apply scaling."""
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    logging.info("Model Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100))
    logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))

def predict_credit_risk(model, scaler, features):
    """Predict credit risk for a given input."""
    try:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        return "High Risk" if prediction[0] == 1 else "Low Risk"
    except Exception as e:
        logging.error("Error in prediction: %s", e)
        return None

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Credit Risk Prediction Model")
    parser.add_argument("file_path", help="Path to the dataset CSV file")
    args = parser.parse_args()

    logging.info("Loading dataset...")
    data = load_dataset(args.file_path)

    logging.info("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    logging.info("Training model...")
    model = train_model(X_train, y_train)

    logging.info("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Example usage
    example_features = [30, 50000, 10000]  # Modify as per your dataset
    result = predict_credit_risk(model, scaler, example_features)
    logging.info("Predicted Credit Risk: %s", result)

if __name__ == "__main__":
    main()
