import pandas as pd
import numpy as np
import json
import joblib
import yaml
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from datetime import datetime
import mlflow
import mlflow.sklearn
import os  # Added for directory creation

def load_params():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Warning: params.yaml not found, using default parameters")
        return {
            'tfidf': {
                'max_features': 10000,
                'ngram_range': [1, 2],
                'min_df': 2,
                'max_df': 0.95,
                'C': 1.0,
                'max_iter': 2000
            },
            'training': {
                'cv_folds': 3
            }
        }

def train_tfidf_model():
    # Load parameters from DVC params file
    params = load_params()
    tfidf_params = params['tfidf']
    training_params = params['training']
    
    # Ensure mlruns directory exists
    os.makedirs("mlruns", exist_ok=True)
    
    # Set up MLflow with corrected tracking URI
    mlflow.set_tracking_uri("./mlruns")  # Changed from "file://./mlruns"
    mlflow.set_experiment("callcenter-classification")
    
    print("Loading data...")
    # Load data
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Validation data: {len(val_df)} samples")
    print(f"Categories: {train_df['Topic_group'].nunique()}")
    
    # Show parameters being used
    print(f"Parameters: max_features={tfidf_params['max_features']}, "
          f"ngram_range={tfidf_params['ngram_range']}, C={tfidf_params['C']}")
    
    # Prepare data
    X_train, y_train = train_df['Document'], train_df['Topic_group']
    X_val, y_val = val_df['Document'], val_df['Topic_group']
    
    print("Creating TF-IDF + SVM pipeline...")
    # Create pipeline using DVC parameters
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=tfidf_params['max_features'],
            ngram_range=tuple(tfidf_params['ngram_range']),
            min_df=tfidf_params['min_df'],
            max_df=tfidf_params['max_df'],
            stop_words='english'
        )),
        ('svm', LinearSVC(
            C=tfidf_params['C'], 
            random_state=42, 
            max_iter=tfidf_params['max_iter']
        ))
    ])
    
    # Train with calibration using DVC parameters
    print("Training with probability calibration...")
    calibrated_clf = CalibratedClassifierCV(
        pipeline, 
        cv=training_params['cv_folds']
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"tfidf_C_{tfidf_params['C']}"):
        # Log parameters
        mlflow.log_params({
            **tfidf_params,
            **training_params,
            "model_type": "TFIDF+LinearSVC"
        })
        
        # Train model
        start_time = datetime.now()
        print("Training model... (this may take a few minutes)")
        calibrated_clf.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        print("Making predictions...")
        # Make predictions
        y_pred = calibrated_clf.predict(X_val)
        y_prob = calibrated_clf.predict_proba(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        f1_macro = f1_score(y_val, y_pred, average='macro')
        
        # Get classification report
        class_report = classification_report(y_val, y_pred, output_dict=True)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("training_time", training_time)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(calibrated_clf, "tfidf_model")
        
        # Log classification report as an artifact
        with open('models/tfidf/classification_report.json', 'w') as f:
            json.dump(class_report, f, indent=2)
        mlflow.log_artifact('models/tfidf/classification_report.json')
        
        # Create model directory
        Path("models/tfidf").mkdir(parents=True, exist_ok=True)
        
        print("Saving model and metrics...")
        # Save model
        joblib.dump(calibrated_clf, 'models/tfidf/model.pkl')
        
        # Save main metrics for DVC tracking
        with open('models/tfidf/metrics.json', 'w') as f:
            json.dump({
                "accuracy": float(accuracy),
                "f1_weighted": float(f1_weighted),
                "f1_macro": float(f1_macro),
                "training_time": float(training_time)
            }, f, indent=2)
        mlflow.log_artifact('models/tfidf/metrics.json')
        
        print("\nTF-IDF model training completed!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score (weighted): {f1_weighted:.4f}")
        print(f"F1-score (macro): {f1_macro:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Model saved to: models/tfidf/model.pkl")
        
        # Show per-class performance
        print("\nPer-class F1 scores:")
        for category in sorted(class_report.keys()):
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                f1_class = class_report[category]['f1-score']
                print(f"  {category}: {f1_class:.3f}")
                mlflow.log_metric(f"f1_{category}", f1_class)
        
        # Test with examples
        test_examples()
    
    return calibrated_clf, {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "training_time": training_time
    }

def test_examples():
    """Test the trained model with a few examples"""
    try:
        print("\nTesting trained model...")
        model = joblib.load('models/tfidf/model.pkl')
        
        # Test examples
        test_cases = [
            "My laptop screen is broken and I can't see anything",
            "I forgot my password and cannot access my account", 
            "I need to purchase new office supplies for my team",
            "The server storage is full and I can't save files",
            "I need administrative rights to install software",
            "HR needs help with employee onboarding process"
        ]
        
        predictions = model.predict(test_cases)
        probabilities = model.predict_proba(test_cases)
        
        print("\nTest predictions:")
        for i, (text, pred) in enumerate(zip(test_cases, predictions)):
            max_prob = np.max(probabilities[i])
            print(f"{i+1}. '{text[:60]}...'")
            print(f"   → {pred} (confidence: {max_prob:.3f})")
            print()
            
    except Exception as e:
        print(f"Testing failed: {e}")

if __name__ == "__main__":
    model, metrics = train_tfidf_model()