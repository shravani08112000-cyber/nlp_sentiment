# Classification Probelm in ML we have regression Logistic regression, SVM, NaiveBayes, RandomForest XG

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class SentimentModel:

    def __init__(self, model_type='logistic_regression'):

        self.model_type = model_type
        self.model = None
        self.history = {}

    def create_model(self):
        
        models = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'logistic_regression': LogisticRegression(
                max_iter = 1000,
                random_state=42,
                C=1.0
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=20
            ),
            'svm': LinearSVC(
                max_iter=1000,
                random_state=42,
                C=1.0
            )
        }
        self.model = models.get(self.model_type)
        if self.model is None:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):

        if self.model is None:
            self.create_model()

        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f})%")

        # Evaluation on validation data if provided

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)

            print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f})%")

            self.history['val_accuracy'] = val_acc
        
        self.history['train_accuracy'] = train_acc
        
        return self.model
    
    def evaluate(self, X_test, y_test):

        """Evcaluate model performamnce"""

        y_pred = self.model.predict(X_test)

        metrics = {
            'accuracy':accuracy_score(y_test, y_pred),
            'precision':precision_score(y_test,y_pred),
            'recall':recall_score(y_test,y_pred),
            "f1_score":f1_score(y_test,y_pred)
        }

        cm = confusion_matrix(y_test,y_pred)

        

        return metrics
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For SVM, convert decision function to probabilities
            from scipy.special import expit
            decision = self.model.decision_function(X)
            proba = expit(decision)
            return np.vstack([1-proba, proba]).T
        else:
            return None
    

if __name__ == "__main__":
    
    model = SentimentModel(model_type="logistic_regression")
    print(model)