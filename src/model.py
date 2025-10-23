# src/model.py
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model(X_train, y_train):
    """Entrena un modelo de regresión logística."""
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    """Evalúa el modelo y devuelve la precisión."""
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc


def save_model(model, path="outputs/models/model.pkl"):
    """Guarda el modelo entrenado."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
