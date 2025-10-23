# src/run.py
import os
from sklearn.model_selection import train_test_split
from data_utils import load_data, preprocess_data
from model import train_model, evaluate_model, save_model


def main():
    # Paths
    train_path = os.path.join("..", "data", "train.csv")
    test_path = os.path.join("..", "data", "test.csv")

    # 1. Cargar datos
    train_df, test_df = load_data(train_path, test_path)

    # 2. Preprocesar
    X = preprocess_data(train_df.drop("Survived", axis=1))
    y = train_df["Survived"]

    # 3. División train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Entrenar modelo
    model = train_model(X_train, y_train)

    # 5. Evaluar
    acc = evaluate_model(model, X_val, y_val)
    print(f"Precisión en validación: {acc:.3f}")

    # 6. Guardar modelo
    save_model(model)


if __name__ == "__main__":
    main()

