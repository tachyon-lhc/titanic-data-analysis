# src/data_utils.py
"""
Carga y utilidades básicas para Titanic.
"""

import pandas as pd


def load_data(train_path: str, test_path: str):
    """Carga los datasets de entrenamiento y prueba."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def basic_info(df):
    """Imprime info útil: shape, columnas nulas y tipos."""
    print("Shape: ", df.shape)
    print("\n Info(): ")
    print(df.info())
    print("\n Nulos por columna: ")
    print(df.isnull().sum())


def preprocess_data(df: pd.DataFrame, is_train: bool = True):
    """
    Limpieza y codificación básica de los datos.
    - Rellena valores nulos
    - Codifica variables categóricas
    - Elimina columnas irrelevantes
    """
    df = df.copy()
    # Elimina columnas con poca utiidad
    df.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True, errors="ignore")

    # Rellenar edades y embarque con la moda/media
    df.fillna(df["Age"].median(), inplace=True)
    df.fillna(df["Embarked"].mode()[0], inplace=True)

    # Codificación simple de sexo y embarque
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Si es test, no tiene "Survived"
    if "Survived" in df.columns and not is_train:
        df.drop("Survived", axis=1, inplace=True)

    return df
