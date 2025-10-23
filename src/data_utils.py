# src/data_utils.py
"""
Carga y utilidades básicas para Titanic.
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Carga los datos del Titanic desde un archivo CSV."""
    data = pd.read_csv(path)
    return data


def basic_info(df):
    """Imprime info útil: shape, columnas nulas y tipos."""
    print("Shape: ", df.shape)
    print("\n Info(): ")
    print(df.info())
    print("\n Nulos por columna: ")
    print(df.isnull().sum())
