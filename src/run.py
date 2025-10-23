# src/run.py
from data_utils import load_data, basic_info
from eda import (
    plot_survived_by_sex,
    plot_pclass_survival,
    plot_age_distribution,
    survival_by_class_and_sex,
)
import os

DATA_PATH = os.path.join("..", "data", "train.csv")


def main():
    df = load_data(DATA_PATH)
    df = df.dropna(subset=["Age", "Fare", "Sex", "Pclass", "Survived"])

    basic_info(df)
    plot_survived_by_sex(df)
    plot_age_distribution(df)
    plot_pclass_survival(df)
    survival_by_class_and_sex(df)


if __name__ == "__main__":
    main()
