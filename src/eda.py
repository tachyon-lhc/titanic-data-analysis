# src/eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ------------------------------------------------------------


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("Saved: ", path)


def save_report(df, filename):
    path = os.path.join(REPORTS_DIR, filename)
    df.to_csv(path, index=True)
    print("Report saved in: ", path)


# -----------------------------------------------------------


def plot_survived_by_sex(df):
    counts = df.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)
    fig = counts.plot(kind="bar", stacked=False).get_figure()
    save_fig(fig, "survived_by_sex.png")


def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Age"].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Age distribution")
    save_fig(fig, "age_distribution.png")


def plot_pclass_survival(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Pclass", y="Survived", data=df, estimator=lambda x: sum(x) / len(x))
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate per class")
    save_fig(fig, "pclas_survival.png")


def survival_by_class_and_sex(df):
    pivot = df.pivot_table(
        values="Survived", index="Pclass", columns="Sex", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_title("Survival rate by class and sex")
    save_fig(fig, "survival_heatmap.png")
    save_report(pivot, "survival_by_class_and_sex.csv")
