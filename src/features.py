import numpy as np
import pandas as pd


def add_rowwise_features(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    out_df["FamilySize"] = out_df["SibSp"] + out_df["Parch"] + 1
    out_df["IsAlone"] = (out_df["FamilySize"] == 1).astype(int)
    out_df["HasCabin"] = out_df["Cabin"].notna().astype(int)
    out_df["Title"] = out_df["Name"].str.extract(
        r" ([A-Za-z]+)\.", expand=False)
    out_df["LogFare"] = np.log1p(out_df["Fare"])
    return out_df


def fit_train_stats(train_df: pd.DataFrame) -> dict:
    stats = {
        "embarked_mode": train_df["Embarked"].mode()[0],
        "fare_median": float(train_df["Fare"].median()),
        "age_global_median": float(train_df["Age"].median()),
        "age_medians_by_title": train_df.groupby("Title")["Age"].median().to_dict(),
    }
    return stats


def apply_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out_df = df.copy()
    out_df["Embarked"] = out_df["Embarked"].fillna(stats["embarked_mode"])
    out_df["Fare"] = out_df["Fare"].fillna(stats["fare_median"])
    out_df["LogFare"] = np.log1p(out_df["Fare"])

    def _fill_age(row):
        if pd.isna(row["Age"]):
            return stats["age_medians_by_title"].get(row["Title"], stats["age_global_median"])
        return row["Age"]
    out_df["Age"] = out_df.apply(_fill_age, axis=1)
    return out_df


def add_age_category(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
 
    bins = [-np.inf, 12, 59, np.inf]
    labels = ["Child", "Adult", "Senior"]
    out_df["Age_category"] = pd.cut(out_df["Age"], bins=bins, labels=labels)


    out_df["Age_category"] = out_df["Age_category"].astype("category")
    if "Unknown" not in out_df["Age_category"].cat.categories:
        out_df["Age_category"] = out_df["Age_category"].cat.add_categories("Unknown")
    out_df["Age_category"] = out_df["Age_category"].fillna("Unknown")

    return out_df


def prepare_matrix(df: pd.DataFrame, drop_target: bool = False) -> pd.DataFrame:

    X = df.copy()
    if X["Sex"].dtype == "object":
        X["Sex"] = X["Sex"].map({"male": 0, "female": 1}).astype(int)

    onehot_cols = ["Pclass", "Embarked", "Title"]
    if "Age_category" in X.columns:
        onehot_cols.append("Age_category")

    X = pd.get_dummies(X, columns=onehot_cols, prefix=onehot_cols)

    bool_cols = X.select_dtypes(include=bool).columns
    X[bool_cols] = X[bool_cols].astype(int)
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    if drop_target:
        drop_cols += ["Survived"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    return X
