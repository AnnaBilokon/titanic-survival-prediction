import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score



def make_logreg(random_state=42):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state))

def make_knn(n_neighbors=5):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))

def make_decision_tree(max_depth=3, min_samples_leaf=20, random_state=42):
    return DecisionTreeClassifier(criterion="gini", max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, random_state=random_state)

def make_random_forest(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=-1)

def make_gradboost(random_state=42):
    return GradientBoostingClassifier(random_state=random_state)



def get_model_zoo(random_state=42):
    return {
        "LogReg": make_logreg(random_state=random_state),
        "KNN(k=5)": make_knn(n_neighbors=5),
        "DecisionTree": make_decision_tree(max_depth=3, min_samples_leaf=20, random_state=random_state),
        "RandomForest": make_random_forest(random_state=random_state),
        "GradBoost": make_gradboost(random_state=random_state),
    }



def crossval_report(model, X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    summary = {
        "accuracy_mean": res["test_accuracy"].mean(), "accuracy_std": res["test_accuracy"].std(),
        "roc_auc_mean": res["test_roc_auc"].mean(), "roc_auc_std": res["test_roc_auc"].std(),
        "precision_mean": res["test_precision"].mean(), "precision_std": res["test_precision"].std(),
        "recall_mean": res["test_recall"].mean(), "recall_std": res["test_recall"].std(),
        "f1_mean": res["test_f1"].mean(), "f1_std": res["test_f1"].std(),
    }
    return pd.DataFrame([summary]), res



def evaluate_models(models, X, y, n_splits=5, random_state=42):
    leaderboard = []
    raw_results = {}
    for name, model in models.items():
        summary_df, res = crossval_report(model, X, y, n_splits=n_splits, random_state=random_state)
        summary_df.insert(0, "model", name)
        leaderboard.append(summary_df)
        raw_results[name] = res
    lb = pd.concat(leaderboard, ignore_index=True).sort_values("roc_auc_mean", ascending=False)
    return lb, raw_results



def holdout_evaluate(model, X, y, test_size=0.2, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_te)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    else:
        y_prob = None

    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_te, y_prob))

    return model, metrics
