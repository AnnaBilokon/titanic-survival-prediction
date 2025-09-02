from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


def make_logreg(random_state=42):
    return LogisticRegression(max_iter=1000, random_state=random_state)


def crossval_report(model, X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }
    res = cross_validate(model, X, y, cv=cv,
                         scoring=scoring, return_train_score=False)
    summary = {
        "acc_mean": res["test_accuracy"].mean(), "acc_std": res["test_accuracy"].std(),
        "auc_mean": res["test_roc_auc"].mean(), "auc_std": res["test_roc_auc"].std(),
    }
    return summary, res
