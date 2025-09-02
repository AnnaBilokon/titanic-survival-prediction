import argparse
import pandas as pd
from features import add_rowwise_features, fit_train_stats, apply_stats, prepare_matrix
from pipeline import make_logreg, crossval_report


def main(args):
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    train = add_rowwise_features(train)
    test = add_rowwise_features(test)

    stats = fit_train_stats(train)
    train = apply_stats(train, stats)
    test = apply_stats(test, stats)

    y = train["Survived"].astype(int)
    X = prepare_matrix(train, drop_target=True)
    X_test = prepare_matrix(test, drop_target=False)

    model = make_logreg()
    summary, _ = crossval_report(model, X, y, n_splits=5)
    print(f"CV Accuracy: {summary['acc_mean']:.3f} ± {summary['acc_std']:.3f}")
    print(f"CV ROC-AUC:  {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")

    model.fit(X, y)
    pred = model.predict(X_test)
    out = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred})
    out.to_csv(args.out, index=False)
    print(f"Saved submission to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", default="outputs/submission.csv")
    args = p.parse_args()
    main(args)
