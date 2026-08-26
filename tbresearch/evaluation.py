"""The payoff notebook/module: naive K-Fold CV vs Purged+Embargoed K-Fold CV,
on the exact same data, same model, same everything else. The gap between
them IS the leakage naive CV was hiding.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

from .cross_validation import PurgedKFold


@dataclass
class FoldResult:
    train_accuracy: float
    test_accuracy: float
    test_f1: float
    n_train: int
    n_test: int


@dataclass
class CVReport:
    method: str
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_train_accuracy(self) -> float:
        return float(np.mean([f.train_accuracy for f in self.folds]))

    @property
    def mean_test_accuracy(self) -> float:
        return float(np.mean([f.test_accuracy for f in self.folds]))

    @property
    def mean_test_f1(self) -> float:
        return float(np.mean([f.test_f1 for f in self.folds]))

    @property
    def train_test_gap(self) -> float:
        """How much better the model looks on train than on held-out test --
        a large gap on data that SHOULD be leakage-free is a red flag; a
        naive CV showing a SMALL gap despite real leakage is the tell that
        it's actually measuring overlap, not genuine generalization."""
        return self.mean_train_accuracy - self.mean_test_accuracy


def _make_classifier(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=random_state)


def run_cv(X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series, cv_splitter, method_name: str, random_state: int = 42) -> CVReport:
    report = CVReport(method=method_name)
    X_arr, y_arr, w_arr = X.to_numpy(), y.to_numpy(), sample_weight.to_numpy()

    for train_idx, test_idx in cv_splitter.split(X_arr):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        clf = _make_classifier(random_state)
        clf.fit(X_arr[train_idx], y_arr[train_idx], sample_weight=w_arr[train_idx])

        train_pred = clf.predict(X_arr[train_idx])
        test_pred = clf.predict(X_arr[test_idx])

        report.folds.append(
            FoldResult(
                train_accuracy=accuracy_score(y_arr[train_idx], train_pred),
                test_accuracy=accuracy_score(y_arr[test_idx], test_pred),
                test_f1=f1_score(y_arr[test_idx], test_pred, average="macro", zero_division=0),
                n_train=len(train_idx),
                n_test=len(test_idx),
            )
        )
    return report


def compare_naive_vs_purged(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    t1: pd.Series,
    n_splits: int = 5,
    pct_embargo: float = 0.02,
    random_state: int = 42,
) -> dict[str, CVReport]:
    naive_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    naive_report = run_cv(X, y, sample_weight, naive_cv, "naive KFold (shuffled)", random_state)

    purged_cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    purged_report = run_cv(X, y, sample_weight, purged_cv, "purged + embargoed KFold", random_state)

    return {"naive": naive_report, "purged": purged_report}
