"""Purged, embargoed K-Fold cross-validation (Lopez de Prado, chapter 7).

Standard K-Fold CV assumes samples are independent. Triple-barrier labels
aren't: each one is built from a window of future prices, so a training
sample whose window overlaps a test sample's window has effectively "seen"
part of the test answer. PurgedKFold removes those training samples
(purging) and a further short window right after the test fold (embargo,
guarding against serial correlation carrying information forward). The
payoff is a CV score that's pessimistic but honest, instead of optimistic
and wrong.
"""

import numpy as np
import pandas as pd


class PurgedKFold:
    """A K-Fold splitter for data with overlapping, interval-valued labels.

    `t1` must be a pandas Series aligned 1:1 (same length, same order) with
    the feature matrix rows: index = event start time (t0), value = event
    end time (t1, the triple-barrier touch time). `pct_embargo` is the
    fraction of the total sample count embargoed immediately after each
    test fold.
    """

    def __init__(self, n_splits: int, t1: pd.Series, pct_embargo: float = 0.0):
        self.n_splits = n_splits
        self.t1 = t1.reset_index(drop=True) if not isinstance(t1.index, pd.RangeIndex) else t1
        self._t1_values = t1.to_numpy()
        self._t0_values = t1.index.to_numpy()
        self.pct_embargo = pct_embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        if n != len(self._t1_values):
            raise ValueError("X and t1 must have the same length and the same row order")

        indices = np.arange(n)
        embargo = int(n * self.pct_embargo)
        test_folds = np.array_split(indices, self.n_splits)

        for test_idx in test_folds:
            test_start, test_end = test_idx[0], test_idx[-1]
            test_window_start = self._t0_values[test_start]
            test_window_end = self._t1_values[test_idx].max()

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start : test_end + 1] = False  # the test fold itself

            sample_t0 = self._t0_values
            sample_t1 = self._t1_values
            overlaps_test = (sample_t0 <= test_window_end) & (sample_t1 >= test_window_start)
            train_mask &= ~overlaps_test

            if embargo > 0:
                embargo_start, embargo_end = test_end + 1, min(test_end + embargo, n - 1)
                if embargo_start <= embargo_end:
                    train_mask[embargo_start : embargo_end + 1] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx


def purge_fraction(n_splits: int, t1: pd.Series, pct_embargo: float = 0.0) -> float:
    """What fraction of all samples get purged/embargoed out of training,
    averaged across folds -- a quick, honest number for how much overlap
    labeling is costing you before you even get to model performance."""
    n = len(t1)
    cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    dummy_X = np.zeros((n, 1))
    dropped_fracs = []
    for train_idx, test_idx in cv.split(dummy_X):
        available_for_training = n - len(test_idx)
        dropped = available_for_training - len(train_idx)
        dropped_fracs.append(dropped / available_for_training if available_for_training else 0.0)
    return float(np.mean(dropped_fracs))
