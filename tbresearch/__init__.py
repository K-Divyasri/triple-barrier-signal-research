"""tbresearch -- a leakage-free signal research pipeline.

Triple-barrier labeling (Lopez de Prado) turns a price series into
directional labels a classifier can learn from, and purged + embargoed
cross-validation stops those labels' overlapping time windows from leaking
information between train and test folds -- the two ideas this package
exists to make concrete.
"""

__version__ = "0.1.0"
