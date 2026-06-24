"""chromabench -- a minimal validation for chromatic (colour-aware) topological data analysis.

A small, controlled family of labelled two-colour point clouds for checking
whether a method actually uses its colours. The four classes factor as spatial
(uniform | cluster) x colour (mixed | separated) distribution; persistent
homology of the pooled points is colour-blind by construction, so it recovers
the spatial layout but not the colouring. A method that genuinely sees colour
should tell the colouring apart as well. It is the simplest sanity check to run
before trusting a chromatic invariant on real data.

Quick start::

    from chromabench import load_dataset, run_baseline

    ds = load_dataset()
    print(run_baseline(ds)["score"])  # colour-blind reference; ~0.47
"""

from .simulator import CLASS_NAMES, simulate
from .dataset import Dataset, load_dataset, generate
from .baseline import PHBaseline, run_baseline
from .evaluation import Method, evaluate, score

__all__ = [
    "CLASS_NAMES",
    "simulate",
    "Dataset",
    "load_dataset",
    "generate",
    "Method",
    "evaluate",
    "score",
    "PHBaseline",
    "run_baseline",
]

__version__ = "0.1.0"
