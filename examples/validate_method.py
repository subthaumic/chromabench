"""Template for validating a chromatic method against chromabench.

A method is any object with::

    fit(samples, y)          # learn from training point clouds
    predict(samples) -> y    # label new point clouds

where ``samples`` is a list of point clouds, each a dict with ``coords`` of
shape ``(n, 2)`` and ``colours`` of shape ``(n,)``. A method that means to use
colour must read *both* -- a colour-blind method cannot rise above the
reference.

Pass your method to ``evaluate`` and compare against ``run_baseline``. For a
complete, runnable example of this interface, read ``chromabench.PHBaseline``
(the colour-blind PH baseline -- it is implemented exactly as a method here).
"""

from chromabench import evaluate, load_dataset, run_baseline


class MyChromaticMethod:
    """Replace the bodies with a colour-aware topological pipeline -- for
    example one built on chromatic alpha complexes -- followed by a classifier."""

    def fit(self, samples, y):
        # TODO: extract colour-aware features from (coords, colours) and learn.
        raise NotImplementedError

    def predict(self, samples):
        # TODO: return an array of predicted class indices.
        raise NotImplementedError


if __name__ == "__main__":
    ds = load_dataset()

    reference = run_baseline(ds)
    print(f"colour-blind reference accuracy: {reference['score']['balanced_accuracy']:.3f}")

    # result = evaluate(MyChromaticMethod(), ds)
    # print(f"my method accuracy: {result['score']['balanced_accuracy']:.3f}")
