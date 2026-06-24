"""Build the dataset, run the colour-blind PH reference, print where it lands."""

from chromabench import load_dataset, run_baseline

ds = load_dataset()
print(f"{len(ds.samples)} point clouds in {len(ds.class_names)} classes: {ds.class_names}")

result = run_baseline(ds)
s = result["score"]
print(f"\ncolour-blind PH reference (persistence images + linear readout, {result['n_splits']}-fold CV):")
print(f"  accuracy : {s['balanced_accuracy']:.3f}")
print("\nColour-blind PH plateaus here -- a method that genuinely uses colour")
print("should rise clearly above it.")
