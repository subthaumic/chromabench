"""Render the README figures into ``assets/``.

Produces:

- one representative point cloud per class (``<class>.png``), the first dataset
  realization of each class under the frozen seed, so they are genuine samples;
- the confusion matrix of the colour-blind PH reference (``ph_confusion.png``),
  which shows what that reference can and cannot tell apart.

Regenerate with::

    uv run python scripts/render_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from chromabench import load_dataset, run_baseline  # noqa: E402
from chromabench.simulator import CLASS_NAMES  # noqa: E402

COLOUR_A_HEX = "#006BA4"
COLOUR_B_HEX = "#FF800E"
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def render_class_panels(ds) -> None:
    for class_idx, name in enumerate(CLASS_NAMES):
        i = int(np.flatnonzero(ds.y == class_idx)[0])
        coords = ds.samples[i]["coords"]
        colours = ds.samples[i]["colours"]

        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        for c_val, hex_colour in [(0, COLOUR_A_HEX), (1, COLOUR_B_HEX)]:
            mask = colours == c_val
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=12, c=hex_colour, edgecolors="white", linewidths=0.3, alpha=0.9,
            )
        ax.set_aspect("equal")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        fig.tight_layout(pad=0.2)
        out = ASSETS / f"{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def render_ph_confusion(ds) -> None:
    result = run_baseline(ds)
    cm = confusion_matrix(result["y_true"], result["y_pred"], labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                color="white" if cm_norm[i, j] < 0.5 else "black", fontsize=8,
            )
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = ASSETS / "ph_confusion.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    ds = load_dataset()
    render_class_panels(ds)
    render_ph_confusion(ds)


if __name__ == "__main__":
    main()
