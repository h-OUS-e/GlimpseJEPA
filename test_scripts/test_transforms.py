"""Visual + sanity-check script for ``GlimpseTransform``.

Run:
    python test_scripts/test_transforms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow running from repo root or from inside test_scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms as T

from transforms import Action, GlimpseTransform


def load_mnist_batch(n: int = 8) -> torch.Tensor:
    """Load the first ``n`` MNIST training images as a (n, 1, 28, 28) tensor."""
    ds = datasets.MNIST(REPO_ROOT / "dataset", train=True, transform=T.ToTensor())
    return torch.stack([ds[i][0] for i in range(n)])


def plot_three_grids(
    original: torch.Tensor,
    initialized: torch.Tensor,
    transformed: torch.Tensor,
    titles: tuple[str, str, str] = ("Original", "Initialized", "Transformed"),
    save_path: Path | None = None,
) -> None:
    """Plot three batches side-by-side as rows of a single grid.

    Each input is expected to be ``(B, 1, H, W)`` — one row per batch, columns
    are the samples.

    Args:
        original: Source images.
        initialized: Views after a random initial action.
        transformed: Views after an additional delta on top of the init.
        titles: Row labels.
        save_path: If given, save the figure to this path; otherwise plt.show().
    """
    batches = (original, initialized, transformed)
    B = original.shape[0]

    fig, axes = plt.subplots(3, B, figsize=(B * 1.4, 4.2))
    if B == 1:
        axes = axes.reshape(3, 1)

    for row, (batch, title) in enumerate(zip(batches, titles)):
        for col in range(B):
            ax = axes[row, col]
            # detach for safety; squeeze single-channel dim
            ax.imshow(batch[col, 0].detach().cpu(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=11)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"saved figure to {save_path}")
    else:
        plt.show()


def run_assertions(x: torch.Tensor) -> None:
    """Quick correctness checks — same set as the verification block in the plan."""
    B = x.shape[0]
    g = GlimpseTransform()

    g.set_batch(x)
    assert torch.allclose(g.transform(Action()), x, atol=1e-5), "identity failed"

    g.set_batch(x)
    g.transform(Action(zoom=0.1))
    chained = g.transform(Action(zoom=0.1))
    g.set_batch(x)
    direct = g.transform(Action(zoom=0.2))
    assert torch.allclose(chained, direct, atol=1e-5), "cumulative zoom failed"

    g.set_batch(x)
    g.transform(Action(tx=0.3))
    back = g.transform(Action(tx=-0.3))
    assert torch.allclose(back, x, atol=1e-5), "translate roundtrip failed"

    g.set_batch(x)
    assert g.transform(Action(zoom=-1.0)).shape == x.shape, "zoom-out shape failed"

    g2 = GlimpseTransform()
    init = g2.initialize_batch(x)
    assert init.zoom.shape == (B,), "initialize_batch shape failed"

    print("all assertions pass")


def main() -> None:
    torch.manual_seed(0)

    x = load_mnist_batch(n=8)

    run_assertions(x)

    # build the three grids the user asked for: original, post-init, post-delta
    g = GlimpseTransform()
    g.set_batch(x)

    # 1. original — render with identity action so it goes through the same pipe
    original_view = g.transform(Action())

    # 2. initialized — random initial action sampled from init_bounds
    g.initialize_batch(x)
    initialized_view = g.transform(Action())

    # 3. transformed — apply a uniform delta on top of the init state
    delta = Action(zoom=0.4, tx=0.2, ty=-0.2)
    transformed_view = g.transform(delta)

    save_path = REPO_ROOT / "test_scripts" / "transforms_preview.png"
    plot_three_grids(original_view, initialized_view, transformed_view, save_path=save_path)


if __name__ == "__main__":
    main()
