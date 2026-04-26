"""Visual + sanity-check script for ``ActionGenerator``.

Run:
    "C:/Users/Ous/miniconda3/envs/ML/python.exe" test_scripts/test_action_generator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms as T

from glimpse import (
    Action,
    ActionGenerator,
    GlimpseTransform,
    RandomWalkGenerator,
    ReturnToOriginGenerator,
)


class _DummyGenerator(ActionGenerator):
    """Minimal concrete subclass for testing the base-class helpers."""

    def sample(self, B, device, dtype, generator=None):
        init = self._sample_init(B, device, dtype, generator)
        t_stop = self._sample_t_stop(B, device, generator)
        deltas = torch.zeros(B, self.T_max, 3, device=device, dtype=dtype)
        return init, deltas, t_stop


def test_base_helpers_shapes_and_dtypes():
    gen = _DummyGenerator(init_bounds=Action(zoom=0.5, tx=0.4, ty=0.4), T_max=6)
    B = 32
    init, deltas, t_stop = gen.sample(B, device=torch.device("cpu"), dtype=torch.float32)

    assert isinstance(init, Action), "init must be an Action"
    assert init.zoom.shape == (B,), f"init.zoom shape {init.zoom.shape}"
    assert init.tx.shape == (B,), f"init.tx shape {init.tx.shape}"
    assert init.ty.shape == (B,), f"init.ty shape {init.ty.shape}"
    assert init.zoom.dtype == torch.float32
    assert deltas.shape == (B, 6, 3), f"deltas shape {deltas.shape}"
    assert deltas.dtype == torch.float32
    assert t_stop.shape == (B,)
    assert t_stop.dtype == torch.long
    assert (t_stop >= 1).all() and (t_stop <= 6).all(), "t_stop out of range"
    print("[base helpers] shapes & dtypes OK")


def test_random_walk_shapes_bounds_and_padding():
    init_bounds = Action(zoom=0.4, tx=0.5, ty=0.5)
    step_bounds = Action(zoom=0.05, tx=0.1, ty=0.1)
    gen = RandomWalkGenerator(init_bounds=init_bounds, T_max=8, step_bounds=step_bounds)
    B = 256
    init, deltas, t_stop = gen.sample(B, torch.device("cpu"), torch.float32)

    # shapes
    assert deltas.shape == (B, 8, 3)
    assert t_stop.shape == (B,)
    assert t_stop.dtype == torch.long

    # t_stop within range and covers all values for a large enough B
    assert (t_stop >= 1).all() and (t_stop <= 8).all()
    seen = set(t_stop.unique().tolist())
    assert seen == set(range(1, 9)), f"missing t_stop values: {set(range(1, 9)) - seen}"

    # bounds: |delta_axis| <= step_bound for k < t_stop
    bounds_per_axis = torch.tensor(
        [float(step_bounds.zoom), float(step_bounds.tx), float(step_bounds.ty)]
    )  # (3,)
    k_idx = torch.arange(8).unsqueeze(0)  # (1, 8)
    active = k_idx < t_stop.unsqueeze(1)   # (B, 8) bool
    active_deltas = deltas[active]         # (N_active, 3)
    assert (active_deltas.abs() <= bounds_per_axis + 1e-6).all(), "bounds violated"

    # padding: deltas at k >= t_stop must be exactly zero
    padded = deltas[~active]
    assert (padded == 0).all(), "padding contains non-zero deltas"
    print("[random-walk] shapes, bounds, padding OK")


def test_random_walk_default_step_bounds_division():
    init_bounds = Action(zoom=1.0, tx=2.0, ty=3.0)
    gen = RandomWalkGenerator(init_bounds=init_bounds, T_max=10)  # no step_bounds
    assert float(gen.step_bounds.zoom) == 1.0 / 10
    assert float(gen.step_bounds.tx) == 2.0 / 10
    assert float(gen.step_bounds.ty) == 3.0 / 10
    print("[random-walk] default step_bounds = init_bounds / T_max OK")


def test_random_walk_reproducibility():
    gen = RandomWalkGenerator(init_bounds=Action(zoom=0.3, tx=0.4, ty=0.4), T_max=5)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = gen.sample(16, torch.device("cpu"), torch.float32, generator=g1)
    b = gen.sample(16, torch.device("cpu"), torch.float32, generator=g2)
    assert torch.equal(a[0].zoom, b[0].zoom)
    assert torch.equal(a[0].tx, b[0].tx)
    assert torch.equal(a[0].ty, b[0].ty)
    assert torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2])
    print("[random-walk] reproducibility OK")


def test_return_to_origin_lands_on_zero_and_constant_deltas():
    init_bounds = Action(zoom=0.3, tx=0.5, ty=0.5)
    gen = ReturnToOriginGenerator(init_bounds=init_bounds, T_max=7)
    B = 128
    init, deltas, t_stop = gen.sample(B, torch.device("cpu"), torch.float32)

    # shapes
    assert deltas.shape == (B, 7, 3)

    # 1) cumulative sum of deltas equals -init (per axis, per sample)
    delta_sum = deltas.sum(dim=1)  # (B, 3)
    init_stack = torch.stack([init.zoom, init.tx, init.ty], dim=-1)  # (B, 3)
    assert torch.allclose(delta_sum, -init_stack, atol=1e-5), "did not land on origin"

    # 2) for k < t_stop, all deltas[b, k] are identical across k (constant)
    for b in range(B):
        ts = int(t_stop[b].item())
        active = deltas[b, :ts]            # (ts, 3)
        first = active[0]
        assert torch.allclose(active, first.expand_as(active), atol=1e-6), (
            f"sample {b}: deltas not constant for k < t_stop={ts}"
        )

    # 3) padding: k >= t_stop must be zero
    k_idx = torch.arange(7).unsqueeze(0)
    padding_mask = k_idx >= t_stop.unsqueeze(1)  # (B, 7)
    assert (deltas[padding_mask] == 0).all(), "padding contains non-zero deltas"

    print("[return-to-origin] lands on origin, constant deltas, padding OK")


def test_return_to_origin_t_stop_one_edge_case():
    gen = ReturnToOriginGenerator(init_bounds=Action(zoom=0.5, tx=0.5, ty=0.5), T_max=4)

    # force t_stop = 1 by patching _sample_t_stop
    B = 16
    init = gen._sample_init(B, torch.device("cpu"), torch.float32, None)
    t_stop = torch.ones(B, dtype=torch.long)
    deltas = gen._build_deltas(init, t_stop, torch.device("cpu"), torch.float32)

    # delta[b, 0] = -init[b]; delta[b, 1:] = 0
    expected_first = -torch.stack([init.zoom, init.tx, init.ty], dim=-1)
    assert torch.allclose(deltas[:, 0], expected_first, atol=1e-6)
    assert (deltas[:, 1:] == 0).all()
    print("[return-to-origin] t_stop=1 edge case OK")


def test_glimpse_transform_set_state_installs_action():
    g = GlimpseTransform()
    x = torch.zeros(4, 1, 28, 28)
    g.set_batch(x)

    # known state, broadcast to (B,)
    init = Action(zoom=0.1, tx=0.2, ty=-0.3).to_batched(4, device=x.device, dtype=x.dtype)
    g.set_state(init)

    assert torch.allclose(g.state.zoom, init.zoom)
    assert torch.allclose(g.state.tx, init.tx)
    assert torch.allclose(g.state.ty, init.ty)
    print("[glimpse] set_state installs action OK")


def test_integration_return_to_origin_lands_on_origin_via_transform():
    """Apply each delta through GlimpseTransform; verify cumulative state."""
    init_bounds = Action(zoom=0.3, tx=0.4, ty=0.4)
    T_max = 5
    gen = ReturnToOriginGenerator(init_bounds=init_bounds, T_max=T_max)

    B = 16
    x = torch.zeros(B, 1, 28, 28)
    init, deltas, t_stop = gen.sample(B, x.device, x.dtype)

    g = GlimpseTransform()
    g.set_batch(x)
    g.set_state(init)

    for k in range(T_max):
        d = Action(zoom=deltas[:, k, 0], tx=deltas[:, k, 1], ty=deltas[:, k, 2])
        g.transform(d)

    # final cumulative state == 0 (each sample reached origin at t_stop and stayed)
    assert torch.allclose(g.state.zoom, torch.zeros(B), atol=1e-5)
    assert torch.allclose(g.state.tx,   torch.zeros(B), atol=1e-5)
    assert torch.allclose(g.state.ty,   torch.zeros(B), atol=1e-5)
    print("[integration] return-to-origin reaches origin via GlimpseTransform OK")


def test_integration_random_walk_state_matches_cumsum():
    gen = RandomWalkGenerator(
        init_bounds=Action(zoom=0.3, tx=0.4, ty=0.4),
        T_max=6,
        step_bounds=Action(zoom=0.05, tx=0.1, ty=0.1),
    )
    B = 8
    x = torch.zeros(B, 1, 28, 28)
    init, deltas, t_stop = gen.sample(B, x.device, x.dtype)

    g = GlimpseTransform()
    g.set_batch(x)
    g.set_state(init)
    for k in range(gen.T_max):
        d = Action(zoom=deltas[:, k, 0], tx=deltas[:, k, 1], ty=deltas[:, k, 2])
        g.transform(d)

    # final state should equal init + cumulative sum of deltas
    expected_zoom = init.zoom + deltas[..., 0].sum(dim=1)
    expected_tx   = init.tx   + deltas[..., 1].sum(dim=1)
    expected_ty   = init.ty   + deltas[..., 2].sum(dim=1)
    assert torch.allclose(g.state.zoom, expected_zoom, atol=1e-5)
    assert torch.allclose(g.state.tx,   expected_tx,   atol=1e-5)
    assert torch.allclose(g.state.ty,   expected_ty,   atol=1e-5)
    print("[integration] random-walk cumulative state matches cumsum OK")


def _load_mnist_batch(n: int) -> torch.Tensor:
    ds = datasets.MNIST(REPO_ROOT / "dataset", train=True, transform=T.ToTensor())
    return torch.stack([ds[i][0] for i in range(n)])


def _render_trajectory(
    gen: ActionGenerator,
    imgs: torch.Tensor,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a trajectory through GlimpseTransform.

    Returns:
        views: ``(B, T_max+1, 1, H, W)`` — view at each cumulative step.
        t_stop: ``(B,)`` long tensor.
    """
    B = imgs.shape[0]
    g = torch.Generator().manual_seed(seed)
    init, deltas, t_stop = gen.sample(B, imgs.device, imgs.dtype, generator=g)

    transform = GlimpseTransform()
    transform.set_batch(imgs)
    transform.set_state(init)

    views = [transform.transform(Action())]  # step 0: identity delta -> shows init state
    for k in range(gen.T_max):
        d = Action(zoom=deltas[:, k, 0], tx=deltas[:, k, 1], ty=deltas[:, k, 2])
        views.append(transform.transform(d))

    return torch.stack(views, dim=1), t_stop  # (B, T_max+1, 1, H, W)


def plot_trajectory_grid(
    views: torch.Tensor,
    t_stop: torch.Tensor,
    title: str,
    save_path: Path,
) -> None:
    """Plot a (B, T_max+1) grid of glimpse views; mark the ``t_stop`` step.

    Args:
        views: ``(B, T_max+1, 1, H, W)`` tensor.
        t_stop: ``(B,)`` long tensor; column ``t_stop[b]`` is highlighted on
            row ``b``.
    """
    B, T1 = views.shape[:2]
    fig, axes = plt.subplots(B, T1, figsize=(T1 * 1.1, B * 1.2))
    if B == 1:
        axes = axes.reshape(1, -1)

    for b in range(B):
        ts = int(t_stop[b].item())
        for k in range(T1):
            ax = axes[b, k]
            ax.imshow(views[b, k, 0].detach().cpu(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if b == 0:
                ax.set_title(f"k={k}", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"b={b}", fontsize=8)
            if k == ts:
                # red border to mark the t_stop step
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.5)
                ax.set_xlabel("t_stop", color="red", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"saved figure to {save_path}")
    plt.close(fig)


def visual_sanity_plots():
    imgs = _load_mnist_batch(n=4)
    init_bounds = Action(zoom=0.3, tx=0.5, ty=0.5)
    T_max = 6

    rw_gen = RandomWalkGenerator(init_bounds=init_bounds, T_max=T_max)
    rw_views, rw_tstop = _render_trajectory(rw_gen, imgs, seed=1)
    plot_trajectory_grid(
        rw_views, rw_tstop,
        title="RandomWalkGenerator — red border = t_stop",
        save_path=REPO_ROOT / "test_scripts" / "action_generator_random_walk.png",
    )

    rt_gen = ReturnToOriginGenerator(init_bounds=init_bounds, T_max=T_max)
    rt_views, rt_tstop = _render_trajectory(rt_gen, imgs, seed=2)
    plot_trajectory_grid(
        rt_views, rt_tstop,
        title="ReturnToOriginGenerator — red border = t_stop",
        save_path=REPO_ROOT / "test_scripts" / "action_generator_return_to_origin.png",
    )


def main():
    torch.manual_seed(0)
    test_base_helpers_shapes_and_dtypes()
    test_random_walk_shapes_bounds_and_padding()
    test_random_walk_default_step_bounds_division()
    test_random_walk_reproducibility()
    test_return_to_origin_lands_on_zero_and_constant_deltas()
    test_return_to_origin_t_stop_one_edge_case()
    test_glimpse_transform_set_state_installs_action()
    test_integration_return_to_origin_lands_on_origin_via_transform()
    test_integration_random_walk_state_matches_cumsum()
    visual_sanity_plots()


if __name__ == "__main__":
    main()
