"""Visual + sanity-check script for ``ActionGenerator``.

Run:
    "C:/Users/Ous/miniconda3/envs/ML/python.exe" test_scripts/test_action_generator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from glimpse import Action, ActionGenerator, GlimpseTransform


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
    from glimpse import RandomWalkGenerator

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
    from glimpse import RandomWalkGenerator

    init_bounds = Action(zoom=1.0, tx=2.0, ty=3.0)
    gen = RandomWalkGenerator(init_bounds=init_bounds, T_max=10)  # no step_bounds
    assert float(gen.step_bounds.zoom) == 1.0 / 10
    assert float(gen.step_bounds.tx) == 2.0 / 10
    assert float(gen.step_bounds.ty) == 3.0 / 10
    print("[random-walk] default step_bounds = init_bounds / T_max OK")


def test_random_walk_reproducibility():
    from glimpse import RandomWalkGenerator

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


def main():
    torch.manual_seed(0)
    test_base_helpers_shapes_and_dtypes()
    test_random_walk_shapes_bounds_and_padding()
    test_random_walk_default_step_bounds_division()
    test_random_walk_reproducibility()


if __name__ == "__main__":
    main()
