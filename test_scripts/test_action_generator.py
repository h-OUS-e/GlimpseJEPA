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


def main():
    torch.manual_seed(0)
    test_base_helpers_shapes_and_dtypes()


if __name__ == "__main__":
    main()
