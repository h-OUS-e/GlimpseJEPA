"""Batched glimpse transform for JEPA-style training.

Generates zoomed/translated views of an image batch by accumulating per-step
actions on top of a stored source. All ops are vectorized over the batch via
``F.affine_grid`` + ``F.grid_sample``.

ActionGenerator design note (Q1 — trajectory length):
    Use a fixed ``T = T_max`` for the whole batch. Each sample independently
    picks its own ``t_stop`` ∈ [1, T_max]; deltas after ``t_stop`` are zeros
    (glimpse stays still for the remaining steps). Tensors stay rectangular
    (B, T, ...) and the "random number of steps" intent is preserved
    per-sample.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict


class Action(BaseModel):
    """A (zoom, tx, ty) triple in the source-image normalized frame.

    ``zoom`` is in log-scale space: the actual scale factor applied is
    ``exp(zoom)``. Positive values zoom in, negative values zoom out, and
    deltas compose by simple addition (which becomes multiplication in
    scale-space). ``tx`` / ``ty`` are absolute offsets in ``[-1, 1]`` where
    ``±1`` is the image edge.

    Fields may be Python floats (scalars — useful for sampling bounds or
    broadcasting a uniform delta across the batch) or 1-D tensors of shape
    ``(B,)`` for per-sample batched deltas and cumulative state.

    Attributes:
        zoom: Log-scale zoom. ``0`` is identity.
        tx: Horizontal offset in normalized source coords.
        ty: Vertical offset in normalized source coords.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    zoom: float | torch.Tensor = 0.0
    tx: float | torch.Tensor = 0.0
    ty: float | torch.Tensor = 0.0

    def to_batched(self, B: int, device, dtype) -> "Action":
        """Materialize all fields as ``(B,)`` tensors on the given device/dtype.

        Args:
            B: Batch size to broadcast scalar fields to.
            device: Target device.
            dtype: Target floating-point dtype.

        Returns:
            A new ``Action`` whose fields are all ``(B,)`` tensors.
        """

        def b(v):
            # tensor: just move; 0-d tensors expand to (B,)
            if isinstance(v, torch.Tensor):
                t = v.to(device=device, dtype=dtype)
                if t.dim() == 0:
                    t = t.expand(B)
                return t
            # scalar -> filled (B,) tensor
            return torch.full((B,), float(v), device=device, dtype=dtype)

        return Action(zoom=b(self.zoom), tx=b(self.tx), ty=b(self.ty))

    def __add__(self, other: "Action") -> "Action":
        """Component-wise sum of two actions (used to accumulate state)."""
        return Action(
            zoom=self.zoom + other.zoom,
            tx=self.tx + other.tx,
            ty=self.ty + other.ty,
        )


class GlimpseTransform:
    """Stateful, batched zoom + translate transform.

    Stores a source batch and a cumulative ``Action`` state. Each call to
    ``transform`` applies a *delta* on top of the current state and renders
    the resulting view at the source resolution. The whole batch is processed
    in a single ``affine_grid`` + ``grid_sample`` op (one fused CUDA kernel).

    Attributes:
        init_bounds: Half-widths of the uniform sampling ranges used by
            ``initialize_batch``. Each field gives the symmetric range
            ``[-field, +field]`` for that parameter.
        device: Optional device to move stored batches onto.
        mode: ``grid_sample`` interpolation mode (default ``"bilinear"``).
        padding_mode: ``grid_sample`` padding mode for samples that fall
            outside the source canvas (default ``"border"`` — replicates the
            edge pixel, which is what we want for zoom-out).
        align_corners: Forwarded to ``affine_grid`` / ``grid_sample``.
    """

    def __init__(
        self,
        init_bounds: Action | None = None,
        device: torch.device | None = None,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
    ):
        # default bounds: ±0.3 log-zoom (~0.74x to ~1.35x), ±0.5 translation
        self.init_bounds = (
            init_bounds if init_bounds is not None else Action(zoom=1, tx=1, ty=1)
        )
        self.device = device
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        # source batch (B, C, H, W) — set via set_batch / initialize_batch
        self._batch: torch.Tensor | None = None
        # cumulative per-sample state; fields are (B,) tensors
        self._state: Action | None = None

    def set_batch(self, x: torch.Tensor) -> "GlimpseTransform":
        """Store a fresh source batch and reset state to identity.

        Args:
            x: Source images of shape ``(B, C, H, W)``.

        Returns:
            ``self``, to allow chaining.
        """
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        # save the source; reset accumulated state since this is a new batch
        self._batch = x.to(self.device) if self.device is not None else x
        self.reset_state()
        return self

    def reset_state(self) -> None:
        """Zero the cumulative action state (next render shows the source as-is)."""
        x = self.batch
        B = x.shape[0]
        zeros = torch.zeros(B, device=x.device, dtype=x.dtype)
        # separate clones so future in-place ops on one field don't alias others
        self._state = Action(zoom=zeros, tx=zeros.clone(), ty=zeros.clone())

    def set_state(self, state: Action) -> None:
        """Install a known cumulative action as the current state.

        Each field of ``state`` must already be a ``(B,)`` tensor matching the
        stored batch (use :meth:`Action.to_batched` to broadcast scalars).
        Useful for initializing the transform from an externally-sampled
        action (e.g. one returned by :class:`ActionGenerator`).
        """
        x = self.batch
        B = x.shape[0]
        for name in ("zoom", "tx", "ty"):
            t = getattr(state, name)
            if not isinstance(t, torch.Tensor) or t.shape != (B,):
                raise ValueError(
                    f"state.{name} must be a tensor of shape ({B},); got "
                    f"{type(t).__name__} {getattr(t, 'shape', None)}"
                )
        self._state = state

    def initialize_batch(
        self,
        x: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Action:
        """Set state to a per-sample random action drawn from ``init_bounds``.

        Each field is sampled uniformly from ``[-bound, +bound]`` where
        ``bound`` is the corresponding field of ``self.init_bounds``.

        Args:
            x: Optional fresh source batch. If given, ``set_batch(x)`` is
                called first, otherwise the previously-stored batch is used.
            generator: Optional ``torch.Generator`` for reproducible sampling.

        Returns:
            The sampled initial ``Action`` (also stored as ``self.state``).
        """
        if x is not None:
            self.set_batch(x)
        b = self.batch
        B = b.shape[0]

        def sample(half_width):
            # scalar bound -> uniform in [-hw, +hw]
            hw = (
                float(half_width)
                if not isinstance(half_width, torch.Tensor)
                else float(half_width.item())
            )
            u = torch.rand(B, device=b.device, dtype=b.dtype, generator=generator)
            return (u * 2.0 - 1.0) * hw

        # save the new initial action state
        self._state = Action(
            zoom=sample(self.init_bounds.zoom),
            tx=sample(self.init_bounds.tx),
            ty=sample(self.init_bounds.ty),
        )
        return self._state

    @property
    def batch(self) -> torch.Tensor:
        """The currently stored source batch."""
        if self._batch is None:
            raise RuntimeError("call set_batch() or initialize_batch() first")
        return self._batch

    @property
    def state(self) -> Action:
        """The current cumulative action (per-sample tensors of shape ``(B,)``)."""
        if self._state is None:
            raise RuntimeError("no state; call set_batch() or initialize_batch() first")
        return self._state

    def transform(self, delta: Action) -> torch.Tensor:
        """Accumulate ``delta`` into state and render the resulting view.

        The new state is ``self.state + delta``. The rendered view samples
        the source via an inverse affine map built from the new state:
        scale ``= exp(state.zoom)`` along both axes, then translation by
        ``(state.tx, state.ty)`` in normalized source coords.

        Args:
            delta: Per-sample delta to apply on top of the current state.
                Scalar fields broadcast across the batch; tensor fields must
                be shape ``(B,)`` (or 0-d, which also broadcasts).

        Returns:
            View tensor of shape ``(B, C, H, W)`` — same resolution as the
            source.
        """
        x = self.batch
        B = x.shape[0]

        # broadcast / move the delta onto the batch device/dtype
        delta_b = delta.to_batched(B, device=x.device, dtype=x.dtype)
        for name, t in (("zoom", delta_b.zoom), ("tx", delta_b.tx), ("ty", delta_b.ty)):
            assert isinstance(t, torch.Tensor)
            if t.shape != (B,):
                raise ValueError(
                    f"delta.{name} must broadcast to ({B},), got {tuple(t.shape)}"
                )

        # accumulate: new_state = old_state + delta
        self._state = self.state + delta_b

        zoom = self._state.zoom
        tx = self._state.tx
        ty = self._state.ty
        assert (
            isinstance(zoom, torch.Tensor)
            and isinstance(tx, torch.Tensor)
            and isinstance(ty, torch.Tensor)
        )

        # affine_grid's theta maps OUTPUT coords -> INPUT coords, so the
        # diagonal is the inverse of the visual scale: zoom-in (scale > 1)
        # samples a smaller window of the source (1/scale < 1).
        inv_scale = torch.exp(-zoom)
        zero = torch.zeros_like(inv_scale)

        # theta[b] = [[inv_scale, 0, tx],
        #             [0, inv_scale, ty]]
        theta = torch.stack(
            [
                torch.stack([inv_scale, zero, tx], dim=-1),
                torch.stack([zero, inv_scale, ty], dim=-1),
            ],
            dim=-2,
        )

        # one fused kernel: build sampling grid + bilinear sample with border padding
        grid = F.affine_grid(theta, size=x.shape, align_corners=self.align_corners)
        return F.grid_sample(
            x,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


class ActionGenerator(ABC):
    """Abstract trajectory schedule sampler for the glimpse environment.

    Subclasses implement :meth:`sample`, which returns a triple
    ``(init, deltas, t_stop)``:

    - ``init`` (:class:`Action`): per-sample cumulative state at step 0; each
      field is a ``(B,)`` tensor sampled uniform in
      ``[-init_bounds.<axis>, +init_bounds.<axis>]``.
    - ``deltas``: ``(B, T_max, 3)`` float tensor with column order
      ``[zoom, tx, ty]``. Index ``k`` is the delta from cumulative state at
      step ``k`` to step ``k+1``. For each sample ``b``, entries with
      ``k >= t_stop[b]`` are zero (glimpse stays still for the remaining steps).
    - ``t_stop``: ``(B,)`` long tensor, values in ``[1, T_max]``, sampled
      uniform integer per sample. Number of non-zero deltas.

    Args:
        init_bounds: Half-widths of the symmetric uniform sampling ranges for
            the initial action. Fields must be non-negative scalars or 0-d
            tensors (negative bounds are meaningless under uniform-symmetric
            sampling; per-sample bounds are not supported).
        T_max: Maximum trajectory length (number of deltas). Must be ``>= 1``.
    """

    def __init__(self, init_bounds: Action, T_max: int):
        if T_max < 1:
            raise ValueError(f"T_max must be >= 1, got {T_max}")
        for name in ("zoom", "tx", "ty"):
            self._check_bound(getattr(init_bounds, name), f"init_bounds.{name}")
        self.init_bounds = init_bounds
        self.T_max = T_max

    @staticmethod
    def _check_bound(v, name: str) -> None:
        if isinstance(v, torch.Tensor):
            if v.dim() != 0:
                raise ValueError(f"{name} must be scalar or 0-d tensor, got shape {tuple(v.shape)}")
            f = float(v.item())
        else:
            f = float(v)
        if f < 0:
            raise ValueError(f"{name} must be non-negative, got {f}")

    def _sample_init(
        self,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> Action:
        """Sample a per-sample initial action uniform in ``±init_bounds``."""

        def sample(half_width):
            hw = (
                float(half_width.item())
                if isinstance(half_width, torch.Tensor)
                else float(half_width)
            )
            u = torch.rand(B, device=device, dtype=dtype, generator=generator)
            return (u * 2.0 - 1.0) * hw

        return Action(
            zoom=sample(self.init_bounds.zoom),
            tx=sample(self.init_bounds.tx),
            ty=sample(self.init_bounds.ty),
        )

    def _sample_t_stop(
        self,
        B: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Sample ``t_stop ~ U{1, ..., T_max}`` per sample as a long tensor."""
        return torch.randint(
            low=1,
            high=self.T_max + 1,
            size=(B,),
            device=device,
            dtype=torch.long,
            generator=generator,
        )

    @abstractmethod
    def sample(
        self,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> tuple[Action, torch.Tensor, torch.Tensor]:
        """Return ``(init, deltas, t_stop)`` for a fresh batch of trajectories."""
        raise NotImplementedError


class RandomWalkGenerator(ActionGenerator):
    """Random-walk trajectory schedule.

    For each sample ``b``, the trajectory is:
    ``state_0 = init[b]`` then for ``k < t_stop[b]``,
    ``deltas[b, k, axis] ~ U[-step_bounds.<axis>, +step_bounds.<axis>]``
    independently per axis and step. For ``k >= t_stop[b]``, ``deltas`` is
    zero (glimpse stays still). No clipping — cumulative state may drift
    outside ``init_bounds``.

    Edge cases:
        * ``T_max = 1``: legal; ``t_stop`` is always 1, ``deltas`` is ``(B, 1, 3)``.
        * ``init_bounds`` field is 0: that axis is always 0 in ``init`` but the
          delta on that axis is unaffected (sampled from ``step_bounds``).
        * ``step_bounds > init_bounds``: legal but unusual — random walk may
          drift outside the init range.

    Args:
        init_bounds: See :class:`ActionGenerator`.
        T_max: See :class:`ActionGenerator`.
        step_bounds: Half-widths of the per-step uniform delta ranges. Must be
            non-negative scalars or 0-d tensors. Defaults to
            ``init_bounds / T_max`` so per-step delta magnitudes are
            comparable to a return-to-origin schedule.
    """

    def __init__(
        self,
        init_bounds: Action,
        T_max: int,
        step_bounds: Action | None = None,
    ):
        super().__init__(init_bounds, T_max)
        if step_bounds is None:
            step_bounds = Action(
                zoom=float(self._scalar(init_bounds.zoom)) / T_max,
                tx=float(self._scalar(init_bounds.tx)) / T_max,
                ty=float(self._scalar(init_bounds.ty)) / T_max,
            )
        for name in ("zoom", "tx", "ty"):
            self._check_bound(getattr(step_bounds, name), f"step_bounds.{name}")
        self.step_bounds = step_bounds

    @staticmethod
    def _scalar(v):
        return float(v.item()) if isinstance(v, torch.Tensor) else float(v)

    def sample(self, B, device, dtype, generator=None):
        init = self._sample_init(B, device, dtype, generator)
        t_stop = self._sample_t_stop(B, device, generator)

        # uniform deltas in [-bound, +bound] per axis, full (B, T_max) tensor
        u = torch.rand(B, self.T_max, 3, device=device, dtype=dtype, generator=generator)
        u = u * 2.0 - 1.0
        bounds = torch.tensor(
            [
                self._scalar(self.step_bounds.zoom),
                self._scalar(self.step_bounds.tx),
                self._scalar(self.step_bounds.ty),
            ],
            device=device,
            dtype=dtype,
        )
        deltas = u * bounds  # broadcast (B, T_max, 3) * (3,)

        # zero out deltas at k >= t_stop[b]
        k_idx = torch.arange(self.T_max, device=device).unsqueeze(0)  # (1, T_max)
        active = (k_idx < t_stop.unsqueeze(1)).to(dtype)               # (B, T_max)
        deltas = deltas * active.unsqueeze(-1)

        return init, deltas, t_stop


class ReturnToOriginGenerator(ActionGenerator):
    """Linear return-to-origin trajectory schedule.

    For each sample ``b``, ``init[b]`` is sampled per the base class. The
    constant per-step delta ``-init[b] / t_stop[b]`` is broadcast across
    ``k = 0, ..., t_stop[b] - 1`` so the cumulative state at step
    ``t_stop[b]`` is exactly the origin ``(0, 0, 0)``. For ``k >= t_stop[b]``
    the delta is zero (already at origin).

    Edge cases:
        * ``T_max = 1``: legal; ``t_stop`` is always 1, the single delta is
          ``-init`` (lands on origin in one step).
        * ``t_stop = 1`` (any ``T_max``): single delta of ``-init``; falls out
          of the ``-init / t_stop`` formula with no special case.
        * ``init_bounds`` field is 0: that axis is always 0 in ``init``, so
          the corresponding delta is also 0.

    Args:
        init_bounds: See :class:`ActionGenerator`.
        T_max: See :class:`ActionGenerator`.
    """

    def __init__(self, init_bounds: Action, T_max: int):
        super().__init__(init_bounds, T_max)

    def _build_deltas(
        self,
        init: Action,
        t_stop: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Construct ``(B, T_max, 3)`` deltas given a sampled init and t_stop."""
        B = t_stop.shape[0]
        init_stack = torch.stack([init.zoom, init.tx, init.ty], dim=-1)  # (B, 3)
        per_step = -init_stack / t_stop.to(dtype).unsqueeze(-1)          # (B, 3)

        # broadcast per_step across T_max, then zero out k >= t_stop
        deltas = per_step.unsqueeze(1).expand(B, self.T_max, 3).clone()  # (B, T_max, 3)
        k_idx = torch.arange(self.T_max, device=device).unsqueeze(0)     # (1, T_max)
        active = (k_idx < t_stop.unsqueeze(1)).to(dtype).unsqueeze(-1)   # (B, T_max, 1)
        return deltas * active

    def sample(self, B, device, dtype, generator=None):
        init = self._sample_init(B, device, dtype, generator)
        t_stop = self._sample_t_stop(B, device, generator)
        deltas = self._build_deltas(init, t_stop, device, dtype)
        return init, deltas, t_stop
