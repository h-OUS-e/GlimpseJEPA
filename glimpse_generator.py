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
