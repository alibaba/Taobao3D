"""Minimal Euler flow sampler used by all inference stages."""

from __future__ import annotations

import numpy as np
import torch


class EulerFlowSampler:
    """Integrate a velocity or clean-sample flow from Gaussian noise."""

    def __init__(self, prediction_type: str = "velocity"):
        if prediction_type not in {"velocity", "x1"}:
            raise ValueError(f"Unsupported prediction type: {prediction_type!r}")
        self.prediction_type = prediction_type

    @staticmethod
    def _apply_time_shift(timesteps, shift: float):
        if shift == 1.0:
            return timesteps
        denominator = timesteps + shift * (1.0 - timesteps)
        shifted = timesteps / np.maximum(denominator, 1e-8)
        shifted[0] = 0.0
        shifted[-1] = 1.0
        return shifted

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        *,
        num_steps: int,
        device,
        dtype,
        model_kwargs,
        cfg_scale: float = 1.0,
        unconditional_kwargs=None,
        time_shift: float = 1.0,
        start_time: float = 0.0,
        initial_state=None,
        show_progress: bool = True,
        description: str = "Euler sampling",
    ):
        shape = tuple(shape)
        if not shape:
            raise ValueError("Sample shape cannot be empty")
        if not 0.0 <= start_time < 1.0:
            raise ValueError("start_time must be in [0, 1)")
        batch_size = shape[0]
        if initial_state is None:
            state = torch.randn(*shape, device=device, dtype=dtype)
        else:
            if tuple(initial_state.shape) != shape:
                raise ValueError(
                    f"Initial state shape {tuple(initial_state.shape)} != {shape}"
                )
            state = initial_state.to(device=device, dtype=dtype)
        timesteps = np.linspace(0, 1, num_steps + 1)
        if time_shift != 1.0:
            timesteps = self._apply_time_shift(timesteps, time_shift)
        if start_time:
            timesteps = start_time + (1.0 - start_time) * timesteps
        steps = range(num_steps)
        if show_progress:
            from tqdm import tqdm

            steps = tqdm(
                steps,
                desc=description,
                leave=False,
                dynamic_ncols=True,
                unit="step",
            )

        for index in steps:
            current_time = timesteps[index]
            delta_time = timesteps[index + 1] - current_time
            time_tensor = torch.full(
                (batch_size,),
                current_time,
                device=device,
                dtype=dtype,
            )
            conditional = model(state, time_tensor, **model_kwargs)
            if cfg_scale != 1.0 and unconditional_kwargs is not None:
                unconditional = model(
                    state,
                    time_tensor,
                    **unconditional_kwargs,
                )
                prediction = unconditional + cfg_scale * (
                    conditional - unconditional
                )
            else:
                prediction = conditional
            if self.prediction_type == "velocity":
                velocity = prediction
            else:
                velocity = (prediction - state) / max(
                    1.0 - current_time, 1e-5
                )
            state = state + velocity * delta_time
        return state
