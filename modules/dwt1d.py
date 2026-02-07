import torch
import torch.nn as nn


class DWT1D(nn.Module):
    """
    1D Haar DWT for audio.
    Input:  x  shape (B, C, L)  where L is even
    Output: y  shape (B, 2C, L/2)  = concat(low, high) along channel dim

    low  = (x_even + x_odd) / 2
    high = (x_even - x_odd) / 2
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"DWT1D expects (B,C,L), got {tuple(x.shape)}")
        B, C, L = x.shape
        if L % 2 != 0:
            raise ValueError(f"Length L must be even for Haar DWT, got L={L}")

        x_even = x[:, :, 0::2]
        x_odd  = x[:, :, 1::2]

        low  = (x_even + x_odd) * 0.5
        high = (x_even - x_odd) * 0.5

        return torch.cat([low, high], dim=1)


class IWT1D(nn.Module):
    """
    Inverse 1D Haar DWT.
    Input:  y shape (B, 2C, L/2)  where first C channels are low, next C are high
    Output: x shape (B, C, L)

    x_even = low + high
    x_odd  = low - high
    """
    def __init__(self):
        super().__init__()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 3:
            raise ValueError(f"IWT1D expects (B,2C,L/2), got {tuple(y.shape)}")
        B, CC, Lh = y.shape
        if CC % 2 != 0:
            raise ValueError(f"Channel dim must be even (2C), got {CC}")

        C = CC // 2
        low  = y[:, :C, :]
        high = y[:, C:, :]

        x_even = low + high
        x_odd  = low - high

        # interleave even/odd back to length 2*Lh
        L = Lh * 2
        x = torch.empty((B, C, L), device=y.device, dtype=y.dtype)
        x[:, :, 0::2] = x_even
        x[:, :, 1::2] = x_odd
        return x

