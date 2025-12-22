"""
params.py - OU parameter container for CT-SSMT.

Shapes (single source of truth)
--------------------------------
    lam     : (J, M) decay rates λ (s⁻¹)
    sig_v   : (J, M) innovation σ_v
    sig_eps : (J, M) observation σ_ε

All downstream modules must abide by these shapes.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

__all__ = ["OUParams"]


@dataclass
class OUParams:
    """
    Ornstein–Uhlenbeck dynamics, one set of scalars per (band, taper).

    Shapes
    ------
    lam     : (J, M)    decay rates   λ  (s⁻¹)
    sig_v   : (J, M)    innovation σ_v
    sig_eps : (J, M)    observation σ_ε  (multitaper FFT noise)
    """
    lam: np.ndarray       # (J,M)
    sig_v: np.ndarray     # (J,M)
    sig_eps: np.ndarray   # (J,M)

    def copy(self) -> "OUParams":
        """Deep copy – avoids accidental in-place modification."""
        return OUParams(
            lam=self.lam.copy(),
            sig_v=self.sig_v.copy(),
            sig_eps=self.sig_eps.copy(),
        )
