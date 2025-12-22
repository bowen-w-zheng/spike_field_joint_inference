"""
utils_joint.py - Trace container for MCMC inference.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Trace:
    """Container for MCMC trace samples."""
    beta:        List[np.ndarray] = field(default_factory=list)
    gamma:       List[np.ndarray] = field(default_factory=list)
    theta:       List = field(default_factory=list)  # List[OUParams]
    latent:      List[np.ndarray] = field(default_factory=list)
    fine_latent: List[np.ndarray] = field(default_factory=list)
