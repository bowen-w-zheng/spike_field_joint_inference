# src/__init__.py
# Updated for run_joint_inference_single_trial.py interface

from src.params import OUParams
from src.utils_joint import Trace
from src.state_index import StateIndex

# Single-trial inference (NEW interface)
from src.run_joint_inference_single_trial import (
    run_joint_inference_single_trial,
    SingleTrialInferenceConfig,
)

# Simulation
from src.simulate_single_trial import (
    SingleTrialSimConfig,
    simulate_single_trial,
    build_history_design_single,
)

# EM
from src.em_ct_single_jax import em_ct_single_jax, EMSingleResult
from src.upsample_ct_single_fine import upsample_ct_single_fine, UpsampleSingleResult

# Utilities  
from src.utils_multitaper import derotate_tfr_align_start
from src.utils_common import (
    centres_from_win,
    map_blocks_to_fine,
    build_t2k,
    normalize_Y_to_RJMK,
    separated_to_interleaved,
    interleaved_to_separated,
)

from src.ou import _phi_q, kalman_filter_ou, kalman_filter_ou_numba

__all__ = [
    # Core
    'OUParams',
    'Trace', 
    'StateIndex',
    # Inference
    'run_joint_inference_single_trial',
    'SingleTrialInferenceConfig',
    # Simulation
    'SingleTrialSimConfig',
    'simulate_single_trial',
    'build_history_design_single',
    # EM
    'em_ct_single_jax',
    'EMSingleResult',
    'upsample_ct_single_fine',
    'UpsampleSingleResult',
    # Utilities
    'derotate_tfr_align_start',
    'centres_from_win',
    'map_blocks_to_fine',
    'build_t2k',
    'normalize_Y_to_RJMK',
    'separated_to_interleaved',
    'interleaved_to_separated',
]