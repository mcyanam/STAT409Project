"""DNNPype/appSound.py: Driver for sound synthesis with Scipy."""

from __future__ import annotations
from typing import Optional, List

import os
import numpy as np
import argparse
import sounddevice as sd
import scipy.io.wavfile as wav

# import .sound as sound


###############################################################################
# Auxiliary functions
###############################################################################
def _argparse() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sound synthesis with Scipy.")
    # Required arguments
    parser.add_argument(
        "--frequency",
        type=float,
        default=440.0,
        help="Fundamental frequency of the sound in Hz.",
        required=True,
    )
    # Optional arguments
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration of the sound in seconds.",
        required=False,
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=44100,
        help="Sample rate of the sound in Hz.",
        required=False,
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of (random) samples to generate.",
        default=5,
        required=False,
    )
    _str_out: str = (
        "Output CSV file name (without .csv extension)."
        " If not provided, the name will be generated from previous arguments."
    )
    parser.add_argument(
        "--output",
        type=Optional[str],
        help=_str_out,
        default=None,
        required=False,
    )
    return parser.parse_args()


###############################################################################
# Partials distribution functions
###############################################################################
def _exponentialPartials(freq: float, theta: np.ndarray) -> np.ndarray:
    """Compute exponential partials."""
    n_part_from_freq = freq * np.arange(1, 9)
    slicing_idx: list[int] = [1]
    shift, intercept = np.split(theta, slicing_idx, axis=-1)
    partials = np.exp(-n_part_from_freq * intercept) + shift
    partials = partials / np.max(partials, axis=1, keepdims=True)
    return partials


def _linearPartials(freq: float, theta: np.ndarray) -> np.ndarray:
    """Compute linear partials."""
    n_part_from_freq = freq * np.arange(1, 9)
    slicing_idx: list[int] = [1]
    slope, intercept = np.split(theta, slicing_idx, axis=-1)
    partials = n_part_from_freq * slope + intercept
    partials = partials / np.max(partials, axis=1, keepdims=True)
    return partials


def _logPartials(freq: float, theta: np.ndarray) -> np.ndarray:
    """Compute log partials."""
    n_part_from_freq = freq * np.arange(1, 9)
    slicing_idx: list[int] = [1]
    slope, intercept = np.split(theta, slicing_idx, axis=-1)
    partials = (
        np.log(np.ones_like(n_part_from_freq) + n_part_from_freq * slope) + intercept
    )
    partials = partials / np.max(partials, axis=1, keepdims=True)
    return partials


###############################################################################
# Random parameter generation functions
###############################################################################
def _gen_theta_row(
    *,
    n_samples: int,
    min_val: float,
    max_val: float,
) -> np.ndarray:
    """Generate random parameters for exponential partials."""
    return np.random.uniform(min_val, max_val, size=n_samples)


###############################################################################
# Main function
###############################################################################
def main() -> None:
    pass


if __name__ == "__main__":
    args = _argparse()
    for arg in vars(args):
        print(f"{arg}: {vars(args)[arg]}")
