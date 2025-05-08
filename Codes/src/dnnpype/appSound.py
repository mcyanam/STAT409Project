"""DNNPype/appSound.py: Driver for sound synthesis with Scipy."""

from __future__ import annotations
from typing import Optional, Tuple

import os
import argparse
import sounddevice as sd

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


if __name__ == "__main__":
    args = _argparse()
    for arg in vars(args):
        print(f"{arg}: {vars(args)[arg]}")
