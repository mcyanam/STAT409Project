"""DNNPype/appSound.py: Driver for sound synthesis with Scipy."""

from __future__ import annotations
from typing import Optional, List

import os
import argparse
import rich as r
import numpy as np
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.getcwd(), "output"),
        help="Directory to save the output files.",
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
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save wav files of the generated samples.",
        default=False,
        required=False,
    )
    return parser.parse_args()


def _handle_output_dir(
    *,
    output_dir: str,
) -> None:
    """Create the output directory if it does not exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        r.print(
            f"[bold green]Output directory [/bold green]"
            f"[cyan]{output_dir}[/cyan] "
            f"[bold green]created.[/bold green]"
        )
    else:
        r.print(
            f"[bold yellow]Output directory [/bold yellow]"
            f"[cyan]{output_dir}[/cyan] "
            f"[bold yellow]already exists.[/bold yellow]"
        )


def _print_args(
    *,
    args: argparse.Namespace,
) -> None:
    """Print the command line arguments."""
    r.print(
        f"[bold blue]Interactive sound generation and rating\n[/bold blue]"
        f"[bold blue]=======================================\n[/bold blue]"
        f"[bold]\tBase frequency: [/bold] {args.frequency} Hz\n"
        f"[bold]\tDuration: [/bold] {args.duration} s\n"
        f"[bold]\tSample rate: [/bold] {args.samplerate} Hz\n"
        f"[bold]\tNumber of samples: [/bold] {args.samples}\n"
        f"[bold]\tOutput directory: [/bold] {args.output_dir}\n"
        f"[bold]\tOutput summary file: [/bold] {args.output}\n"
        f"[bold]\tSave samples: [/bold] {args.save_samples}\n"
        f"[bold blue]=======================================\n\n[/bold blue]"
    )


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
# Handler rating functions
###############################################################################
def _play_sound(
    *,
    sound: np.ndarray,
    samplerate: int,
) -> None:
    """Play sound using sounddevice."""
    sd.play(sound, samplerate)
    sd.wait()


def _get_user_rating() -> float:
    """Get user rating for the sound."""
    rating: float = -1.0
    while rating < 0.0 or rating > 1.0:
        try:
            rating = float(input("Rate the sound (0-1): "))
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 1.")
    return rating


###############################################################################
# Main function
###############################################################################


def main() -> None:
    """Run with 'classify_samples' command."""


if __name__ == "__main__":
    args = _argparse()
    _handle_output_dir(output_dir=args.output_dir)
    _print_args(args=args)
