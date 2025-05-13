"""DNNPype/appDNN.py: Wrapper for DNNPype to run the model."""

from __future__ import annotations
from typing import Optional, Dict, Any
from enum import Enum

import os
import argparse
import rich as r
import numpy as np
import sounddevice as sd
import polars as pl
import plotly.graph_objects as go

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

import model
import loss
import opt


###############################################################################
# Defaults and constants # TODO: Use a toml
###############################################################################
_lr: float = 0.01


###############################################################################
# Auxiliary functions
###############################################################################
def _argsparse() -> argparse.Namespace:
    """Parse command line arguments."""
    _dsc: str = (
        "DNNPype: DNN-based Ising number prediction\n"
        "This script is a wrapper for DNNPype to run the model.\n"
    )
    parser = argparse.ArgumentParser(description=_dsc)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model",
        default=False,
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Predict the model",
        default=False,
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the model",
        default=False,
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        help="Number of hidden layers",
        default=2,
    )
    parser.add_argument(
        "--dim-hidden",
        type=int,
        help="Number of hidden units",
        default=12,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=0,
    )
    # TODO: Add load model option
    # parser.add_argument(
    #     "--load",
    #     action="store_true",
    #     help="Load the model",
    #     default=False,
    # )
    return parser.parse_args()


def _print_args(
    *,
    args: argparse.Namespace,
) -> None:
    """Print the command line arguments."""
    r.print(
        f"[bold blue]DNNPype - DNN-based Ising number prediction\n[/bold blue]"
        f"[bold blue]===========================================\n[/bold blue]"
        f"[bold]\tTrain: [/bold] {args.train}\n"
        f"[bold]\tPredict: [/bold] {args.predict}\n"
        f"[bold]\tSave: [/bold] {args.save}\n"
        # f"[bold]\tLoad: [/bold] {args.load}\n"
        f"[bold]\tNumber of hidden layers: [/bold] {args.n_hidden}\n"
        f"[bold]\tNumber of hidden units: [/bold] {args.dim_hidden}\n"
        f"[bold blue]===========================================\n\n[/bold blue]"
    )


###############################################################################
# Data loading functions
###############################################################################
def _load_data() -> pl.DataFrame:
    """Load the data from the given path."""
    pass

def _load_model() -> nnx.Module:
    """Load the model from the given path."""
    pass

###############################################################################
# Main function
###############################################################################
def main() -> None:
    """Run with 'run_model' command."""
    args = _argsparse()
    _print_args(args=args)

    # Set up the model
    rngs = nnx.Rngs(args.seed)
    dnn = model.SmallDNN(
        n_hidden=args.n_hidden,
        dim_hidden=args.dim_hidden,
        rngs=rngs,
    )

    # Set up loss function
    loss_fn = loss.refLoss  # takes model, inputs, theta, refPartials

    # Set up optimizer
    optimizer = opt.get_optimizer(
        model=dnn,
        optax_optimizer=optax.adam,
        learning_rate=_lr,
    )




if __name__ == "__main__":
    main()
