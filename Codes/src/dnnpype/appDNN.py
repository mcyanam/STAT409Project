"""DNNPype/appDNN.py: Wrapper for DNNPype to run the model."""

from __future__ import annotations
from typing import Tuple

import os
import argparse
import rich as r
import polars as pl

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from functools import partial

import model
import loss
import opt


###############################################################################
# Defaults and constants # TODO: Use a toml
###############################################################################
_lr: float = 0.01
_n_epochs: int = 100
_n_batches: int = 10


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
    # TODO: Add predict option
    # parser.add_argument(
    #     "--predict",
    #     action="store_true",
    #     help="Predict the model",
    #     default=False,
    # )
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
        # f"[bold]\tPredict: [/bold] {args.predict}\n"
        f"[bold]\tSave: [/bold] {args.save}\n"
        # f"[bold]\tLoad: [/bold] {args.load}\n"
        f"[bold]\tNumber of hidden layers: [/bold] {args.n_hidden}\n"
        f"[bold]\tNumber of hidden units: [/bold] {args.dim_hidden}\n"
        f"[bold blue]===========================================\n\n[/bold blue]"
    )


###############################################################################
# Data loading functions
###############################################################################
def _load_data() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load the data from the given path."""
    # TODO: Hardcode the path
    path = "../../../Data/allOrgan.csv"
    df = pl.read_csv(path)
    input_cols = [
        "isBourdon",
        "flueDepth",
        "frequency",
        "cutUpHeight",
        "diameterToe",
        "acousticIntensity",
    ]
    output_cols = [f"partial{i}" for i in range(1, 9)]
    inputs = df.select(input_cols).to_numpy()
    outputs = df.select(output_cols).to_numpy()
    inputs = jnp.array(inputs)
    outputs = jnp.array(outputs)
    return inputs, outputs


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

    # Set up optimizer
    optimizer = opt.get_optimizer(
        model=dnn,
        optax_optimizer=optax.adam,
        learning_rate=_lr,
    )




if __name__ == "__main__":
    main()
