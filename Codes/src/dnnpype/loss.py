"""DNNPype/loss.py: Loss functions for DNNPype."""

from __future__ import annotations
from typing import List

import jax
import jax.numpy as jnp
import flax.nnx as nnx

def _isingNumber(
    x: jnp.ndarray):
    """Compute the (batched) Ising number."""
    # isBourdon, flueDepth, frequency, cutUpHeight, diameterToe, acousticIntensity
    # organ$Ising <- sqrt((((0.77 * 2) * organ$flueDepth)/(1.185 * (organ$cutUpHeight)^3))/organ$frequency)
    # return isingNumber
    pass

# @jax.jit
def flatNaiveLoss(
    model: nnx.Module,
    inputs: jnp.ndarray,
    ):
    """Flat loss + ideal Ising number

    Assume all ideal partials must be close to 1.
    Assume Ising number is 2.

    Args:
        model: The model to be trained.
        inputs: The input data.
    Returns:
        The loss value.
    """
    outputs = model(inputs)
    # This may not work
    isingNumber = outputs[:, 0]
    partials = outputs[:, 1:]
    # Compute the loss
    loss = jnp.mean(
        jnp.square(isingNumber - 2)
        + jnp.sum(jnp.square(partials - 1), axis=1)
    )
    return loss

def flatIsingLoss(
    model: nnx.Module,
    inputs: jnp.ndarray,
    ):
    """Flat loss + ideal Ising number

    Assume all ideal partials must be close to 1.
    Assume Ising number is given from data.

    Args:
        model: The model to be trained.
        inputs: The input data.
    Returns:
        The loss value.
    """
    outputs = model(inputs)
    # This may not work
    isingNumber = outputs[:, 0]
    partials = outputs[:, 1:]
    # Compute the loss
    loss = jnp.mean(
        jnp.square(isingNumber - 2)
        + jnp.sum(jnp.square(partials - 1), axis=1)
    )
    return loss



