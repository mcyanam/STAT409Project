"""DNNPype/loss.py: Loss functions for DNNPype."""

from __future__ import annotations
from typing import List

import jax
import jax.numpy as jnp
import flax.nnx as nnx


import jax.numpy as jnp

@jax.jit
def _isingNumber(x_pipe: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Compute the (batched) Ising number."""
    isBourdon, flueDepth, frequency, cutUpHeight, diameterToe, acousticIntensity = jnp.split(x_pipe, 6, axis=1)
    pressure, density = theta
    isingNumber = (1 / frequency) * jnp.sqrt((2 * pressure * flueDepth) / (density * jnp.power(cutUpHeight, 3)))
    return isingNumber

@jax.jit
def _exponentialPartials(freq: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Compute the (batched) exponential partials."""
    n_part_from_freq = jnp.expand_dims(freq, axis=1) * jnp.arange(1, 9)
    shift, intercept = jnp.split(theta, 2, axis=-1)
    partials = jnp.exp(-n_part_from_freq * intercept) + shift
    partials = partials / jnp.max(partials, axis=1, keepdims=True)
    return partials


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
        jnp.square(isingNumber - 2) + jnp.sum(jnp.square(partials - 1), axis=1)
    )
    return loss

if __name__ == "__main__":
    # Test the loss function
    fakeData = jnp.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=jnp.float32)
    fakeTheta = jnp.array([1, 2], dtype=jnp.float32)
    print(_isingNumber(fakeData, fakeTheta))
    fakeFreq = jnp.array([1, 2, 3, 4, 5, 6], dtype=jnp.float32)
    print(_exponentialPartials(fakeFreq, fakeTheta))
