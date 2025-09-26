from typing import Sequence
import mlx.core as mx
import math


def softmax(x: mx.array, axis: int | Sequence[int] | None) -> mx.array:
    x_max = mx.max(x, axis=axis, keepdims=True)
    x_exp = mx.exp(x - x_max)
    x_exp_sum = mx.sum(x_exp, axis=axis, keepdims=True)
    output = x_exp / x_exp_sum
    return output


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    y = x @ w.swapaxes(-1, -2)
    if bias is not None:
        y = y + bias
    return y


def silu(x: mx.array) -> mx.array:
    return x / (1 + mx.exp(-x))
