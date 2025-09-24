import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # Saft Softmax Implmentation
    x_max = mx.max(x, axis=axis, keepdims=True)
    x_exp = mx.exp(x - x_max)
    x_exp_sum = mx.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_exp_sum


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    y = x @ w.T
    if bias is not None:
        y = y + bias
    return y


def silu(x: mx.array) -> mx.array:
    pass
