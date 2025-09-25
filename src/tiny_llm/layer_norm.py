import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        norm = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True)) + self.eps
        output = x / norm * self.weight
        return output.astype(dtype)
