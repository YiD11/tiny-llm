import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2) / dims))
        positions = mx.arange(seq_len)
        sinusoid_inp = mx.einsum("i,j->ij", positions, inv_freq)
        self.sin = mx.sin(sinusoid_inp).astype(mx.float32)
        self.cos = mx.cos(sinusoid_inp).astype(mx.float32)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        _, seq_len, _, head_dim = x.shape
        assert head_dim == self.dims, "head_dim must be equal to dims"
        if offset is None:
            sin = self.sin[:seq_len, :].reshape(1, seq_len, 1, head_dim // 2)
            cos = self.cos[:seq_len, :].reshape(1, seq_len, 1, head_dim // 2)
        elif isinstance(offset, slice):
            sin = self.sin[offset, :].reshape(1, -1, 1, head_dim // 2)
            cos = self.cos[offset, :].reshape(1, -1, 1, head_dim // 2)
        else:
            sin = mx.concat(
                [self.sin[o, :].reshape(1, -1, 1, head_dim) for o in offset], axis=1
            )
            cos = mx.concat(
                [self.cos[o, :].reshape(1, -1, 1, head_dim) for o in offset], axis=1
            )
        output = mx.zeros_like(x)
        if self.traditional:
            x1, x2 = x[..., ::2], x[..., 1::2]
            output[..., ::2] = x1 * cos - x2 * sin
            output[..., 1::2] = x1 * sin + x2 * cos
        else:
            x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
            output[..., : head_dim // 2] = x1 * cos - x2 * sin
            output[..., head_dim // 2 :] = x1 * sin + x2 * cos
        return output
            
