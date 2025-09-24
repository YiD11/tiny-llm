import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    dim_size = len(key.shape)
    key_transpose = key.transpose(*range(dim_size-2), dim_size-1, dim_size-2)
    attention = mx.matmul(query, key_transpose)
    attention = attention * (scale if scale else 1 / key.shape[-1]**0.5)
    if mask is not None:
        attention = attention + mask
    output = mx.matmul(softmax(attention, axis=-1), value)
    return output


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = query.shape
        query = (
            linear(query, self.wq)
            .reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        key = (
            linear(key, self.wk)
            .reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        value = (
            linear(value, self.wv)
            .reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn_output = (
            scaled_dot_product_attention_simple(query, key, value, mask=mask)
            .transpose(0, 2, 1, 3)
            .reshape(batch_size, seq_len, self.hidden_size)
        )
        return linear(attn_output, self.wo)
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
