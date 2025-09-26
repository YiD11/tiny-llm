import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    attention = mx.matmul(query, key.swapaxes(-2, -1))
    attention = attention * (scale if scale is not None else mx.rsqrt(key.shape[-1]))
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


def causal_mask(L: int, S: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=max(0, S - L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    output_shape = query.shape
    query_head, query_len, head_dim = query.shape[-3:]
    kv_head, key_len, _ = key.shape[-3:]
    assert query_head % kv_head == 0, "query_head must be divisible by key_head"
    group_size = query_head // kv_head
    batch_dims = query.shape[:-3]
    dtype = query.dtype
    query = query.reshape(
        *batch_dims, kv_head, group_size, query_len, head_dim
    ).astype(mx.float32)
    key = key.reshape(
        *batch_dims, kv_head, 1, key_len, head_dim
    ).astype(mx.float32)
    value = value.reshape(
        *batch_dims, kv_head, 1, key_len, head_dim
    ).astype(mx.float32)
    scale = scale if scale is not None else mx.rsqrt(head_dim)
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(query_len, key_len, query.dtype)
        else:
            mask = mask.reshape(*batch_dims, kv_head, group_size, query_len, key_len)
        scores = scores + mask
    output = mx.matmul(softmax(scores, axis=-1), value)
    return output.reshape(output_shape).astype(dtype)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
