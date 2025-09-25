import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(
            dims=hidden_size // num_heads,
            seq_len=max_seq_len,
            base=theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        q = linear(x, self.wq, self.bq).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = linear(x, self.wk, self.bk).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = (
            linear(x, self.wv, self.bv)
            .reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        q_rope = self.rope(q, offset=slice(0, seq_len)).swapaxes(1, 2)
        k_rope = self.rope(k, offset=slice(0, seq_len)).swapaxes(1, 2)
        attention_output = scaled_dot_product_attention_grouped(
            q_rope, k_rope, v, mask=mask, scale=self.scale
        )
        attention_output = attention_output.swapaxes(1, 2)
        attention_output = attention_output.reshape(
            batch_size, seq_len, self.hidden_size
        )
        output = linear(attention_output, self.wo)
        return output

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def scale(self):
        return mx.rsqrt(self.head_dim)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        up = linear(x, self.w_up)
        gate = linear(x, self.w_gate)
        gated = silu(gate) * up
        output = linear(gated, self.w_down)
        return output


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.w_input_layernorm = w_input_layernorm
        self.w_post_attention_layernorm = w_post_attention_layernorm
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.input_layernorm = RMSNorm(
            self.hidden_size, self.w_input_layernorm, eps=self.rms_norm_eps
        )
        self.mha = Qwen2MultiHeadAttention(
            self.hidden_size,
            self.num_attention_heads,
            self.num_kv_heads,
            self.wq,
            self.wk,
            self.wv,
            self.wo,
            self.bq,
            self.bk,
            self.bv,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
        )
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, self.w_post_attention_layernorm, eps=self.rms_norm_eps
        )

        self.mlp = Qwen2MLP(
            dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pre_normed = self.input_layernorm(x)
        attn = self.mha(pre_normed, mask=mask)
        hidden = x + attn
        post_normed = self.post_attention_layernorm(hidden)
        mlp_out = self.mlp(post_normed)
        output = hidden + mlp_out
        return output


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.mlx_model = mlx_model

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        return self.mlx_model(inputs)
