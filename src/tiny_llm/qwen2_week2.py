import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights
from .kv_cache import TinyKvCache, TinyKvFullCache


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
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
        self.use_flash_attention = use_flash_attention

    def __call__(
        self,
        x: mx.array,
        offsets: list[int] | int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        wq = dequantize_linear(self.wq)
        wk = dequantize_linear(self.wk)
        wv = dequantize_linear(self.wv)
        wo = dequantize_linear(self.wo)
        batch_size, seq_len, _ = x.shape
        q = linear(x, wq, self.bq).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = linear(x, wk, self.bk).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = linear(x, wv, self.bv).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        if isinstance(offsets, int):
            positional_offsets = slice(int(offsets), int(offsets + seq_len))
        else:
            positional_offsets = [
                slice(int(offset), int(offset + seq_len)) for offset in offsets
            ]
        q_rope = self.rope(q, offset=positional_offsets)
        k_rope = self.rope(k, offset=positional_offsets)
        q_rope = q_rope.swapaxes(1, 2)
        k_rope = k_rope.swapaxes(1, 2)
        v = v.swapaxes(1, 2)
        k, v = cache.update_and_fetch(k_rope, v, mask_length=seq_len, mask=mask)
        attention_output = scaled_dot_product_attention_grouped(
            q_rope, k, v, scale=self.scale, mask=mask
        )
        attention_output = attention_output.swapaxes(1, 2)
        attention_output = attention_output.reshape(
            batch_size, seq_len, self.hidden_size
        )
        output = linear(attention_output, wo)
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
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        w_gate = dequantize_linear(self.w_gate)
        w_up = dequantize_linear(self.w_up)
        w_down = dequantize_linear(self.w_down)
        up = linear(x, w_up)
        gate = linear(x, w_gate)
        gated = silu(gate) * up
        output = linear(gated, w_down)
        return output


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
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
        self.input_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, weight=w_input_layernorm
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, weight=w_post_attention_layernorm
        )
        self.attention_layer = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            self.wq,
            self.wk,
            self.wv,
            self.wo,
            self.bq,
            self.bk,
            self.bv,
            max_seq_len,
            theta,
            use_flash_attention,
        )
        self.mlp_layer = Qwen2MLP(
            hidden_size, intermediate_size, self.w_gate, self.w_up, self.w_down
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pre_normed = self.input_layernorm(x)
        attention = self.attention_layer(
            pre_normed, mask=mask, offsets=offset, cache=cache
        )
        hidden = x + attention
        post_normed = self.post_attention_layernorm(hidden)
        mlp_out = self.mlp_layer(post_normed)
        output = hidden + mlp_out
        return output


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_dim = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        self.precision = mx.float16

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_dim,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(
                self.precision
            ),
        )
        self.layers = []

        for i in range(self.num_hidden_layers):
            wq = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.q_proj
            )
            wk = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.k_proj
            )
            wv = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.v_proj
            )
            wo = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.o_proj
            )
            w_gate = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].mlp.gate_proj
            )
            w_up = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].mlp.up_proj
            )
            w_down = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].mlp.down_proj
            )

            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(
                    self.precision
                ),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(
                    self.precision
                ),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(
                    self.precision
                ),
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=mlx_model.model.layers[
                    i
                ].input_layernorm.weight.astype(self.precision),
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight.astype(self.precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
                use_flash_attention=enable_flash_attn,
            )
            self.layers.append(layer)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(self.precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        hidden = self.embedding(inputs)
        for i in range(self.num_hidden_layers):
            if len(cache) < i + 1:
                cache.extend([TinyKvFullCache() for _ in range(max(1, 3 * len(cache) // 4))])
            hidden = self.layers[i](hidden, offset, cache[i], mask="causal")
        hidden = self.norm(hidden)
        if self.w_lm_head is not None:
            return linear(hidden, self.w_lm_head)
        else:
            return self.embedding.as_linear(hidden)
