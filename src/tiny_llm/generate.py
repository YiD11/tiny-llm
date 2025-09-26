import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper, NaiveStreamingDetokenizer
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable
from tqdm import tqdm
from itertools import count


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        output_logits = model(y[None])
        logits = output_logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        if sampler is None:
            next_token_id = mx.argmax(logprobs, axis=-1)
        else:
            next_token_id = sampler(logprobs)
        return next_token_id, output_logits

    token_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer: NaiveStreamingDetokenizer = tokenizer.detokenizer
    detokenizer.reset()
    for _ in tqdm(count(), desc="Looping"):
        next_token_id, _ = _step(model, token_ids)
        if next_token_id.item() in tokenizer.eos_token_ids:
            break
        token_ids = mx.concat([token_ids, next_token_id])
        detokenizer.add_token(next_token_id.item())
    response = detokenizer.last_segment
    print(response)
    return response


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass
