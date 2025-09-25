import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        logprobs = copy.copy(logprobs)
        if top_k is not None and top_k > 0:
            masked = mx.argpartition(-logprobs, kth=top_k-1, axis=-1)[:, top_k:]
            logprobs = mx.put_along_axis(logprobs, indices=masked, values=mx.array(-mx.inf), axis=-1)
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = mx.take_along_axis(logprobs, indices=sorted_idx, axis=-1)
            cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
            mask_elements = cumsum < top_p
            mask_elements[..., 0] = True
            filtered = mx.where(mask_elements, sorted_logprobs, -mx.inf)
            logprobs = mx.put_along_axis(logprobs, indices=sorted_idx, values=filtered, axis=-1)
        logprobs = logprobs / temp
        return mx.random.categorical(logprobs, axis=-1)

    return sample

if __name__ == '__main__':
    x = mx.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    sampler = make_sampler(temp=0.7, top_p=2, top_k=2)
    print(sampler(x))