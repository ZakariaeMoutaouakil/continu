from typing import Callable

from torch import Tensor, cat, zeros_like, max


def calculate_lower_bound_bonferroni(x: Tensor, means: Tensor, index: int, term: Callable[[Tensor], float],
                          func: Callable[[float], float]) -> float:
    assert 0 <= index < means.numel(), "Index is out of range."
    assert means.numel() == x.shape[1], "Number of means does not match the number of features."

    result = zeros_like(means)
    for i in range(means.numel()):
        if i == index:
            result[i] = means[i].item() - term(x[:, i])
        else:
            result[i] = means[i].item() + term(x[:, i])

    value_at_index = result[index].item()
    max_of_others = max(cat([result[:index], result[index + 1:]])).item()
    difference = func(value_at_index) - func(max_of_others)

    return difference
