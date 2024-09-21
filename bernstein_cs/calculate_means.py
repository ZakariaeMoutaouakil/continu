from torch import Tensor, cumsum, arange, zeros


def calculate_means(x: Tensor) -> Tensor:
    t = x.size(0)

    # Create a range tensor for indexing
    indices = arange(2, t + 1, device=x.device)

    # Calculate cumulative sum of x[1:]
    cum_sum = cumsum(x[1:], dim=0)

    # Calculate numerator for all indices at once
    numerator = 0.5 + cum_sum

    # Calculate means using vectorized operations
    means = zeros(t, device=x.device)
    means[1:] = numerator / indices

    return means
