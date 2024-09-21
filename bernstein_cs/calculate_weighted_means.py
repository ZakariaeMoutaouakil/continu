from torch import Tensor, cumsum, zeros


def calculate_weighted_means(x: Tensor, lambdas: Tensor) -> Tensor:
    t = x.size(0)

    # Calculate cumulative sums
    lambda_cumsum = cumsum(lambdas[1:], dim=0)
    weighted_cumsum = cumsum(lambdas[1:] * x[1:], dim=0)

    # Calculate weighted means
    weighted_means = zeros(t, device=x.device)
    weighted_means[1:] = weighted_cumsum / lambda_cumsum

    return weighted_means
