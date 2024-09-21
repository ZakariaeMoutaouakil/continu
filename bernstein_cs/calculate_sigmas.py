from torch import Tensor, cumsum, sqrt, arange, zeros

from torch.jit import script


@script
def calculate_sigmas(x: Tensor, mu: Tensor) -> Tensor:
    t = x.size(0)

    # Create indices tensor
    indices = arange(2, t + 1, device=x.device)

    # Calculate squared differences
    squared_diff = (x[1:] - mu[1:]) ** 2

    # Calculate cumulative sum of squared differences
    cumsum_squared_diff = cumsum(squared_diff, dim=0)

    # Calculate numerator for all indices at once
    numerator = 0.25 + cumsum_squared_diff

    # Calculate sigmas using vectorized operations
    sigmas = zeros(t, device=x.device)
    sigmas[0] = 0.25
    sigmas[1:] = sqrt(numerator / indices)

    return sigmas
