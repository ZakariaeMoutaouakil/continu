from math import log, sqrt

from torch import Tensor, log, sqrt, arange, log1p, tensor, zeros


def calculate_lambdas(x: Tensor, sigma: Tensor, alpha: float, c: float) -> Tensor:
    t = x.size(0)
    lambdas = zeros(t, device=x.device)

    log_term = log(tensor(1 / alpha))
    i_range = arange(1, t, device=x.device)

    numerator = 2 * log_term
    denominator = sigma[:-1] * i_range * log1p(i_range)

    lambdas[1:] = sqrt(numerator / denominator).clamp(max=c)

    return lambdas
