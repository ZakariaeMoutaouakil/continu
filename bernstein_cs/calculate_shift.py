from typing import Tuple

from torch import Tensor, tensor, cat
from torch.jit import script

from .calculate_lambdas import calculate_lambdas
from .calculate_means import calculate_means
from .calculate_sigmas import calculate_sigmas
from .calculate_terms import calculate_terms
from .calculate_v import calculate_v
from .calculate_weighted_means import calculate_weighted_means


@script
def calculate_shift(x: Tensor, alpha: float, c: float) -> Tuple[float, float]:
    if x.dim() != 1:
        raise ValueError("x must be a 1D tensor.")
    if x.numel() < 2:
        raise ValueError("x must have at least two elements.")

    # Tensor with a single zero
    zero_tensor = tensor([0.], device=x.device)

    # Prepend zero to the original tensor
    y = cat((zero_tensor, x))

    means = calculate_means(x=y)

    sigmas = calculate_sigmas(x=y, mu=means)

    lambdas = calculate_lambdas(x=y, sigma=sigmas, alpha=alpha, c=c)

    weighted_means = calculate_weighted_means(x=y, lambdas=lambdas)

    v = calculate_v(x=y, mu=weighted_means)

    terms = calculate_terms(x=y, lambdas=lambdas, v=v, alpha=alpha)

    return weighted_means[-1].item(), terms[-1].item()
