from torch import Tensor, zeros

from torch.jit import script


@script
def calculate_v(x: Tensor, mu: Tensor) -> Tensor:
    t = x.size(0)

    # Create v tensor with zeros
    v = zeros(t, device=x.device)

    # Calculate v[1:] in one vectorized operation
    v[1:] = 4 * (x[1:] - mu[:-1]) ** 2

    return v
