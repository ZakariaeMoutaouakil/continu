from torch import Tensor, cumsum, log, log1p, tensor, zeros


def calculate_terms(x: Tensor, lambdas: Tensor, v: Tensor, alpha: float) -> Tensor:
    t = x.size(0)

    # Precompute log term
    log_term = log(tensor(1 / alpha, device=x.device))

    # Vectorized psi_e calculation
    psi_e_values = (-log1p(-lambdas[1:]) - lambdas[1:]) / 4

    # Calculate cumulative sums
    cumulative_lambda = cumsum(lambdas[1:], dim=0)
    cumulative_v_psi = cumsum(v[1:] * psi_e_values, dim=0)

    # Calculate terms
    terms = zeros(t, device=x.device)
    terms[1:] = (log_term + cumulative_v_psi) / cumulative_lambda

    return terms
