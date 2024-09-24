from logging import Logger
from math import floor

from numpy import clip
from scipy.stats import norm
from torch import Tensor, mean

from algorithm.test_k_first_classes import test_k_first_classes
from bernstein.calculate_term import calculate_term
from h5py_to_csv.calculate_lower_bound_bonferroni import calculate_lower_bound_bonferroni
from utils.get_basic_logger import get_basic_logger


def calculate_radius(predictions: Tensor, alpha: float, debug: bool, logger: Logger) -> float:
    if predictions.dim() != 2:
        raise ValueError("predictions must be a 2D tensor")
    if predictions.shape[1] < 4:
        raise ValueError("predictions must have at least 4 columns")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be strictly between 0 and 1")

    means = mean(predictions, dim=0)
    for k in range(1, floor(predictions.shape[1] / 2)):
        if test_k_first_classes(predictions=predictions, k=k, alpha=alpha, debug=debug, logger=logger):
            bernstein_func = lambda x: calculate_term(vector=x, alpha=alpha / (2 * (k + 1)))
            lower_bound = calculate_lower_bound_bonferroni(x=predictions[:, :k + 1], means=means[:k + 1], index=0,
                                                           term=bernstein_func, func=norm.ppf)
            if debug:
                logger.debug(f"final lower_bound {k}: {lower_bound}")
            return float(lower_bound)

    num_classes = predictions.shape[1]
    bernstein_func = lambda x: calculate_term(vector=x, alpha=alpha / num_classes)
    lower_bound = calculate_lower_bound_bonferroni(x=predictions, means=means, index=0, term=bernstein_func,
                                                   func=norm.ppf)
    if debug:
        logger.debug(f"lower_bound {num_classes}: {lower_bound}")
    return float(lower_bound)


def main():
    from algorithm.sample_dirichlet import sample_dirichlet
    from torch import set_printoptions
    # Example usage
    set_printoptions(threshold=20)
    alphas = (10.0, 8.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    num_samples = 10000

    samples = sample_dirichlet(alpha=alphas, num_samples=num_samples, seed=42)
    print(f"samples:\n{samples}")

    alpha = 0.001
    logger = get_basic_logger("test_k_first_classes")
    result = calculate_radius(predictions=samples, alpha=alpha, debug=True, logger=logger)

    print(f"My lower bound: {result}")

    # Baseline
    num_classes = len(samples[0])
    means = mean(samples, dim=0)
    bernstein_func = lambda x: calculate_term(vector=x, alpha=alpha / num_classes)
    lower_bound = calculate_lower_bound_bonferroni(x=samples, means=means, index=0, term=bernstein_func, func=norm.ppf)
    lower_bound = clip(lower_bound, 0, 1)
    print(f"Reference Lower bound: {lower_bound}")
    print(f"Gain: {(result - lower_bound) / lower_bound * 100:.2f}%")


if __name__ == "__main__":
    main()
