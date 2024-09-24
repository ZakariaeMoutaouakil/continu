from logging import Logger
from math import floor
from typing import Optional

from numpy import clip
from torch import Tensor, mean

from algorithm.select_columns import select_columns
from bernstein.calculate_term import calculate_term
from h5py_to_csv.row_diff_from_max import row_diff_from_max
from utils.get_basic_logger import get_basic_logger


def test_k_first_classes(predictions: Tensor, k: int, alpha: float, debug: bool, logger: Optional[Logger]) -> bool:
    if predictions.dim() != 2:
        raise ValueError("predictions must be a 2D tensor")
    if predictions.shape[1] < 4:
        raise ValueError("predictions must have at least 4 columns")
    if not 1 <= k <= floor(predictions.shape[1] / 2) - 1:
        raise ValueError("k must be between 2 and floor(n_classes / 2)")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be strictly between 0 and 1")

    for i in range(1, k + 1):
        sub_matrix = select_columns(matrix=predictions, k=k, include_col=i)
        if debug:
            logger.debug(f"sub_matrix {i}: {sub_matrix}")
        difference_tensor = row_diff_from_max(matrix=sub_matrix, index=0)
        if debug:
            logger.debug(f"difference_tensor {i}: {difference_tensor}")
        normalized_difference_tensor = difference_tensor + 1 / 2
        if debug:
            logger.debug(f"normalized_difference_tensor {i}: {normalized_difference_tensor}")
        normalized_lower_bound = (mean(normalized_difference_tensor).item() -
                                  calculate_term(vector=normalized_difference_tensor, alpha=alpha / (2 * k)))
        normalized_lower_bound = float(clip(normalized_lower_bound, 0, 1))
        if debug:
            logger.debug(f"normalized_lower_bound {i}: {normalized_lower_bound}")
        lower_bound = normalized_lower_bound - 1 / 2
        if debug:
            logger.debug(f"lower_bound {i}: {lower_bound}")
        if lower_bound >= 0:
            return True

    return False


def main():
    from algorithm.sample_dirichlet import sample_dirichlet
    from torch import set_printoptions
    # Example usage
    set_printoptions(threshold=20)
    alphas = (10.0, 8.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    num_samples = 10000

    samples = sample_dirichlet(alpha=alphas, num_samples=num_samples, seed=42)
    print(f"samples: {samples}")

    k = 2
    alpha = 0.001
    logger = get_basic_logger("test_k_first_classes")
    result = test_k_first_classes(predictions=samples, k=k, alpha=alpha, debug=True, logger=logger)
    logger.debug(f"result: {result}")



if __name__ == "__main__":
    main()
