from torch import rand, Tensor, mean, manual_seed, sort
from typing import Tuple


def reorder_columns_by_mean(matrix: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Reorders the columns of a 2D PyTorch tensor based on their means in decreasing order.

    Args:
    matrix (torch.Tensor): A 2D PyTorch tensor.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: A tuple containing the reordered matrix and the indices used for reordering.

    Raises:
    ValueError: If the input is not a 2D tensor.
    """
    if matrix.dim() != 2:
        raise ValueError("Input must be a 2D tensor.")

    # Calculate column means
    col_means = mean(matrix, dim=0)

    # Get indices that would sort the means in descending order
    _, sorted_indices = sort(col_means, descending=True)

    # Use the sorted indices to reorder the columns
    reordered_matrix = matrix[:, sorted_indices]

    return reordered_matrix, sorted_indices


def main():
    # Example usage
    manual_seed(42)  # For reproducibility

    # Create a random 3x4 matrix
    matrix = rand(3, 4)
    print("Original matrix:")
    print(matrix)
    print("\nOriginal column means:")
    print(mean(matrix, dim=0))

    try:
        reordered_matrix, indices = reorder_columns_by_mean(matrix)
        print("\nReordered matrix:")
        print(reordered_matrix)
        print("\nReordered column means:")
        print(mean(reordered_matrix, dim=0))
        print("\nIndices used for reordering:")
        print(indices)
    except ValueError as e:
        print(f"Error: {e}")

    # Example with invalid input
    invalid_matrix = rand(2, 3, 4)
    try:
        reorder_columns_by_mean(invalid_matrix)
    except ValueError as e:
        print(f"\nError with invalid input: {e}")


if __name__ == "__main__":
    main()