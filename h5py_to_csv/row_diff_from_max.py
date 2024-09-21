from torch import rand, Tensor, bool, max, manual_seed, cat, ones


def row_diff_from_max(matrix: Tensor, index: int) -> Tensor:
    """
    Compute the difference between the element at the specified index and the maximum
    of all other elements for each row in the matrix.

    Args:
    matrix (torch.Tensor): A 2D tensor (matrix) of shape (n_rows, n_cols).
    index (int): The index of the element to compare against the max of others.

    Returns:
    torch.Tensor: A 1D tensor of length n_rows, where each element is the difference
                  between the element at the specified index and the max of other elements
                  in the corresponding row.
    """
    if matrix.dim() != 2:
        raise ValueError("Input must be a 2D tensor (matrix)")

    if index < 0 or index >= matrix.shape[1]:
        raise ValueError(f"Index must be between 0 and {matrix.shape[1] - 1}")

    # Create a boolean mask for all columns except the specified index
    mask = ones(matrix.shape[1], dtype=bool, device=matrix.device)
    mask[index] = False

    # Compute the maximum of other elements for each row
    max_others = max(matrix[:, mask], dim=1).values

    # Compute the difference
    result = matrix[:, index] - max_others

    return result


def main():
    # Example usage
    manual_seed(0)  # For reproducibility

    # Create a sample 2D tensor (matrix)
    matrix = rand(5, 4)  # 5 rows, 4 columns
    index = 2  # We'll compare the third column (index 2) with the max of others

    print("Original matrix:")
    print(matrix)

    result = row_diff_from_max(matrix, index)

    print(f"\nDifference between column {index} and max of others for each row:")
    print(result)

    # Verification
    print("\nVerification:")
    for i, row in enumerate(matrix):
        others_max = max(cat([row[:index], row[index + 1:]]))
        expected_diff = row[index] - others_max
        print(f"Row {i}: Expected {expected_diff:.4f}, Got {result[i]:.4f}")


if __name__ == "__main__":
    main()
