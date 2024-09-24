from torch import Tensor, cat, tensor


def select_columns(matrix: Tensor, k: int, include_col: int) -> Tensor:
    """
    Select columns from a 2D PyTorch matrix, excluding the first k columns
    but including a specific column from within those first k columns.

    Args:
    matrix (torch.Tensor): Input 2D matrix
    k (int): Number of columns to exclude from the beginning
    include_col (int): Index of the column to include from the first k columns

    Returns:
    torch.Tensor: A new matrix with the selected columns
    """
    if matrix.dim() != 2:
        raise ValueError("Input matrix must be 2-dimensional")

    if matrix.shape[1] <= 1:
        raise ValueError("Input matrix must have at least two columns")

    if k <= 0 or k >= matrix.shape[1] - 1:
        raise ValueError(f"k must be between 0 and {matrix.shape[1] - 2}")

    if include_col < 0 or include_col > k:
        raise ValueError(f"include_col must be between 0 and {k}")

    # Get all columns except the first k
    remaining_cols = matrix[:, k + 1:]

    # Get the specific column from the first k columns
    selected_col = matrix[:, include_col:include_col + 1]

    # Concatenate the selected column with the remaining columns
    result = cat([selected_col, remaining_cols], dim=1)

    return result


def main():
    # Create a sample 2D matrix
    t = tensor([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15]])

    print("Original matrix:")
    print(t)

    # Example 1: Select columns except the first 2, but include column 1 (0-indexed)
    k, include_col = 2, 2
    result = select_columns(t, k, include_col)
    print(f"\nSelected columns (k={k}, include_col={include_col}):")
    print(result)

    # Example 2: Select columns except the first 3, but include column 0
    k, include_col = 3, 0
    result = select_columns(t, k, include_col)
    print(f"\nSelected columns (k={k}, include_col={include_col}):")
    print(result)

    # Example 3: Demonstrate error handling
    try:
        k, include_col = 5, 0  # k is too large
        select_columns(t, k, include_col)
    except ValueError as e:
        print(f"\nValueError caught: {e}")

    try:
        k, include_col = 1, 5  # include_col is too large
        select_columns(t, k, include_col)
    except ValueError as e:
        print(f"\nValueError caught: {e}")


if __name__ == "__main__":
    main()
