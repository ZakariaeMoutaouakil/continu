from torch import Tensor, tensor


def repeat_matrix_vertically(matrix: Tensor, repetitions: int) -> Tensor:
    """
    Repeat a 2D PyTorch tensor vertically (along dim=0) a specified number of times.

    Args:
    matrix (torch.Tensor): Input 2D PyTorch tensor
    repetitions (int): Number of times to repeat the matrix vertically

    Returns:
    torch.Tensor: Vertically repeated matrix

    Raises:
    ValueError: If input is not a 2D tensor or if repetitions is less than 1
    """
    if not isinstance(matrix, Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    if matrix.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional")

    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("Repetitions must be a positive integer")

    return matrix.repeat(repetitions, 1)


def main():
    # Example usage
    # Create a 2x3 matrix
    matrix = tensor([[1, 2, 3],
                     [4, 5, 6]])

    try:
        # Repeat the matrix 3 times vertically
        repeated_matrix = repeat_matrix_vertically(matrix, 3)
        print("Original matrix:")
        print(matrix)
        print("\nRepeated matrix:")
        print(repeated_matrix)

        # Try with invalid inputs
        # repeat_matrix_vertically(torch.tensor([1, 2, 3]), 2)  # 1D tensor
        # repeat_matrix_vertically(matrix, 0)  # Invalid repetitions
        # repeat_matrix_vertically(matrix, 2.5)  # Non-integer repetitions

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
