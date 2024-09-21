from typing import Callable

from torch import Tensor, tensor


def apply_function(x: Tensor, func: Callable[[float], float]) -> Tensor:
    """
    Apply a given function element-wise to a PyTorch tensor.

    Args:
        x (torch.Tensor): The input tensor to which the function will be applied.
        func (Callable[[float], float]): A function that takes a float and returns a float.

    Returns:
        torch.Tensor: A new tensor with the function applied to each element.
    """
    # Apply the function to each element of the tensor manually
    return tensor([func(item) for item in x.flatten()], dtype=x.dtype).reshape(x.shape)


def main():
    """
    Main function to demonstrate the usage of apply_function.
    """

    # Define a simple function to apply: squaring each element
    def square(x: float) -> float:
        return x ** 2

    # Example 1: Applying the function to a 1D tensor
    input_tensor_1d = tensor([1.0, 2.0, 3.0, 4.0])
    output_tensor_1d = apply_function(input_tensor_1d, square)
    print("Input Tensor 1D:", input_tensor_1d)
    print("Output Tensor 1D:", output_tensor_1d)

    # Example 2: Applying the function to a 2D tensor (matrix)
    input_tensor_2d = tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    output_tensor_2d = apply_function(input_tensor_2d, square)
    print("Input Tensor 2D:", input_tensor_2d)
    print("Output Tensor 2D:", output_tensor_2d)


if __name__ == "__main__":
    main()
