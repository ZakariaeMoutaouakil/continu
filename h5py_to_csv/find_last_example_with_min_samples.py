from pathlib import Path
from typing import Union

from h5py import File
from numpy import where


def find_last_example_with_min_samples(file_path: Union[str, Path], dataset_title: str, min_samples: int) -> int:
    """
    Find the index of the last example in the HDF5 file whose count is at least equal to the specified number of samples.

    Args:
    file_path (Union[str, Path]): Path to the HDF5 file.
    dataset_title (str): Title of the dataset within the HDF5 file.
    min_samples (int): Minimum number of samples required.

    Returns:
    int: Index of the last example meeting the criteria, or -1 if no example is found.

    Raises:
    ValueError: If the file doesn't exist, if the dataset is not found, or if min_samples is not positive.
    """
    if min_samples <= 0:
        raise ValueError("min_samples must be a positive integer")

    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    try:
        with File(file_path, 'r') as f:
            counts_dataset = f[f"{dataset_title}_counts"]
            counts = counts_dataset[:]

            # Find indices where count is at least min_samples
            valid_indices = where(counts >= min_samples)[0]

            if len(valid_indices) == 0:
                return -1
            else:
                return valid_indices[-1]
    except KeyError:
        raise ValueError(f"Dataset '{dataset_title}_counts' not found in the file")


def main():
    # Example usage
    file_path = "/home/pc/PycharmProjects/cifar10_0.12.h5"
    dataset_title = "cifar10_0.12"
    min_samples = 10000

    try:
        result = find_last_example_with_min_samples(file_path, dataset_title, min_samples)
        if result == -1:
            print(f"No example found with at least {min_samples} samples.")
        else:
            print(f"Last example index with at least {min_samples} samples: {result}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
