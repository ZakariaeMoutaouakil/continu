from argparse import ArgumentParser
from json import dumps
from os.path import basename, splitext
from time import time
from typing import List, Dict

from colorama import init
from h5py import File
from numpy import clip
from torch import device, set_printoptions, from_numpy, mean
from torch.cuda import is_available as cuda_is_available
from torch.utils.data import DataLoader

from bernstein.calculate_term import calculate_term
from bernstein_cs.calculate_shift import calculate_shift
from cohen2019.datasets import get_dataset
from h5py_to_csv.analyze_hdf5_dataset import analyze_hdf5_dataset
from h5py_to_csv.row_diff_from_max import row_diff_from_max
from h5py_to_csv.softmax_with_temperature import softmax_with_temperature
from taylor.gaussian_quantile_approximation import gaussian_quantile_approximation
from utils.logging_config import basic_logger


def main() -> None:
    parser = ArgumentParser(description='Transform dataset')
    parser.add_argument("--temperature", type=float, required=True, help="softmax temperature")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, help="number of samples to use", required=True)
    parser.add_argument("--order", type=int, help="approximation order", default=15)
    parser.add_argument("--outdir", type=str, help="output directory", required=True)
    parser.add_argument("--c", type=float, help="c", default=0.75)
    args = parser.parse_args()

    args_dict = vars(args)

    # Pretty print the dictionary with json.dumps
    formatted_args = dumps(args_dict, indent=4)

    set_printoptions(threshold=20)

    # Initialize colorama
    init(autoreset=True)

    torch_device = device('cuda' if cuda_is_available() else 'cpu')
    dataset_title: str = splitext(basename(args.dataset))[0]
    dataset_type: str = dataset_title.split('_')[0]
    test_dataset = get_dataset(dataset_type)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=args.num_workers)

    dataset_info = analyze_hdf5_dataset(args.dataset)
    last_nonzero_index = dataset_info['last_nonzero_index']
    min_inference_count = dataset_info['min_inference_count']

    if args.num_samples > min_inference_count:
        raise ValueError(f"num_samples ({args.num_samples}) > min_inference_count ({min_inference_count})")

    if last_nonzero_index == -1:
        raise ValueError(f"no non-zero examples found in {args.dataset}")

    dataset = File(args.dataset, 'r')

    logger = basic_logger(args.outdir + '/' + dataset_title + '.log')

    # Log the formatted arguments
    logger.info(formatted_args)
    logger.info(f"torch_device: {torch_device}")
    logger.debug(f"last_nonzero_index: {last_nonzero_index}")
    logger.debug(f"min_inference_count: {min_inference_count}")
    logger.debug(f"dataset_title: {dataset_title}")
    logger.debug(f"dataset_type: {dataset_type}")

    results_bernstein_first: List[Dict[str, str | float]] = []
    results_bernstein_bonferroni_first: List[Dict[str, str | float]] = []
    results_sequence_first: List[Dict[str, str | float]] = []
    results_sequence_bonferroni_first: List[Dict[str, str | float]] = []

    results_bernstein_second: List[Dict[str, str | float]] = []
    results_bernstein_bonferroni_second: List[Dict[str, str | float]] = []
    results_sequence_second: List[Dict[str, str | float]] = []
    results_sequence_bonferroni_second: List[Dict[str, str | float]] = []

    quantile_function = gaussian_quantile_approximation(args.order)
    maximum = quantile_function(1) - quantile_function(0)

    global_time = time()

    for i, (_, label) in enumerate(test_loader):
        if i >= last_nonzero_index:
            break

        label = label.item()
        logits = from_numpy(dataset[f"{dataset_title}_predictions"][i, :args.num_samples]).to(torch_device)
        logger.debug(f"Example {i} - label: {label} - logits: {logits}")
        predictions = softmax_with_temperature(logits=logits, temperature=args.temperature)
        logger.debug(f"predictions: {predictions}")
        means = mean(predictions, dim=0)
        logger.debug(f"means: {means}")

        predicted = means.argmax().item()
        logger.debug(f"predicted: {predicted}")
        correct = int(predicted == label)
        logger.debug(f"correct: {correct}")

        predicted_tensor = predictions[:, predicted]
        logger.debug(f"predicted_tensor: {predicted_tensor}")

        ### First Radius
        logger.info(f"First Radius")
        ## Ours
        difference_tensor = row_diff_from_max(matrix=predictions, index=predicted)
        logger.debug(f"difference_tensor: {difference_tensor}")
        normalized_difference_tensor = (difference_tensor + 1) / 2
        logger.debug(f"normalized_difference_tensor: {normalized_difference_tensor}")
        # Bernstein
        start_time = time()
        normalized_bernstein_term = calculate_term(vector=normalized_difference_tensor, alpha=args.alpha)
        normalized_bernstein_lb = mean(normalized_difference_tensor).item() - normalized_bernstein_term
        normalized_bernstein_lb = clip(normalized_bernstein_lb, 0, 1)
        bernstein_lower_bound = 2 * normalized_bernstein_lb - 1
        end_time = time()
        results_bernstein_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., bernstein_lower_bound),
            'time': f"{end_time - start_time:.4f}"
        })
        logger.info("First Radius + Ours + Bernstein")
        logger.debug(results_bernstein_first[-1])
        ## CS
        start_time = time()
        normalized_weighted_mean, normalized_cs_term = calculate_shift(x=normalized_difference_tensor, alpha=args.alpha,
                                                                       c=args.c)
        normalized_cs_lb = normalized_weighted_mean - normalized_cs_term
        normalized_cs_lb = clip(normalized_cs_lb, 0, 1)
        cs_lower_bound = 2 * normalized_cs_lb - 1
        end_time = time()
        results_sequence_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., cs_lower_bound),
            'time': f"{end_time - start_time:.4f}"
        })
        logger.info("First Radius + CS")
        logger.debug(results_sequence_first[-1])
        ## Bonferroni
        # Bernstein


if __name__ == "__main__":
    main()
