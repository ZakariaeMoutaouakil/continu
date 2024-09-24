from argparse import ArgumentParser
from json import dumps
from os.path import basename, splitext
from time import time
from typing import List, Dict

from colorama import init
from h5py import File
from numpy import clip
from pandas import DataFrame
from scipy.stats import norm
from torch import device, set_printoptions, from_numpy, mean
from torch.cuda import is_available as cuda_is_available
from torch.utils.data import DataLoader

from algorithm.calculate_radius import calculate_radius
from bernstein.calculate_term import calculate_term
from cohen2019.datasets import get_dataset, get_num_classes
from h5py_to_csv.reorder_columns_by_mean import reorder_columns_by_mean
from h5py_to_csv.repeat_matrix_vertically import repeat_matrix_vertically
from utils.logging_config import basic_logger
from .analyze_hdf5_dataset import analyze_hdf5_dataset
from .calculate_lower_bound_bonferroni import calculate_lower_bound_bonferroni
from .row_diff_from_max import row_diff_from_max
from .softmax_with_temperature import softmax_with_temperature


def seconds_to_minutes_seconds(seconds: float):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return minutes, remaining_seconds


def main() -> None:
    parser = ArgumentParser(description='Transform dataset')
    parser.add_argument("--temperature", type=float, required=True, help="softmax temperature")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, help="number of samples to use", required=True)
    parser.add_argument("--outdir", type=str, help="output directory", required=True)
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
    num_classes = get_num_classes(dataset_type)

    dataset_info = analyze_hdf5_dataset(args.dataset)
    last_nonzero_index = dataset_info['last_nonzero_index']
    min_inference_count = dataset_info['min_inference_count']

    if last_nonzero_index == -1:
        raise ValueError(f"no non-zero examples found in {args.dataset}")

    if args.num_samples > min_inference_count:
        raise ValueError(f"num_samples ({args.num_samples}) > min_inference_count ({min_inference_count})")

    dataset = File(args.dataset, 'r')

    logger = basic_logger(args.outdir + '/' + dataset_title + '.log')

    # Log the formatted arguments
    logger.info(formatted_args)
    logger.info(f"torch_device: {torch_device}")
    logger.debug(f"last_nonzero_index: {last_nonzero_index}")
    logger.debug(f"min_inference_count: {min_inference_count}")
    logger.debug(f"dataset_title: {dataset_title}")
    logger.debug(f"dataset_type: {dataset_type}")
    logger.debug(f"num_classes: {num_classes}")

    results_bernstein_first: List[Dict[str, str | float]] = []
    results_bernstein_bonferroni_first: List[Dict[str, str | float]] = []

    results_bernstein_second: List[Dict[str, str | float]] = []
    results_bernstein_bonferroni_second: List[Dict[str, str | float]] = []

    global_time = time()

    for i, (_, label) in enumerate(test_loader):
        if i > last_nonzero_index:
            break

        label = label.item()
        logits = from_numpy(dataset[f"{dataset_title}_predictions"][i, :args.num_samples]).to(torch_device)
        logger.debug(f"Example {i} - label: {label} - logits: {logits}")
        predictions = softmax_with_temperature(logits=logits, temperature=args.temperature)
        logger.debug(f"predictions: {predictions}")
        predictions = repeat_matrix_vertically(matrix=predictions, repetitions=10)
        means = mean(predictions, dim=0)
        logger.debug(f"means: {means}")

        predicted = means.argmax().item()
        logger.debug(f"predicted: {predicted}")
        correct = int(predicted == label)
        logger.debug(f"correct: {correct}")

        predicted_tensor = predictions[:, predicted]
        logger.debug(f"predicted_tensor: {predicted_tensor}")

        ordered_predictions, _ = reorder_columns_by_mean(matrix=predictions)
        logger.debug(f"ordered_predictions: {ordered_predictions}")

        ### First Radius
        logger.info(f"First Radius")
        ## Bonferroni
        # Bernstein
        start_time = time()
        bernstein_func = lambda x: calculate_term(vector=x, alpha=args.alpha / num_classes)
        func = lambda x: x
        bernstein_bonferroni_lb = calculate_lower_bound_bonferroni(x=predictions, means=means, index=predicted,
                                                                   term=bernstein_func, func=func)
        bernstein_bonferroni_lb = clip(bernstein_bonferroni_lb, 0, 1)
        results_bernstein_bonferroni_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., float(bernstein_bonferroni_lb)),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("First Radius + Bonferroni + Bernstein")
        logger.debug(results_bernstein_bonferroni_first[-1])
        ## Ours
        difference_tensor = row_diff_from_max(matrix=predictions, index=predicted)
        logger.debug(f"difference_tensor: {difference_tensor}")
        normalized_difference_tensor = (difference_tensor + 1) / 2
        logger.debug(f"normalized_difference_tensor: {normalized_difference_tensor}")
        # Bernstein
        start_time = time()
        normalized_bernstein_lb = (mean(normalized_difference_tensor).item()
                                   - calculate_term(vector=normalized_difference_tensor, alpha=args.alpha))
        normalized_bernstein_lb = clip(normalized_bernstein_lb, 0, 1)
        bernstein_lower_bound = 2 * normalized_bernstein_lb - 1
        results_bernstein_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., float(bernstein_lower_bound)),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("First Radius + Ours + Bernstein")
        logger.debug(results_bernstein_first[-1])

        ### Second Radius
        logger.info("Second Radius")
        ## Bonferroni
        # Bernstein
        start_time = time()
        bernstein_bonferroni_lb = calculate_lower_bound_bonferroni(x=predictions, means=means, index=predicted,
                                                                   term=bernstein_func, func=norm.ppf)
        results_bernstein_bonferroni_second.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., bernstein_bonferroni_lb),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("Second Radius + Bonferroni + Bernstein")
        logger.debug(results_bernstein_bonferroni_second[-1])
        ## Ours
        start_time = time()
        bernstein_lb = calculate_radius(predictions=ordered_predictions, alpha=args.alpha, debug=True, logger=logger)
        results_bernstein_second.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., bernstein_lb),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("Second Radius + Ours + Bernstein")
        logger.debug(results_bernstein_second[-1])

    df_bernstein_first = DataFrame(results_bernstein_first)
    df_bernstein_bonferroni_first = DataFrame(results_bernstein_bonferroni_first)

    df_bernstein_second = DataFrame(results_bernstein_second)
    df_bernstein_bonferroni_second = DataFrame(results_bernstein_bonferroni_second)

    df_bernstein_first.to_csv(args.outdir + '/' + dataset_title + '_bernstein_first.csv', index=False)
    df_bernstein_bonferroni_first.to_csv(args.outdir + '/' + dataset_title + '_bernstein_bonferroni_first.csv',
                                         index=False)

    df_bernstein_second.to_csv(args.outdir + '/' + dataset_title + '_bernstein_second.csv', index=False)
    df_bernstein_bonferroni_second.to_csv(
        args.outdir + '/' + dataset_title + '_bernstein_bonferroni_second.csv',
        index=False)

    logger.info(f"Saved results to {args.outdir} directory")
    final_time = seconds_to_minutes_seconds(time() - global_time)
    logger.info(f"Done in {final_time[0]} minutes and {final_time[1]:.0f} seconds")


if __name__ == "__main__":
    main()
