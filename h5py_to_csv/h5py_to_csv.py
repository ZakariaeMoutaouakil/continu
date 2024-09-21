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
from torch import device, set_printoptions, from_numpy, mean, tensor
from torch.cuda import is_available as cuda_is_available
from torch.utils.data import DataLoader

from apply_elementwise import apply_function
from bernstein.calculate_term import calculate_term
from bernstein_cs.calculate_shift import calculate_shift
from cohen2019.datasets import get_dataset, get_num_classes
from taylor.gaussian_quantile_approximation import gaussian_quantile_approximation
from utils.logging_config import basic_logger
from .analyze_hdf5_dataset import analyze_hdf5_dataset
from .calculate_lower_bound_bonferroni import calculate_lower_bound_bonferroni
from .row_diff_from_max import row_diff_from_max
from .softmax_with_temperature import softmax_with_temperature


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
    num_classes = get_num_classes(dataset_type)

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
    logger.debug(f"num_classes: {num_classes}")

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
        results_bernstein_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., bernstein_lower_bound),
            'time': f"{time() - start_time:.4f}"
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
        results_sequence_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., cs_lower_bound),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("First Radius + Ours + CS")
        logger.debug(results_sequence_first[-1])
        ## Bonferroni
        # Bernstein
        start_time = time()
        bernstein_func = lambda x: calculate_term(vector=x, alpha=args.alpha / num_classes)
        func = lambda x: x
        normalized_bernstein_bonferroni_lb = calculate_lower_bound_bonferroni(x=normalized_difference_tensor,
                                                                              means=means, index=predicted,
                                                                              term=bernstein_func, func=func)
        normalized_bernstein_bonferroni_lb = clip(normalized_bernstein_bonferroni_lb, 0, 1)
        bernstein_bonferroni_lb = 2 * normalized_bernstein_bonferroni_lb - 1
        results_bernstein_bonferroni_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., bernstein_bonferroni_lb),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("First Radius + Bonferroni + Bernstein")
        logger.debug(results_bernstein_bonferroni_first[-1])
        # CS
        start_time = time()
        weighted_means = tensor([calculate_shift(x=predictions[:, i], alpha=args.alpha / num_classes, c=args.c)[0]
                                 for i in range(num_classes)], device=torch_device)
        cs_func = lambda x: calculate_shift(x=x, alpha=args.alpha / num_classes, c=args.c)[1]
        normalized_cs_bonferroni_lb = calculate_lower_bound_bonferroni(x=normalized_difference_tensor,
                                                                       means=weighted_means, index=predicted,
                                                                       term=cs_func, func=func)
        normalized_cs_bonferroni_lb = clip(normalized_cs_bonferroni_lb, 0, 1)
        cs_bonferroni_lb = 2 * normalized_cs_bonferroni_lb - 1
        results_sequence_bonferroni_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., cs_bonferroni_lb),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("First Radius + Bonferroni + CS")
        logger.debug(results_sequence_bonferroni_first[-1])

        ### Second Radius
        logger.info("Second Radius")
        ## Ours
        predictions_quantiles = apply_function(x=predictions, func=quantile_function)
        logger.debug(f"predictions quantiles: {predictions_quantiles}")
        quantile_difference_tensor = row_diff_from_max(matrix=predictions_quantiles, index=predicted)
        logger.debug(f"quantile_difference_tensor: {quantile_difference_tensor}")
        normalized_quantile_difference_tensor = quantile_difference_tensor / maximum
        logger.debug(f"normalized_quantile_difference_tensor: {normalized_quantile_difference_tensor}")
        # Bernstein
        start_time = time()
        normalized_bern_term = calculate_term(vector=normalized_quantile_difference_tensor, alpha=args.alpha)
        normalized_bernstein_lb = mean(normalized_quantile_difference_tensor).item() - normalized_bern_term
        normalized_bern_term = clip(normalized_bern_term, 0, 1)
        bern_lb = normalized_bern_term * maximum
        results_bernstein_second.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., bern_lb),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("Second Radius + Ours + Bernstein")
        logger.debug(results_bernstein_second[-1])
        # CS
        start_time = time()
        normalized_weighted_mean, normalized_cs_term = calculate_shift(x=normalized_quantile_difference_tensor,
                                                                       alpha=args.alpha, c=args.c)
        normalized_cs_lb = normalized_weighted_mean - normalized_cs_term
        normalized_cs_lb = clip(normalized_cs_lb, 0, 1)
        cs_lb = normalized_cs_lb * maximum
        results_sequence_second.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., cs_lb),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("Second Radius + Ours + CS")
        logger.debug(results_sequence_second[-1])
        ## Bonferroni
        # Bernstein
        start_time = time()
        normalized_bernstein_bonferroni_lb = calculate_lower_bound_bonferroni(x=normalized_difference_tensor,
                                                                              means=means, index=predicted,
                                                                              term=bernstein_func, func=norm.ppf)
        normalized_bernstein_bonferroni_lb = clip(normalized_bernstein_bonferroni_lb, 0, 1)
        bernstein_bonferroni_lb = normalized_bernstein_bonferroni_lb * maximum
        results_bernstein_second.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., bernstein_bonferroni_lb),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("Second Radius + Bonferroni + Bernstein")
        logger.debug(results_bernstein_second[-1])
        # CS
        start_time = time()
        weighted_means = tensor([calculate_shift(x=predictions[:, i], alpha=args.alpha / num_classes, c=args.c)[0]
                                 for i in range(num_classes)], device=torch_device)
        cs_func = lambda x: calculate_shift(x=x, alpha=args.alpha / num_classes, c=args.c)[1]
        normalized_cs_bonferroni_lb = calculate_lower_bound_bonferroni(x=normalized_difference_tensor,
                                                                       means=weighted_means, index=predicted,
                                                                       term=cs_func, func=norm.ppf)
        normalized_cs_bonferroni_lb = clip(normalized_cs_bonferroni_lb, 0, 1)
        cs_bonferroni_lb = normalized_cs_bonferroni_lb * maximum
        results_sequence_bonferroni_second.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., cs_bonferroni_lb),
            'time': f"{time() - start_time:.4f}"
        })
        logger.info("Second + Radius + Bonferroni + CS")
        logger.debug(results_sequence_bonferroni_second[-1])

    df_bernstein_first = DataFrame(results_bernstein_first)
    df_bernstein_bonferroni_first = DataFrame(results_bernstein_bonferroni_first)
    df_sequence_first = DataFrame(results_sequence_first)
    df_sequence_bonferroni_first = DataFrame(results_sequence_bonferroni_first)

    df_bernstein_second = DataFrame(results_bernstein_second)
    df_bernstein_bonferroni_second = DataFrame(results_bernstein_bonferroni_second)
    df_sequence_second = DataFrame(results_sequence_second)
    df_sequence_bonferroni_second = DataFrame(results_sequence_bonferroni_second)

    df_bernstein_first.to_csv(args.outdir + '/' + dataset_title + '_bernstein_first.csv', index=False)
    df_bernstein_bonferroni_first.to_csv(args.outdir + '/' + dataset_title + '_bernstein_bonferroni_first.csv',
                                         index=False)
    df_sequence_first.to_csv(args.outdir + '/' + dataset_title + '_sequence_first.csv', index=False)
    df_sequence_bonferroni_first.to_csv(args.outdir + '/' + dataset_title + '_sequence_bonferroni_first.csv',
                                        index=False)

    df_bernstein_second.to_csv(args.outdir + '/' + dataset_title + '_bernstein_second.csv', index=False)
    df_bernstein_bonferroni_second.to_csv(
        args.outdir + '/' + dataset_title + '_bernstein_bonferroni_second.csv',
        index=False)
    df_sequence_second.to_csv(args.outdir + '/' + dataset_title + '_sequence_second.csv', index=False)
    df_sequence_bonferroni_second.to_csv(args.outdir + '/' + dataset_title + '_sequence_bonferroni_second.csv',
                                         index=False)

    logger.info(f"Saved results to {args.outdir} directory")
    logger.info("Done in {:.4f}s".format(time() - global_time))


if __name__ == "__main__":
    main()
