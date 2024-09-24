from time import time
from typing import List

from matplotlib.pyplot import figure, ylabel, plot, legend, grid, savefig, show, xlabel, title
from numpy import sort, append
from pandas import DataFrame, read_csv


def process_file(df: DataFrame):
    """Load and process the dataset to calculate certified accuracy for various radii."""
    # Filter rows where the prediction is correct
    df_correct = df[df['correct'] == 1]  # Filtering the dataframe to include only correct predictions

    # Sort by radius
    df_correct = df_correct.sort_values(by='radius')  # Sorting the filtered dataframe by the 'radius' column

    # Unique radius values in the dataset
    unique_radii = sort(df_correct['radius'].unique())  # Extracting unique radius values from the sorted dataframe

    # Calculate the certified accuracy at each unique radius
    certified_accuracies = []  # List to store certified accuracies
    for r in unique_radii:
        certified_count = len(df_correct[df_correct['radius'] >= r])  # Counting certified predictions for each radius
        certified_accuracy = certified_count / len(df)  # Calculating certified accuracy
        certified_accuracies.append(certified_accuracy)  # Appending certified accuracy to the list

    # Append a final point projecting the curve to the x-axis
    if len(unique_radii) > 0:
        last_r = unique_radii[-1]  # Extracting the last radius value
        unique_radii = append(unique_radii, last_r + 0.01)  # Adding a small increment to project the curve
        certified_accuracies.append(0)  # Appending 0 to certified accuracies for the final point

    return unique_radii, certified_accuracies  # Returning unique radii and certified accuracies


def plot_multiple_files(dfs: List[DataFrame], labels: List[str], title_name: str = None, save_path: str = None) -> None:
    """Plot certified accuracy curves from multiple files in the same figure."""
    figure(figsize=(12, 8))  # Creating a new figure with specified size

    for i, (df, label) in enumerate(zip(dfs, labels)):
        unique_radii, certified_accuracies = process_file(df)  # Processing each file

        plot(unique_radii, certified_accuracies, linestyle='-', label=label)  # Plotting certified accuracy curve

    xlabel('Radius (r)')  # Setting the label for x-axis
    ylabel('Certified Accuracy')  # Setting the label for y-axis
    # Setting the title of the plot
    title('Certified Accuracy in Terms of Radius') if not title_name else title(title_name)
    legend()  # Displaying legend
    grid(False)  # Removing the grid lines from the plot

    # Save the plot if save_path is provided
    if save_path:
        savefig(save_path)  # Saving the plot to the specified path
    else:
        show()  # Displaying the plot


def main():
    file_paths = ['/home/pc/PycharmProjects/continu/results2/cifar10_0.12_bernstein_second.csv',
                  '/home/pc/PycharmProjects/continu/results2/cifar10_0.12_bernstein_bonferroni_second.csv']
    dfs = [read_csv(file_path) for file_path in file_paths]  # Reading the CSV files
    labels = ['Ours', 'Bonferroni']
    plot_multiple_files(dfs, labels, title_name='CIFAR-10')


if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print(f"Total time: {end_time - start_time:.4f}" + " seconds")
