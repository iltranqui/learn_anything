from rich.console import Console
import numpy as np
import random
console = Console()

import time
from algorithms.sorting_algo import bubble_sort, merge_sort, quick_sort, insertion_sort, selection_sort, bucket_sort, counting_sort
from rich.progress import Progress

# Step 1: Load random_names.txt file
def load_names(file_path):
    with open(file_path, "r") as file:
        names = file.read().splitlines()  # Read each line as an element in a list
    return np.array(names)  # Convert list to numpy array


def generate_random_array(n: int = 100, lower_bound: int = 1, upper_bound: int = 1000):
    """
    Generate a random array of n elements.

    Parameters:
    n (int): The number of elements in the array.
    lower_bound (int): The minimum value of the random integers (inclusive).
    upper_bound (int): The maximum value of the random integers (inclusive).

    Returns:
    list: A list containing n random integers.
    """
    assert lower_bound <= upper_bound, "Lower bound must be less than or equal to upper bound."
    assert n > 0, "Number of elements must be greater than 0."
    assert isinstance(n, int), "Number of elements must be an integer."
    assert lower_bound >= 0 , "Lower bound must be a positive integer."
    return [random.randint(lower_bound, upper_bound) for _ in range(n)]

# Example usage
random_array = generate_random_array(7, lower_bound=10, upper_bound=100)


# Timing wrapper function to measure execution time
def time_operation(operation, *args):
    start_time = time.time()
    result = operation(*args)
    end_time = time.time()
    assert result == sorted(result), "The array is not sorted correctly."
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    return result, elapsed_time

import matplotlib.pyplot as plt

def time_operation_at_scale(n_values, lower_bound=1, upper_bound=5000):
    results = {}
    with Progress() as progress:
        task = progress.add_task("[green]Processing...", total=len(range(1000, 10001, 1000)))
        for n in range(1000, 10001, 1000):
            progress.update(task, description=f"[green]Processing array of {n} elements...")
            random_array = generate_random_array(n, lower_bound, upper_bound)
            _, bubble_sort_time = time_operation(bubble_sort, random_array.copy())
            _, merge_sort_time = time_operation(merge_sort, random_array.copy())
            _, quick_sort_time = time_operation(quick_sort, random_array.copy())
            _, insertion_sort_time = time_operation(insertion_sort, random_array.copy())
            _, selection_sort_time = time_operation(selection_sort, random_array.copy())
            _, bucket_sort_time = time_operation(bucket_sort, random_array.copy())
            _, counting_sort_time = time_operation(counting_sort, random_array.copy())
            results[n] = {
                "bubble_sort": bubble_sort_time,
                "merge_sort": merge_sort_time,
                "quick_sort": quick_sort_time,
                "insertion_sort": insertion_sort_time,
                "selection_sort": selection_sort_time,
                "bucket_sort": bucket_sort_time,
                "counting_sort": counting_sort_time
            }
            progress.advance(task)
    
    # Plotting the results
    n_values = sorted(results.keys())
    bubble_sort_times = [results[n]["bubble_sort"] for n in n_values]
    merge_sort_times = [results[n]["merge_sort"] for n in n_values]
    quick_sort_times = [results[n]["quick_sort"] for n in n_values]
    insertion_sort_times = [results[n]["insertion_sort"] for n in n_values]
    selection_sort_times = [results[n]["selection_sort"] for n in n_values]
    bucket_sort_times = [results[n]["bucket_sort"] for n in n_values]
    counting_sort_times = [results[n]["counting_sort"] for n in n_values]

    plt.figure(figsize=(10, 6))

    # Define time complexity functions for illustration purposes
    O_1 = np.ones_like(n_values)                  # O(1)
    O_log_n = np.log(n_values)                    # O(log n)
    O_n = np.asarray(n_values)                                # O(n)
    O_n_log_n = np.asarray(n_values) * np.log(n_values)       # O(n log n)
    O_n2 = np.asarray(n_values) ** 2                          # O(n^2)
    O_2n = 2 ** np.asarray(n_values)                          # O(2^n)
    # Limit factorial computation to avoid overflow (use n_values[:10] for smaller values)
    O_log_factorial = [n * np.log(n) - n if n > 0 else 0 for n in n_values[:20]]  # For first 20 elements only

    # Fill regions with colors corresponding to different complexities
    plt.fill_between(n_values, O_1, O_n_log_n, color='green', alpha=0.3, label='O(1), O(log n)')
    plt.fill_between(n_values, O_n_log_n, O_n, color='yellow', alpha=0.3, label='O(n log n)')
    plt.fill_between(n_values, O_n, O_n2, color='orange', alpha=0.3, label='O(n)')
    plt.fill_between(n_values, O_n2, O_2n, color='red', alpha=0.3, label='O(n^2)')
    plt.fill_between(n_values, O_2n, O_log_factorial, color='darkred', alpha=0.3, label='O(2^n), O(n!)')

    plt.plot(n_values, bubble_sort_times, label='Bubble Sort', marker='o')
    plt.plot(n_values, merge_sort_times, label='Merge Sort', marker='o')
    plt.plot(n_values, quick_sort_times, label='Quick Sort', marker='o')
    plt.plot(n_values, insertion_sort_times, label='Insertion Sort', marker='o')
    plt.plot(n_values, selection_sort_times, label='Selection Sort', marker='o')
    plt.plot(n_values, bucket_sort_times, label='Bucket Sort', marker='o')
    plt.plot(n_values, counting_sort_times, label='Counting Sort', marker='o')

    plt.xlabel('Number of Elements (n)')
    plt.ylabel('Sorting Time (ms)')
    plt.ylim(0, 600)  # Set y-axis limit to 2000 ms
    # plt.yscale('log')  # Set y-axis to log scale -> for log scale, simply uncomment this line
    plt.title('Sorting Algorithm Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('sorting_performance.png')
    plt.show()

    return results

def main():
    file_path = 'random_names.txt'  # Path to your names file
    list = load_names(file_path=file_path)
    console.print(f"Loaded {len(list)} names from {file_path} of type {type(list)}")
    console.print(f"TO DO: Implement the bubble sort algorithm to sort the names")

    """
    Main function to run the sorting algorithm
    """
    random_array = generate_random_array(10, lower_bound=1, upper_bound=100)

    _, bubble_sort_time = time_operation(bubble_sort, random_array)
    _ , merge_sort_time = time_operation(merge_sort, random_array)
    _, quick_sort_time = time_operation(quick_sort, random_array)
    _, insertion_sort_time = time_operation(insertion_sort, random_array)
    console.print(f"Bubble Sort: {bubble_sort_time} ms", style="bold red")
    console.print(f"Merge Sort: {merge_sort_time} ms" , style="bold green")
    console.print(f"Quick Sort: {quick_sort_time} ms",  style="bold blue")
    console.print(f"Insertion Sort: {insertion_sort_time} ms", style="bold yellow")

    # Time the sorting algorithms at scale
    n_values = range(1000, 10001, 1000)
    results = time_operation_at_scale(n_values)

if __name__ == "__main__":
    main()
