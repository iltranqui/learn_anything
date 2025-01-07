# Verify simple scipt to compute the size of a numpy matrix
import torch
import numpy as np

def compute_matrix_size_numpy(matrix):
    """
    Compute and display the size of a numpy matrix.

    Args:
        matrix (numpy.ndarray): The matrix for which to compute the size.

    Returns:
        dict: A dictionary containing the number of elements, shape, and memory size.
    """
    # Get the number of elements
    num_elements = matrix.size

    # Get the shape of the matrix
    shape = matrix.shape

    # Compute memory size in bytes
    memory_size_bytes = matrix.nbytes

    # Convert to kilobytes and megabytes for readability
    memory_size_kb = memory_size_bytes / 1024
    memory_size_mb = memory_size_kb / 1024

    # Print results
    print(f"Matrix Shape: {shape}")
    print(f"Number of Elements: {num_elements}")
    print(f"Memory Size: {memory_size_bytes} bytes ({memory_size_kb:.2f} KB, {memory_size_mb:.2f} MB)")

    return {
        "shape": shape,
        "num_elements": num_elements,
        "memory_size_bytes": memory_size_bytes,
        "memory_size_kb": memory_size_kb,
        "memory_size_mb": memory_size_mb,
    }

def compute_matrix_size_torch(matrix):
    """
    Compute and display the size of a torch tensor.

    Args:
        matrix (torch.Tensor): The tensor for which to compute the size.

    Returns:
        dict: A dictionary containing the number of elements, shape, and memory size.
    """
    # Get the number of elements
    num_elements = matrix.numel()

    # Get the shape of the matrix
    shape = matrix.shape

    # Compute memory size in bytes
    memory_size_bytes = matrix.element_size() * num_elements

    # Convert to kilobytes and megabytes for readability
    memory_size_kb = memory_size_bytes / 1024
    memory_size_mb = memory_size_kb / 1024

    # Print results
    print(f"Matrix Shape: {shape}")
    print(f"Number of Elements: {num_elements}")
    print(f"Memory Size: {memory_size_bytes} bytes ({memory_size_kb:.2f} KB, {memory_size_mb:.2f} MB)")

    return {
        "shape": shape,
        "num_elements": num_elements,
        "memory_size_bytes": memory_size_bytes,
        "memory_size_kb": memory_size_kb,
        "memory_size_mb": memory_size_mb,
    }


# Example usage
if __name__ == "__main__":
    # Create an example NumPy matrix
    matrix = np.random.random((8760, 90, 180))  # A 1000x1000 matrix

    x = np.array([8760, 90, 180, 4], dtype=np.float32)
    # Create a random 3D torch tensor
    t = torch.rand((8760, 90, 180, 4), dtype=torch.float32).cuda()
    input("Tensor loaded on GPU. Press Enter to continue...")

    # Compute the matrix size
    matrix_info = compute_matrix_size_numpy(matrix)

    # Compute the matrix size for torch tensor
    matrix_info_x = compute_matrix_size_torch(t)
