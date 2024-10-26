def quick_sort(arr):
    if len(arr) <= 1:  # if the array has only one element or no element
        return arr  # simply return the array
    else:
        pivot = arr[0]  # select the first element as pivot
        less = [x for x in arr[1:] if x <= pivot]  # create a list of all the elements less than the pivot
        greater = [x for x in arr[1:] if x > pivot]  # create a 2nd list of all the elements greater than the pivot
        return quick_sort(less) + [pivot] + quick_sort(greater) # recursively call the quick_sort function on the less and greater list and concatenate the sorted lists
        # it returns the sorted list on both left and right side of the pivot
    
def merge_sort(arr):
    if len(arr) > 1:     # if the array has more than one element, simèly return the array
        mid = len(arr) // 2   # find the middle of the array
        left_half = arr[:mid]
        right_half = arr[mid:]

        # Recursive call on each half
        # UNtil both halves are only length 1 on 1st passthorugh
        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        # Copy data to temp arrays left_half[] and right_half[]
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]  # if the element in the left half is less than the element in the right half, copy the element to the original array
                i += 1
            else:
                arr[k] = right_half[j]  # else the opposite
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
    return arr

# TODO: Implement Tim Sort 
def tim_sort(arr):
    arr.sort()
    return arr

def heap_sort(arr):
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        # Check if left child exists and is greater than the root
        if left < n and arr[i] < arr[left]:
            largest = left

        # Check if right child exists and is greater than the root
        if right < n and arr[largest] < arr[right]:
            largest = right

        # Change root if needed
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)

    # Build a maxheap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr
    
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        # Track if any swap happens
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                # Swap the elements
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        # If no swap happened, the array is already sorted
        if not swapped:
            break
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def selection_sort(arr):
    for i in range(len(arr)):
        min_index = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

def tree_sort(arr):
    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    def insert(node, key):
        if node is None:
            return Node(key)
        if key < node.key:
            node.left = insert(node.left, key)
        else:
            node.right = insert(node.right, key)
        return node

    def in_order_traversal(node):
        if node is not None:
            in_order_traversal(node.left)
            sorted_arr.append(node.key)
            in_order_traversal(node.right)

    root = None
    for key in arr:
        root = insert(root, key)

    sorted_arr = []
    in_order_traversal(root)
    return sorted_arr

def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

def bucket_sort(arr):
    # Create a list of empty buckets
    n = len(arr)
    buckets = [[] for _ in range(n)]

    # Add elements to the buckets
    for i in range(n):
        bucket_index = n * arr[i] // (max(arr) + 1)
        buckets[bucket_index].append(arr[i])

    # Sort the elements in each bucket
    for i in range(n):
        buckets[i] = insertion_sort(buckets[i])

    # Concatenate the sorted elements
    k = 0
    for i in range(n):
        for j in range(len(buckets[i])):
            arr[k] = buckets[i][j]
            k += 1
    return arr

def radix_sort(arr):
    def counting_sort(arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * 10

        # Store the count of each element
        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1

        # Store the cumulative count of each element
        for i in range(1, 10):
            count[i] += count[i - 1]

        # Build the output array
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1

        # Copy the output array to the original array
        for i in range(n):
            arr[i] = output[i]

    max_element = max(arr)
    exp = 1
    while max_element // exp > 0:
        counting_sort(arr, exp)
        exp *= 10
    return arr

def counting_sort(arr):
    n = len(arr)
    output = [0] * n

    # Find the maximum element in the array
    max_element = max(arr)

    # Initialize count array with zeros
    count = [0] * (max_element + 1)

    # Store the count of each element
    for i in range(n):
        count[arr[i]] += 1

    # Store the cumulative count of each element
    for i in range(1, max_element + 1):
        count[i] += count[i - 1]

    # Find the index of each element of the original array in count array
    # Place the elements in output array
    i = n - 1
    while i >= 0:
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1
        i -= 1

    # Copy the sorted elements into original array
    for i in range(n):
        arr[i] = output[i]
    return arr

# TODO: Implement Cube Sort
def cube_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr