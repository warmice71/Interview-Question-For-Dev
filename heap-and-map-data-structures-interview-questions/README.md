# 44 Essential Heap and Map Data Structures Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

#### You can also find all 44 answers here 👉 [Devinterview.io - Heap and Map Data Structures](https://devinterview.io/questions/data-structures-and-algorithms/heap-and-map-data-structures-interview-questions)

<br>

## 1. What is a _Heap_?

A **Heap** is a **tree-based** data structure that is commonly used to implement **priority queues**. There are two primary types of heaps: Min Heap and Max Heap.

In a **Min Heap**, the root node is the smallest element, and each parent node is smaller than or equal to its children. Conversely, in a **Max Heap**, the root node is the largest element, and each parent node is greater than or equal to its children.

### Key Characteristics

- **Completeness**: All levels of the tree are fully populated except for possibly the last level, which is filled from left to right.
- **Heap Order**: Each parent node adheres to the heap property, meaning it is either smaller (Min Heap) or larger (Max Heap) than or equal to its children.

### Visual Representation

![Min and Max Heap](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/heaps%20and%20maps%2FMin-Max-Heap.png?alt=media&token=cb792085-64d2-4093-bc6f-c202463178cb)

### Common Implementation: Binary Heap

The **Binary Heap** is a popular heap implementation that is essentially a complete binary tree. The tree can represent either a Min Heap or Max Heap, with sibling nodes not being ordered relative to each other.

#### Array Representation and Index Relationships

Binary heaps are usually implemented using an array, where:

- Root: `heap[0]`
- Left Child: `heap[2 * i + 1]`
- Right Child: `heap[2 * i + 2]`
- Parent: `heap[(i - 1) // 2]`

### Core Operations

1. **Insert**: Adds an element while maintaining the heap order, generally using a "heapify-up" algorithm.
2. **Delete-Min/Delete-Max**: Removes the root and restructures the heap, typically using a "heapify-down" or "percolate-down" algorithm.
3. **Peek**: Fetches the root element without removing it.
4. **Heapify**: Builds a heap from an unordered collection.
5. **Size**: Returns the number of elements in the heap.

### Performance Metrics

- **Insert**: $O(\log n)$
- **Delete-Min/Delete-Max**: $O(\log n)$
- **Peek**: $O(1)$
- **Heapify**: $O(n)$
- **Size**: $O(1)$

### Code Example: Min Heap

Here is the Python code:

```python
# Utility functions for heapify-up and heapify-down
def heapify_up(heap, idx):
    parent = (idx - 1) // 2
    if parent >= 0 and heap[parent] > heap[idx]:
        heap[parent], heap[idx] = heap[idx], heap[parent]
        heapify_up(heap, parent)

def heapify_down(heap, idx, heap_size):
    left = 2 * idx + 1
    right = 2 * idx + 2
    smallest = idx
    if left < heap_size and heap[left] < heap[smallest]:
        smallest = left
    if right < heap_size and heap[right] < heap[smallest]:
        smallest = right
    if smallest != idx:
        heap[idx], heap[smallest] = heap[smallest], heap[idx]
        heapify_down(heap, smallest, heap_size)

# Complete MinHeap class
class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, value):
        self.heap.append(value)
        heapify_up(self.heap, len(self.heap) - 1)

    def delete_min(self):
        if not self.heap:
            return None
        min_val = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        heapify_down(self.heap, 0, len(self.heap))
        return min_val

    def peek(self):
        return self.heap[0] if self.heap else None

    def size(self):
        return len(self.heap)
```
<br>

## 2. What is a _Priority Queue_?

A **Priority Queue** is a specialized data structure that manages elements based on their assigned priorities. In this queue, **higher-priority** elements are processed before lower-priority ones.

### Key Features

- **Dynamic Ordering**: The queue adjusts its order as elements are inserted or their priorities are updated.
- **Efficient Selection**: The queue is optimized for quick retrieval of the highest-priority element.

### Core Operations

- **Insert**: Adds an element and its associated priority.
- **Delete-Max (or Min)**: Removes the highest-priority element.
- **Peek**: Retrieves but does not remove the highest-priority element.

### Performance Metrics

- **Insert**: $O(\log n)$
- **Delete-Max (or Min)**: $O(\log n)$
- **Peek**: $O(1)$

### Common Implementations

- **Unsorted Array/List**: Quick inserts $O(1)$ but slower maximum-priority retrieval $O(n)$.
- **Sorted Array/List**: Slower inserts $O(n)$ but quick maximum-priority retrieval $O(1)$.
- **Binary Heap**: Balanced performance for both insertions and deletions.

### Code Example: Binary heap-based priority queue

Here is the Python code:

```python
import heapq

# Initialize an empty Priority Queue
priority_queue = []

# Insert elements
heapq.heappush(priority_queue, 3)
heapq.heappush(priority_queue, 1)
heapq.heappush(priority_queue, 2)

# Remove and display the highest-priority element
print(heapq.heappop(priority_queue))  # Output: 1
```
<br>

## 3. How does the _Binary Heap_ property differ between a max heap and a min heap?

**Binary heaps** can be either a **max heap** or a **min heap**. The main difference lies in the heapifying process, where parent nodes either dominate (in a max heap) or are smaller than (in a min heap) their child nodes.

### Binary Heap Properties for Max Heap

- **Ordering**: Each node is greater than or equal to its children.
- **Visual Representation**: The structure looks like an inverted pyramid or an "upside-down tree". The parent nodes are greater than or equal to the child nodes.

#### Max Heap Example

```
       9
     /   \
    5     7
   / \   / \
  4   1 6   3
```

### Binary Heap Properties for Min Heap

- **Ordering:** Each node is less than or equal to its children.
- **Visual Representation**: The structure looks like a regular "tree", where each parent node is lesser than or equal to its children.

#### Min Heap Example

```
      1
     / \
    2   3
   / \ / \
  9  6 7  8
```

### Commonality in Operations

Both heap types share the following key operations:

- **peek**: Accesses the root element without deleting it.
- **insert**: Adds a new element to the heap.
- **remove**: Removes the root element (top element, or the top priority element in the case of a priority queue).
<br>

## 4. Can you explain heap property maintenance after an _Insert_ operation and a _Delete_ operation?

**Heap Property**, which distinguishes the heap from an ordinary tree, can be maintained in both **Insert** and **Delete** operations.

### Insert Operation

When a new element is inserted, the **upward heapization** or **bubble-up** process is crucial for preserving the heap property.

#### Visual Representation: Upward Heapification (Insert)

![Upward Heapification](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Max-Heap.svg/440px-Max-Heap.svg.png)

#### Python Implementation

Here is the Python code:

```python
# Upward heapify on insertion
def heapify_upwards(heap, i):
    parent = (i - 1) // 2
    if parent >= 0 and heap[parent] < heap[i]:
        heap[parent], heap[i] = heap[i], heap[parent]
        heapify_upwards(heap, parent)

def insert_element(heap, element):
    heap.append(element)
    heapify_upwards(heap, len(heap) - 1)
```

### Delete Operation

Whether it's a **Max Heap** or **Min Heap**, the **downward heapification** or **trickle-down** process is employed for maintaining the heap property after a deletion.

#### Visual Representation: Downward Heapification (Delete)

![Downward Heapification](https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Min-heap.png/440px-Min-heap.png)

#### Python Implementation

Here is the Python code for a **Max Heap**:

```python
# Downward heapify on deletion
def heapify_downwards(heap, i):
    left_child = 2 * i + 1
    right_child = 2 * i + 2
    largest = i

    if left_child < len(heap) and heap[left_child] > heap[largest]:
        largest = left_child
    if right_child < len(heap) and heap[right_child] > heap[largest]:
        largest = right_child

    if largest != i:
        heap[i], heap[largest] = heap[largest], heap[i]
        heapify_downwards(heap, largest)

def delete_max(heap):
    heap[0], heap[-1] = heap[-1], heap[0]
    deleted_element = heap.pop()
    heapify_downwards(heap, 0)
    return deleted_element
```

### Complexity Analysis

- **Time Complexity**: $O(\log n)$ for both insert and delete as these operations are backed by **heapify_upwards** and **heapify_downwards**, both of which have logarithmic time complexities in a balanced binary tree.
- **Space Complexity**: $O(1)$ for both insert and delete as they only require a constant amount of extra space.
<br>

## 5. _Insert_ an item into the _Heap_. Explain your actions.

### Problem Statement

Suppose we have a **binary heap** represented as follows:

```
        77
       /  \
      50   60
     / \   / \
    22  30 44  55
```
The task is to **insert** the value `55` into this binary heap while maintaining its properties.

### Solution

#### Algorithm Steps

1. **Adding the Element to the Bottom Level**: Initially, the element is placed at the next available position in the heap to maintain the heap's **shape property**.
2. **Heapify-Up**: The element is then moved up the tree until it is in the correct position, satisfying both the **shape property** and the **heap property**. This process is often referred to as "sift-up" or "bubble-up".

Let's walk through the steps for inserting `55` into the above heap.

**Step 1**: Add the Element to the Bottom Level

The next available position, following the **shape property**, is the first empty spot in the heap, starting from the left. The element `55` will go to the left of `22` in this case.

```
        77
       /  \
      50   60
     / \   / \
    22  30 44  55
    /
   55
```

**Step 2**: Heapify-Up

We start the heapify-up process from the newly added element and move it up through the heap as needed.

1. Compare `55` with its parent `22`. Since `55 > 22`, they are swapped.

```
        77
       /  \
      50   60
     / \   / \
    55  30 44  22
    /
   55
```

2. Now, compare the updated element `55` with its new parent `50`. As `55 > 50`, they are swapped.

```
        77
       /  \
      55   60
     / \   / \
    50  30 44  22
    /
   55
```

3. Finally, compare `55` with its parent `77`. As `55 < 77`, no more swapping is needed, and we can stop.

**Final Heap**

The final heap looks like this:

```
        77
       /  \
      55   60
     / \   / \
    50  30 44  22
    /
   55
```

#### Implementation

Here is the Python code:

```python
import heapq

# Initial heap
heap = [77, 50, 60, 22, 30, 44, 55]

# Add the new element
heap.append(55)

# Perform heapify-up
heapq._siftdown(heap, 0, len(heap)-1)

print("Final heap:", heap)
```

The output will be:

```
Final heap: [77, 55, 60, 22, 30, 44, 50, 55]
```

This final heap is a Min-Heap, as we used the `_siftdown` function from the `heapq` module, which is designed for Min-Heaps.
<br>

## 6. Compare _Heaps-Based_ vs. _Arrays-Based_ priority queue implementations.

Let's compare **arrays** and **binary heap** data structures used for **implementing a priority queue** in terms of their algorithms and performance characteristics.

### Key Distinctions

#### Definition

- **Array-Based**: An unsorted or sorted list where the highest-priority element is located via a linear scan.
- **Binary Heap-Based**: A binary tree with special properties, ensuring efficient access to the highest-priority element.

#### Core Operations Complexity

- **Array-Based**:
  - getMax: $O(n)$
  - insert: $O(1)$
  - deleteMax: $O(n)$

- **Binary Heap-Based**:
  - getMax: $O(1)$
  - insert: $O(\log n)$
  - deleteMax: $O(\log n)$

#### Practical Use Cases

- **Array-Based**:
  - Small datasets: If the dataset size is small or the frequency of `getMax` and `deleteMax` operations is low, an array can be a simpler and more direct choice.
  - Memory considerations: When working in environments with very constrained memory, the lower overhead of a static array might be beneficial.
  - Predictable load: If the primary operations are insertions and the number of `getMax` or `deleteMax` operations is minimal and predictable, an array can suffice.

- **Binary Heap-Based**:
  - Dynamic datasets: If elements are continuously being added and removed, binary heaps provide more efficient operations for retrieving and deleting the max element.
  - Larger datasets: Binary heaps are more scalable and better suited for larger datasets due to logarithmic complexities for insertions and deletions.
  - Applications demanding efficiency: For applications like task scheduling systems, network packet scheduling, or algorithms like Dijkstra's and Prim's, where efficient priority operations are crucial, binary heaps are the preferred choice.

### Code Example: Array-based priority queue

Here is the Python code:

```python 
class ArrayPriorityQueue:
    def __init__(self):
        self.array = []
    
    def getMax(self):
        return max(self.array)

    def insert(self, item):
        self.array.append(item)

    def deleteMax(self):
        max_val = self.getMax()
        self.array.remove(max_val)
        return max_val
```

### Code Example: Binary heap-based priority queue

Here is the Python code:

```python
import heapq

class BinaryHeapPriorityQueue:
    def __init__(self):
        self.heap = []
    
    def getMax(self):
        return -self.heap[0]  # Max element will be at the root in a max heap

    def insert(self, item):
        heapq.heappush(self.heap, -item)

    def deleteMax(self):
        return -heapq.heappop(self.heap)
```

### Recommendations

For modern applications, using **library-based priority queue implementations** is often the best choice. They are typically optimized and built upon efficient data structures like binary heaps.
<br>

## 7. How can you implement a heap efficiently using _Dynamic Arrays_?

**Dynamic arrays**, or vectors, provide a more memory-efficient representation of **heaps** compared to static arrays.

### Key Benefits

- **Amortized $O(1)$ Insertions and Deletions.** This holds until resizing is necessary.
- **Memory Flexibility.** They can shrink, unlike static arrays. This means you won't waste memory with an over-allocated heap.

### Core Operations

- **Insertion.** Place the new element at the end and then **"sift up"**, swapping with its parent until the heap property is restored.

- **Deletion**. Remove the root (which is the minimum in a min-heap), replace it with the last element, and then **"sift down"**, comparing with children and swapping, ensuring the heap is valid.

### Code Example: Dynamic Array-Based Min-Heap

Here is the Python code:

```python
class DynamicArrayMinHeap:
    def __init__(self):
        self.heap = []
        
    def parent(self, i):
        return (i - 1) // 2
    
    def insert(self, val):
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)
    
    def extract_min(self):
        if len(self.heap) == 0:
            return None

        if len(self.heap) == 1:
            return self.heap.pop()

        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return min_val

    def _sift_up(self, i):
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
    
    def _sift_down(self, i):
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self._sift_down(smallest)
```

### Practical Use-Cases

- **Time-Sensitive Applications.** If time-efficiency is crucial and long waits for array expansions are not acceptable, dynamic arrays are a better choice.
- **Databases.** Dynammic arrays are often used for memory management by database systems to handle variable-length fields in records more efficiently.
<br>

## 8. Name some ways to implement _Priority Queue_.

Let's look at different ways **Priority Queues** can be implemented and the time complexities associated with each approach.

### Common Implementations

#### List-Based

- **Unordered List**:
    - Insertion: $O(1)$
    - Deletion/Find Min/Max: $O(n)$
  
- **Ordered List**:  
    - Insertion: $O(n)$
    - Deletion/Find Min/Max: $O(1)$

#### Array-Based

- **Unordered Array**:
    - Insertion: $O(1)$
    - Deletion/Find Min/Max: $O(n)$

- **Ordered Array**:  
    - Insertion: $O(n)$
    - Deletion/Find Min/Max: $O(1)$

#### Tree-Based

- **Binary Search Tree (BST)**:
    - All Operations: $O(\log n)$ (can degrade to $O(n)$ if unbalanced)

- **Balanced BST (e.g., AVL Tree)**:
    - All Operations: $O(\log n)$

- **Binary Heap**:
    - Insertion/Deletion: $O(\log n)$
    - Find Min/Max: $O(1)$
<br>

## 9. How does the _Lazy deletion_ technique work in a heap and what is its purpose?

**Lazy deletion** keeps the deletion process simple, optimizing for both time and space efficiency. The primary goal is to reduce the number of **"holes" in the heap** to minimize reordering operations.

### Mechanism

When you remove an item:

1. The item to be removed is tagged as deleted, marking it as a "hole" in the heap.
2. The heap structure is restored without actively swapping the item with the last element.

### Advantages

- **Efficiency**: Reduces the noticeable overhead that can come with extensive restructuring.
- **Simplicity**: Provides a straightforward algorithm for the removal process, which aligns well with the rigidity and set rules of a heap.

### Considerations

1. **Time Complexity**: While operations like `removeMin` $or `removeMax`$ remain O$log n$, the actual time complexity can be slightly higher than that, as the algorithm potentially takes more time when it encounters a hole.
2. **"Hole" Management**: Too many "holes" can degrade performance, necessitating eventual cleanup.

### Code Example: Lazy Deletion

Here is the Python code:

```python
class LazyDeletionHeap(MinHeap):
    def __init__(self):
        super().__init__()
        self.deleted = set()

    def delete(self, item):
        if item in self.array:
            self.deleted.add(item)

    def remove_min(self):
        while self.array and self.array[0] in self.deleted:
            self.deleted.remove(self.array[0])
            self.array.pop(0)
            self.heapify()
        if self.array:
            return self.array.pop(0)
```

In this code, `minHeap` is a normal heap structure, and `array` is the list used to represent the heap **indexing starts from 0**.
<br>

## 10. Explain _Heapify_ and where it is used in heap operations.

**Heapify** is a method that ensures the **heap property** is maintained for a given array, typically performed in the background during **heap operations**.

### Why Use Heapify?

Without **Heapify**, operations like insert or delete on heaps can take up to $O(n \log n)$ time, as it could trigger a complete heap sort repeatedly. In contrast, with Heapify, such actions are consistently within $O(\log n)$ time complexity, enhancing the heap's overall efficiency.

### Heapify Process

The starting point is usually the bottom-most, rightmost "subtree" of the heap.

1. **Locate:** 
   - Identify the  multi-level rightmost leaf of the subtree.

2. **Sift Upwards:** Peform a parent-child comparison in binary heaps to correct the heap order.
   - If the child node is greater (for a max heap), swap it with the parent.
   - Repeat this process, moving upwards, until the parent is greater than both its children or until the root is reached.

3. **Repeat:** 
   - Continue the process sequentially for each level, from right to left.

Focused on the local structure, this process simplifies the task to $O(\log n)$ complexity.

### Complexity Analysis of Heapify

- **Time Complexity**: $O(\log n)$
- **Space Complexity**: $O(1)$

### Visual Representation

![Heapify Example](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/heaps%20and%20maps%2Fmax-heapify.png?alt=media&token=b8595b27-0b3d-412f-83dd-5caf36f6974e)

In the example above, we begin heapifying from the right-bottom node (index 6), compare it with its parent (index 2) and move upwards, ensuring the max-heap property is satisfied.

### Code Example: Heapify

Here is the Python code:

```python
def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
  
    if l < n and arr[i] < arr[l]:
        largest = l
  
    if r < n and arr[largest] < arr[r]:
        largest = r
  
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# Example usage to build a max heap
arr = [12, 11, 10, 5, 6, 2, 1]
n = len(arr)
for i in range(n//2 - 1, -1, -1):
    heapify(arr, n, i)
```
<br>

## 11. What is _Heap Sort_?

**Heap Sort** is a robust comparison-based sorting algorithm that uses a **binary heap** data structure to build a "heap tree" and then sorts the elements.

### Key Characteristics

- **Selection-Based**: Iteratively selects the largest (in a max heap) or the smallest (in a min heap) element.
- **In-Place**: Sorts the array within its original storage without the need for additional memory, yielding a space complexity of $O(1)$.
- **Unstable**: Does not guarantee the preservation of the relative order of equal elements.
- **Less Adaptive**: Doesn't take advantage of existing or partial order in a dataset.

### Algorithm Steps

1. **Heap Construction**: Transform the input array into a max heap.
2. **Element Removal**: Repeatedly remove the largest element from the heap and reconstruct the heap.
3. **Array Formation**: Place the removed elements back into the array in sorted order.

### Visual Representation

![Heap Sort](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/sorting%2Fheap-sort.gif?alt=media&token=fea201db-390d-4071-9deb-0343cae6772c&_gl=1*z349t9*_ga*OTYzMjY5NTkwLjE2ODg4NDM4Njg.*_ga_CW55HF8NVT*MTY5NjUyOTYzNy4xNDUuMS4xNjk2NTMxNDIyLjU0LjAuMA..)

### Complexity Analysis

- **Time Complexity**: Best, Average, and Worst Case: $O(n \log n)$ - Building the heap is $O(n)$ and each of the $n$ removals requires $\log n$ time.
- **Space Complexity**: $O(1)$

### Code Example: Heap Sort

Here is the Python code:

```python
def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr
```
<br>

## 12. How does _Heap Sort_ compare to _Quick Sort_ in terms of speed and memory usage?

Let's discuss a novel and a traditional sorting algorithm and their respective performance characteristics.

### Quick Sort

- **Time Complexity**: 
  - Best & Average: $O(n \log n)$
  - Worst: $O(n^2)$ - when the pivot choice is consistently bad, e.g., always selecting the smallest or largest element.
- **Expected Space Complexity**: O($\log n$) - **In-place** algorithm.
- **Partitioning**: Efficient for large datasets with pivot selection and partitioning.

### Heap Sort

- **Time Complexity**: 
  - Best, Average, Worst: $O(n \log n)$ - consistent behavior regardless of data.
- **Space Complexity**: $O(1)$ - **In-place** algorithm.
- **Heap Building**: $O(n)$ - extra initial step to build the heap.
- **Stability**: May become unstable after initial build, unless modifications are made.
- **External Data Buffer**: Maintains data integrity during the sorting process, leading to its widespread use in numerous domains.
<br>

## 13. Describe the _algorithm_ for _Merging k Sorted Arrays_ using a heap.

To **merge** $k$ sorted arrays efficiently, you can **utilize** a **min-heap** to keep track of the **smallest** element remaining in each array. Here, $n$ represents the average length of the arrays.

### Algorithm Steps

1. **Initialize**: Build a min-heap of size $k$ with the first element of each array.
3. **Iterate**: Remove the minimum from the heap, add the next element of its array to the heap if available, and repeat until the heap is empty.
4. **Complete**: The heap holds the smallest remaining elements, output these in order.

### Complexity Analysis

- **Time Complexity**: $O(n \cdot \log k)$ - Both building the initial heap and each element removal and insertion takes $O(\log k)$ time.
- **Space Complexity**: $O(k)$ - The heap can contain at most $k$ elements.

### Code Example: Merge k Sorted Arrays using Min Heap

Here is the Python code:

```python
import heapq

def merge_k_sorted_arrays(arrays):
    result = []
    
    # Initialize min-heap with first element from each array
    heap = [(arr[0], i, 0) for i, arr in enumerate(arrays) if arr]
    heapq.heapify(heap)
    
    # While heap is not empty, keep track of minimum element
    while heap:
        val, array_index, next_index = heapq.heappop(heap)
        result.append(val)
        next_index += 1
        if next_index < len(arrays[array_index]):
            heapq.heappush(heap, (arrays[array_index][next_index], array_index, next_index))
    
    return result

# Example usage
arrays = [[1, 3, 5], [2, 4, 6], [0, 7, 8, 9]]
merged = merge_k_sorted_arrays(arrays)
print(merged)  # Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
<br>

## 14. Implement a heap that efficiently supports _Find-Max and Find-Min_ operations.

### Problem Statement

The goal is to build a **data structure**, representing either a **min or max heap**, that supports efficient `find` operations for both the minimum and maximum elements within the heap.

### Solution

To fulfill the `Find-Min` and `Find-Max` requirements, two heaps are used in tandem:

1. A `Min-Heap` ensures the minimum element can be quickly retrieved, with the root holding the smallest value.
2. A `Max-Heap` facilitates efficient access to the maximum element, positioning the largest value at the root.

The overall time complexity for both `find` operations is $O(1)$.

#### Implementation Steps

1. **Initialize Both Heaps**: Choose the appropriate built-in library or implement the heap data structure. Here, we'll use the `heapq` library in Python.
2. **Insert Element**: With a single heap, inserting an element only involves pushing it onto the heap. However, with two heaps, each new element needs to be placed on the heap that will balance the size. This ensures that the root of one heap will correspond to the minimum or maximum value across all elements.
3. **Retrieve Min & Max**: For both operations, simply access the root of the corresponding heap.

#### Complexity Analysis

- **Time Complexity**:
  - `findMin` and `findMax`: $O(1)$
  - `insert`: $O(\log n)$
- **Space Complexity**: $O(n)$ for both heaps.

#### Python Implementation

Here is the code:

```python
import heapq

class FindMinMaxHeap:
    def __init__(self):
        self.min_heap, self.max_heap = [], []
        self.size = 0

    def insert(self, num):
        if self.size % 2 == 0:
            heapq.heappush(self.max_heap, -1 * num)
            self.size += 1
            if len(self.min_heap) == 0:
                return
            if -1 * self.max_heap[0] > self.min_heap[0]:
                max_root = -1 * heapq.heappop(self.max_heap)
                min_root = heapq.heappop(self.min_heap)
                heapq.heappush(self.max_heap, -1 * min_root)
                heapq.heappush(self.min_heap, max_root)
        else:
            if num < -1 * self.max_heap[0]:
                heapq.heappush(self.min_heap, -1 * heapq.heappop(self.max_heap))
                heapq.heappush(self.max_heap, -1 * num)
            else:
                heapq.heappush(self.min_heap, num)
            self.size += 1

    def findMin(self):
        if self.size % 2 == 0:
            return (-1 * self.max_heap[0] + self.min_heap[0]) / 2.0
        else:
            return -1 * self.max_heap[0]

    def findMax(self):
        if self.size % 2 == 0:
            return (-1 * self.max_heap[0] + self.min_heap[0]) / 2.0
        else:
            return -1 * self.max_heap[0]

# Example
mmh = FindMinMaxHeap()
mmh.insert(3)
mmh.insert(5)
print("Min:", mmh.findMin())  # Output: 3
print("Max:", mmh.findMax())  # Output: 5
mmh.insert(1)
print("Min:", mmh.findMin())  # Output: 2
print("Max:", mmh.findMax())  # Output: 3
```
<br>

## 15. What are some _Practical Applications_ of _Heaps_?

**Heaps** find application in a variety of algorithms and data handling tasks.

### Practical Applications

- **Extreme Value Access**: The root of a min-heap provides the smallest element, while the root of a max-heap provides the largest, allowing constant-time access.
  
- **Selection Algorithms**: Heaps can efficiently find the $k^{th}$ smallest or largest element in $O(k \log k + n)$ time.

- **Priority Queue**: Using heaps, we can effectively manage a set of data wherein each element possesses a distinct priority, ensuring operations like insertion, maximum extraction, and sorting are performed efficiently.

- **Sorting**: Heaps play a pivotal role in the Heap Sort algorithm, offering $O(n \log n)$ time complexity. They're also behind utility functions in some languages like `nlargest()` and `nsmallest()`.

- **Merge Sorted Streams**: Heaps facilitate the merging of multiple pre-sorted datasets or streams into one cohesive sorted output.

- **Graph Algorithms**: Heaps, especially when implemented as priority queues, are instrumental in graph algorithms such as Dijkstra's Shortest Path and Prim's Minimum Spanning Tree, where they assist in selecting the next node to process efficiently.
<br>



#### Explore all 44 answers here 👉 [Devinterview.io - Heap and Map Data Structures](https://devinterview.io/questions/data-structures-and-algorithms/heap-and-map-data-structures-interview-questions)

<br>

<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

