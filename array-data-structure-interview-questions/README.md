# 60 Important Array Data Structure Interview Questions
<div>
<p align="center">
<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

#### You can also find all 60 answers here 👉 [Devinterview.io - Array Data Structure](https://devinterview.io/questions/data-structures-and-algorithms/array-data-structure-interview-questions)

<br>

## 1. What is an _Array_?

An **array** is a fundamental data structure used for storing a **sequence** of elements that can be accessed via an **index**. 

### Key Characteristics

- **Homogeneity**: All elements are of the same data type.
- **Contiguous Memory**: Elements are stored in adjacent memory locations for quick access.
- **Fixed Size**: Arrays are generally static in size, although dynamic arrays exist in modern languages.
- **Indexing**: Usually zero-based, though some languages use one-based indexing.

### Time Complexity of Basic Operations

- **Access**: $O(1)$
- **Search**: $O(1)$, $O(n)$ assuming unsorted array
- **Insertion**: $O(1)$ for the end, $O(n)$ for beginning/middle
- **Deletion**: $O(1)$ for the end, $O(n)$ for beginning/middle
- **Append**: $O(1)$ amortized, $O(n)$ during resizing

### Code Example: Basic Array Operations

Here is the Java code:

```java
public class ArrayExample {
    public static void main(String[] args) {
        // Declare and Initialize Arrays
        int[] myArray = new int[5];  // Declare an array of size 5
        int[] initializedArray = {1, 2, 3, 4, 5};  // Direct initialization
        
        // Access Elements
        System.out.println(initializedArray[0]);  // Output: 1
        
        // Update Elements
        initializedArray[2] = 10;  // Modify the third element
        
        // Check Array Length
        int length = initializedArray.length;  // Retrieve array length
        System.out.println(length);  // Output: 5
    }
}
```
<br>

## 2. What are _Dynamic Arrays_?

**Dynamic arrays** start with a preset capacity and **automatically resize** as needed. When full, they allocate a larger memory block—often doubling in size—and copy existing elements.

### Key Features

- **Adaptive Sizing**: Dynamic arrays adjust their size based on the number of elements, unlike fixed-size arrays.
- **Contiguous Memory**: Dynamic arrays, like basic arrays, keep elements in adjacent memory locations for efficient indexed access.
- **Amortized Appending**: Append operations are typically constant time. However, occasional resizing might take longer, but averaged over multiple operations, it's still $O(1)$ amortized.

### Time Complexity of Basic Operations

- **Access**: $O(1)$
- **Search**: $O(1)$ for exact matches, $O(n)$ linearly for others
- **Insertion**: $O(1)$ amortized, $O(n)$ during resizing
- **Deletion**: $O(1)$ amortized, $O(n)$ during shifting or resizing
- **Append**: $O(1)$ amortized, $O(n)$ during resizing

### Code Example: Java's 'ArrayList': Simplified  Implementation

Here is the Java code:

```java
import java.util.Arrays;

public class DynamicArray<T> {
    private Object[] data;
    private int size = 0;
    private int capacity;

    public DynamicArray(int initialCapacity) {
        this.capacity = initialCapacity;
        data = new Object[initialCapacity];
    }

    public T get(int index) {
        return (T) data[index];
    }

    public void add(T value) {
        if (size == capacity) {
            resize(2 * capacity);
        }
        data[size++] = value;
    }

    private void resize(int newCapacity) {
        Object[] newData = new Object[newCapacity];
        for (int i = 0; i < size; i++) {
            newData[i] = data[i];
        }
        data = newData;
        capacity = newCapacity;
    }

    public int size() {
        return size;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public static void main(String[] args) {
        DynamicArray<Integer> dynArray = new DynamicArray<>(2);
        dynArray.add(1);
        dynArray.add(2);
        dynArray.add(3);  // This will trigger a resize
        System.out.println("Size: " + dynArray.size());  // Output: 3
        System.out.println("Element at index 2: " + dynArray.get(2));  // Output: 3
    }
}
```
<br>

## 3. What is an _Associative Array_ (Dictionary)?

An **Associative Array**, often referred to as **Map**, **Hash**, or **Dictionary** is an abstract data type that enables **key-based access** to its elements and offers **dynamic resizing** and fast retrieval.

### Key Features

- **Unique Keys**: Each key is unique, and adding an existing key updates its value.

- **Variable Key Types**: Keys can be diverse types, including strings, numbers, or objects.

### Common Implementations

- **Hash Table**: Efficiency can degrade due to hash collisions.
  - Average Case $O(1)$
  - Worst Case $O(n)$

- **Self-Balancing Trees**: Consistent efficiency due to balanced structure.
  - Average Case $O(\log n)$
  - Worst Case $O(\log n)$

- **Unbalanced Trees**: Efficiency can vary, making them less reliable.
  - Average Case Variable
  - Worst Case between $O(\log n)$ and $O(n)$

- **Association Lists**: Simple structure, not ideal for large datasets.
  - Average and Worst Case $O(n)$

### Code Example: Associative Arrays vs. Regular Arrays

Here is the Python code:

```python
# Regular Array Example
my_list = ["apple", "banana", "cherry"]
print(my_list[1])  # Outputs: banana

# Trying to access using non-integer index would cause an error:
# print(my_list["fruit_name"])  # This would raise an error.

# Associative Array (Dictionary) Example
my_dict = {
    "fruit_name": "apple",
    42: "banana",
    (1, 2): "cherry"
}

print(my_dict["fruit_name"])  # Outputs: apple
print(my_dict[42])           # Outputs: banana
print(my_dict[(1, 2)])      # Outputs: cherry

# Demonstrating key update
my_dict["fruit_name"] = "orange"
print(my_dict["fruit_name"])  # Outputs: orange
```
<br>

## 4. What defines the _Dimensionality_ of an array?

**Array dimensionality** indicates the number of indices required to select an element within the array. A classic example is the Tic-Tac-Toe board, which is a two-dimensional array, and elements are referenced by their row and column positions.

### Code Example: Tic-Tac-Toe Board (2D Array)

Here is the Python code:

```python
# Setting up the Tic-Tac-Toe board
tic_tac_toe_board = [
    ['X', 'O', 'X'],
    ['O', 'X', 'O'],
    ['X', 'O', 'X']
]

# Accessing the top-left corner (which contains 'X'):
element = tic_tac_toe_board[0][0]
```

### Code Example: 3D Array

Here is the Python code:

```python
arr_3d = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
]
```

A three-dimensional array can be imagined as a **cube** or a **stack** of matrices.

### Mathematical Perspective

Mathematically, an array's dimensionality aligns with the Cartesian product of sets, each set corresponding to an axis. A 3D array, for instance, is formed from the Cartesian product of three distinct sets.

#### Beyond 3D: N-Dimensional Arrays

Arrays can extend into N dimensions, where $N$ can be any positive integer. The total count of elements in an N-dimensional array is:

$$
\text{Number of Elements} = S_1 \times S_2 \times \ldots \times S_N
$$

Where $S_k$ signifies the size of the $k$-th dimension.
<br>

## 5. Name some _Advantages_ and _Disadvantages_ of arrays.

**Arrays** have very specific **strengths** and **weaknesses**, making them better suited for some applications over others.

### Advantages

- **Speed**: Arrays provide $O(1)$ access and append operations when appending at a known index (like the end).

- **Cache Performance**: Arrays, with their contiguous memory layout, are efficient for tasks involving sequential data access.

### Disadvantages

- **Size Limitations**: Arrays have a fixed size after allocation. Resizing means creating a new array, leading to potential memory overhead or data transfer costs.

- **Mid-Array Changes**: Operations like insertions or deletions are $O(n)$ due to necessary element shifting.

### Considerations

- **When to Use**: Arrays are optimal for **known data sizes** and when rapid access or appends are critical. They're popular in numerical algorithms and cache-centric tasks.

- **When to Rethink**: Their static nature and inefficiency for **frequent mid-array changes** make alternatives like linked lists or hash tables sometimes more suitable.
<br>

## 6. Explain _Sparse_ and _Dense_ arrays.

**Sparse arrays** are data structures optimized for arrays where most values are default (e.g., zero or null). They save memory by storing only non-default values and their indices. In contrast, **dense arrays** allocate memory for every element, irrespective of it being a default value.

### Example

- **Sparse Array**: `[0, 0, 3, 0, 0, 0, 0, 9, 0, 0]`
- **Dense Array**: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`

### Advantages of Sparse Arrays

Sparse arrays offer **optimized memory usage**.

For example, in a million-element array where 90% are zeros:

- **Dense Array**: Allocates memory for every single element, even if the majority are zeros.
- **Sparse Array**: Drastically conserves memory by only allocating for non-zero elements.

### Practical Application

1. **Text Processing**: Efficiently represent term-document matrices in analytics where not all words appear in every document.

2. **Computer Graphics**: Represent 3D spaces in modeling where many cells may be empty.

3. **Scientific Computing**: Handle linear systems with sparse coefficient matrices, speeding up computations.

4. **Databases**: Store tables with numerous missing values efficiently.

5. **Networking**: Represent sparsely populated routing tables in networking equipment.

6. **Machine Learning**: Efficiently handle high-dimensional feature vectors with many zeros.

7. **Recommendation Systems**: Represent user-item interaction matrices where many users haven't interacted with most items. 

### Code Example: Sparse Array

Here is a Python code:

```python
class SparseArray:
    def __init__(self):
        self.data = {}

    def set(self, index, value):
        if value != 0:  # Only store non-zero values
            self.data[index] = value
        elif index in self.data:
            del self.data[index]

    def get(self, index):
        return self.data.get(index, 0)  # Return 0 if index is not in the data

# Usage
sparse_array = SparseArray()
sparse_array.set(2, 3)
sparse_array.set(7, 9)

print(sparse_array.get(2))  # Output: 3
print(sparse_array.get(7))  # Output: 9
print(sparse_array.get(3))  # Output: 0
```
<br>

## 7. What are advantages and disadvantages of _Sorted Arrays_?

A **sorted array** is a data structure where elements are stored in a specific, **predetermined sequence**, usually in ascending or descending order.

This ordering provides various benefits, such as **optimized search operations**, at the cost of more complex insertions and deletions.

### Advantages

- **Efficient Searches**: Sorted arrays are optimized for search operations, especially when using algorithms like Binary Search, which has a $O(\log n)$ time complexity.
  
- **Additional Query Types**: They support other specialized queries, like bisection to find the closest element and range queries to identify elements within a specified range.

- **Cache Efficiency**: The contiguous memory layout improves cache utilization, which can lead to faster performance.

### Disadvantages

- **Slow Updates**: Insertions and deletions generally require shifting elements, leading to $O(n)$ time complexity for these operations.
  
- **Memory Overhead**: The need to maintain the sorted structure can require extra memory, especially during updates.

- **Lack of Flexibility**: Sorted arrays are less flexible for dynamic resizing and can be problematic in parallel computing environments.

### Practical Applications

- **Search-Heavy Applications**: Suitable when rapid search operations are more common than updates, such as in financial analytics or in-memory databases.
- **Static or Semi-Static Data**: Ideal for datasets known in advance or that change infrequently.
- **Memory Constraints**: They are efficient for small, known datasets that require quick search capabilities.

### Time Complexity of Basic Operations

- **Access**: $O(1)$.
- **Search**: $O(1)$ for exact matches, $O(\log n)$ with binary search for others.
- **Insertion**: $O(1)$ for the end, but usually $O(n)$ to maintain order.
- **Deletion**: $O(1)$ for the end, but usually $O(n)$ to maintain order.
- **Append**: $O(1)$ if appending a larger value, but can spike to $O(n)$ if resizing or inserting in order.
<br>

## 8. What are the advantages of _Heaps_ over _Sorted Arrays_?

While both **heaps** and **sorted arrays** have their strengths, heaps are often preferred when dealing with dynamic data requiring frequent insertions and deletions.

### Advantages of Heaps Over Sorted Arrays

- **Dynamic Operations**: Heaps excel in scenarios with frequent insertions and deletions, maintaining their structure efficiently.
- **Memory Allocation**: Heaps, especially when implemented as binary heaps, can be efficiently managed in memory as they're typically backed by arrays. Sorted arrays, on the other hand, might require periodic resizing or might have wasted space if over-allocated.
- **Predictable Time Complexity**: Heap operations have consistent time complexities, while sorted arrays can vary based on specific data scenarios.
- **No Overhead for Sorting**: Heaps ensure parents are either always smaller or larger than children, which suffices for many tasks without the overhead of maintaining full order as in sorted arrays.

### Time Complexities of Key Operations

#### Heaps

- **find-min**: $O(1)$ – The root node always contains the minimum value.
- **delete-min**: $O(\log n)$ – Removal of the root is followed by the heapify process to restore order.
- **insert**: $O(\log n)$ – The newly inserted element might need to be bubbled up to its correct position.

#### Sorted Arrays

- **find-min**: $O(1)$ – The first element is the minimum if the array is sorted in ascending order.
- **delete-min**: $O(n)$ – Removing the first element requires shifting all other elements.
- **insert**: $O(n)$ – Even though we can find the insertion point in $O(\log n)$ with binary search, we may need to shift elements, making it $O(n)$ in the worst case.
<br>

## 9. How does _Indexing_ work in arrays?

**Indexing** refers to accessing specific elements in an array using unique indices, which range from 0 to $n-1$ for an array of $n$ elements.

### Key Concepts

#### Contiguous Memory and Fixed Element Size

Arrays occupy adjacent memory locations, facilitating fast random access. All elements are uniformly sized. For example, a 32-bit integer consumes 4 bytes of memory.

#### Memory Address Calculation

The memory address of the $i$-th element is computed as:

$$
\text{Memory Address}_{i} = P + (\text{Element Size}) \times i
$$

Here, $P$ represents the pointer to the array's first element.

### Code Example: Accessing Memory Address

Here is the Python code:

```python
# Define an array
arr = [10, 20, 30, 40, 50, 60]

# Calculate memory address of the third element
element_index = 2
element_address = arr.__array_interface__['data'][0] + element_index * arr.itemsize

# Retrieve the element value
import ctypes
element_value = ctypes.cast(element_address, ctypes.py_object).value

# Output
print(f"The memory address of the third element is: {element_address}")
print(f"The value at that memory address is: {element_value}")
```
<br>

## 10. _Merge_ two _Sorted Arrays_ into one _Sorted Array_.

### Problem Statement

The task is to **merge two sorted arrays** into one combined, sorted array.

### Solution

#### Algorithm Steps

1. Initialize the result array **C**, with counters `i=0` for array **A** and `j=0` for array **B**.
2. While `i` is within the bounds of array **A** and `j` is within the bounds of array **B**:
    a. If `A[i]` is less than `B[j]`, append `A[i]` to `C` and increment `i`.
    b. If `A[i]` is greater than `B[j]`, append `B[j]` to `C` and increment `j`.
    c. If `A[i]` is equal to `B[j]`, append both `A[i]` and `B[j]` to `C` and increment both `i` and `j`.
3. If any elements remain in array **A**, append them to `C`.
4. If any elements remain in array **B**, append them to `C`.
5. Return the merged array `C`.

### Visual Representation

![Merging Two Sorted Arrays into One](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/arrays%2Fmerge-two-sorted-array-algorithm%20(1).png?alt=media&token=580caabc-2bc4-4928-9780-ba7bb13d0cb1&_gl=1*14yao85*_ga*OTYzMjY5NTkwLjE2ODg4NDM4Njg.*_ga_CW55HF8NVT*MTY5NzM3MjYxNC4xNjAuMS4xNjk3MzcyNjQ2LjI8LjAuMA..)

#### Complexity Analysis

- **Time Complexity**: $O(n)$, where $n$ is the combined length of Arrays A and B.
- **Space Complexity**: $O(n)$, considering the space required for the output array.

#### Implementation

Here is the Python code:

```python
def merge_sorted_arrays(a, b):
    merged_array, i, j = [], 0, 0

    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            merged_array.append(a[i])
            i += 1
        elif a[i] > b[j]:
            merged_array.append(b[j])
            j += 1
        else:
            merged_array.extend([a[i], b[j]])
            i, j = i + 1, j + 1

    merged_array.extend(a[i:])
    merged_array.extend(b[j:])
        
    return merged_array

# Sample Test
array1 = [1, 3, 5, 7, 9]
array2 = [2, 4, 6, 8, 10]
print(merge_sorted_arrays(array1, array2))  # Expected Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```
<br>

## 11. Implement three _Stacks_ with one _Array_.

### Problem Statement

The task is to implement **three stacks** using a **single dynamic array**.

### Solution

To solve the task we can **divide** the array into twelve portions, with four sections for each stack, allowing each of them to **grow** and **shrink** without affecting the others.

#### Algorithm Steps

1. Initialize Stack States: 
    - Set `size` as the full array length divided by 3.
    - Set `stackPointers` as `[ start, start + size - 1, start + 2*size - 1 ]`, where `start` is the array's beginning index.

2. Implement `Push` Operation:  For stack 1, check if `stackPointers[0]` is less than `start + size - 1` before pushing.

#### Complexity Analysis

- **Time Complexity**: $O(1)$ for all stack operations.
- **Space Complexity**: $O(1)$

#### Implementation

Here is the Python code:

```python
class MultiStack:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.array = [None] * (3 * stack_size)
        self.stack_pointers = [-1, -1, -1]

    def push(self, stack_number, value):
        if self.stack_pointers[stack_number] >= self.stack_size - 1:
            print("Stack Overflow!")
            return

        self.stack_pointers[stack_number] += 1
        self.array[self.stack_pointers[stack_number]] = value

    def pop(self, stack_number):
        if self.stack_pointers[stack_number] < 0:
            print("Stack Underflow!")
            return None

        value = self.array[self.stack_pointers[stack_number]]
        self.stack_pointers[stack_number] -= 1
        return value

    def peek(self, stack_number):
        if self.stack_pointers[stack_number] < 0:
            print("Stack Underflow!")
            return None

        return self.array[self.stack_pointers[stack_number]]
```
<br>

## 12. How do you perform _Array Rotation_ and what are its applications?

**Array rotation** involves moving elements within an array to shift its position. This operation can be beneficial in various scenarios, from data obfuscation to algorithmic optimizations.

### Types of Array Rotation

1. **Left Rotation**: Shifts elements to the left.
2. **Right Rotation**: Shifts elements to the right.

### Algorithms for Array Rotation

1. **Naive Method**: Directly shifting each element one at a time, $d$ times, where $d$ is the rotation factor.
2. **Reversal Algorithm**: Involves performing specific **reversals** within the array to achieve rotation more efficiently.

### Code Example: Array Rotation using the Reversal Algorithm

Here is the Python code:

```python
def reverse(arr, start, end):
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1

def rotate_array(arr, d):
    n = len(arr)
    reverse(arr, 0, d-1)
    reverse(arr, d, n-1)
    reverse(arr, 0, n-1)

# Example
my_array = [1, 2, 3, 4, 5, 6, 7]
rotate_array(my_array, 3)
print(my_array)  # Output: [4, 5, 6, 7, 1, 2, 3]
```

### Applications of Array Rotation

1. **Obfuscation of Data**: By performing secure operations, such as circular permutations on sensitive arrays, it ensures data confidentiality.
  
2. **Cryptography**: Techniques like the Caesar cipher use array rotation to encrypt and decrypt messages. Modern ciphers similarly rely on advanced versions of this concept.

3. **Memory Optimization**: It ensures that data in the array is arranged for optimal memory access, which is crucial in large datasets or when working with limited memory resources.

4. **Algorithm Optimization**: Certain algorithms, such as search and sorting algorithms, might perform better on a particular range of elements within an array. Rotation allows for tailoring the array to these algorithms for enhanced performance.
<br>

## 13. _Reverse_ an _Array_ in place.

### Problem Statement

Given an array, the objective is to **reverse the sequence of its elements**.

### Solution

Two elements are selected from each end of the array and are swapped. This process continues, with the selected elements moving towards the center, until the entire array is reversed.

#### Algorithm Steps

1. Begin with two pointers: `start` at index 0 and `end` at the last index.
2. Swap the elements at `start` and `end` positions.
3. Increment `start` and decrement `end`.
4. Repeat Steps 2 and 3 until the pointers meet at the center of the array.

This algorithm **reverses the array in place, with a space complexity of $O(1)$**.

#### Complexity Analysis

- **Time Complexity**: $O(n/2)$ as the swapping loop only runs through half of the array.
- **Space Complexity**: Constant, $O(1)$, as no additional space is required.

#### Implementation

Here is the Python code:

```python
def reverse_array(arr):
  start = 0
  end = len(arr) - 1

  while start < end:
    arr[start], arr[end] = arr[end], arr[start]
    start += 1
    end -= 1

# Example
arr = [1, 2, 3, 4, 5]
reverse_array(arr)
print("Reversed array:", arr)  # Output: [5, 4, 3, 2, 1]
```
<br>

## 14. _Remove Duplicates_ from a sorted array without using extra space.

### Problem Statement

Given a **sorted array**, the task is to **remove duplicate elements** in place (using constant space) and return the new length.

### Solution

A two-pointer method provides an efficient solution that removes duplicates **in place** while also recording the new length of the array.

**Algorithm steps**:

1. Initialize `i=0` and `j=1`.
2. Iterate through the array.
   - If `array[i] == array[j]`, move `j` to the next element.
   - If `array[i] != array[j]`, update `array[i+1]` and move both `i` and `j` to the next element.

#### Complexity Analysis

- **Time Complexity**: $O(n)$. Here, $n$ represents the array's length.
- **Space Complexity**: $O(1)$. The process requires only a few additional variables

#### Implementation

Here is the Python code:

```python
def removeDuplicates(array):
    if not array:
        return 0

    i = 0
    for j in range(1, len(array)):
        if array[j] != array[i]:
            i += 1
            array[i] = array[j]

    return i + 1
```
<br>

## 15. Implement a _Queue_ using an array.

### Problem Statement

Implement a **Queue** data structure using a fixed-size array.

### Solution

While a **dynamic array** is a more efficient choice for this purpose, utilizing a standard array helps in demonstrating the principles of queue operations.

- The queue's front should always have a lower index than its rear, reflecting the structure's first-in, first-out (FIFO) nature.
- When the rear pointer hits the array's end, it may switch to the beginning if there are available slots, a concept known as **circular or wrapped around arrays**.

#### Algorithm Steps

1. Initialize the queue: Set `front` and `rear` both to -1.
2. `enqueue(item)`: Check for a full queue then perform the following steps:
   - If the queue is empty (`front = -1, rear = -1`), set `front` to 0.
   - Increment `rear` (with wrapping if needed) and add the item.
3. `dequeue()`: Check for an empty queue then:
   - Remove the item at the `front`.
   - If `front` equals `rear` after the removal, it indicates an empty queue, so set both to -1.

#### Complexity Analysis

- **Time Complexity**:
  - $\text{enqueue}: O(1)$
  - $\text{dequeue}: O(1)$
- **Space Complexity**: $O(n)$

#### Implementation

Here is the Python code:

```python
class Queue:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = self.rear = -1

    def is_full(self) -> bool:
        return self.front == (self.rear + 1) % self.capacity

    def is_empty(self) -> bool:
        return self.front == -1 and self.rear == -1

    def enqueue(self, item):
        if self.is_full():
            print("Queue is full")
            return
        if self.is_empty():
            self.front = self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty")
            return
        if self.front == self.rear:
            self.front = self.rear = -1
        else:
            self.front = (self.front + 1) % self.capacity

    def display(self):
        if self.is_empty():
            print("Queue is empty")
            return
        temp = self.front
        while temp != self.rear:
            print(self.queue[temp], end=" ")
            temp = (temp + 1) % self.capacity
        print(self.queue[self.rear])

# Usage
q = Queue(5)
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)
q.enqueue(5)
q.display()
q.enqueue(6)  # Queue is full
q.dequeue()
q.dequeue()
q.display()
```
<br>



#### Explore all 60 answers here 👉 [Devinterview.io - Array Data Structure](https://devinterview.io/questions/data-structures-and-algorithms/array-data-structure-interview-questions)

<br>

<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

