# 30 Common Big-O Notation Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

#### You can also find all 30 answers here 👉 [Devinterview.io - Big-O Notation](https://devinterview.io/questions/data-structures-and-algorithms/big-o-notation-interview-questions)

<br>

## 1. What is _Big-O Notation_?

**Big O notation** serves as a standardized **measure for algorithmic performance**, focusing on time and space complexity. It's crucial for **comparing algorithms** and understanding their **scalability**.

### Key Concepts

- **Dominant Term**: The notation simplifies complexity expressions to their most significant terms, making them easier to compare.

- **Asymptotic Analysis**: Big O emphasizes how algorithms perform as data scales, offering a high-level understanding of efficiency.

- **Worst-Case Performance**: Big O provides an upper limit on resources needed, offering a conservative estimate for the most challenging scenarios.


### Visual Representation

![Big O Complexity Graph](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/big-o%2Fbig-o-complexity%20(1).jpeg?alt=media&token=fbb27d2b-31bf-4456-8f93-f4b62b0e5344)

### Complexity Classes

- **Constant Complexity** $O(1)$
  - Resources used are independent of the input size.
  - Time Example: Arithmetic operations, Array index access
  - Space Example: Using a fixed-size array
  
- **Logarithmic Complexity** $O(\log n)$
  - Resource usage grows logarithmically with input size.
  - Time Example: Binary search in a sorted array
  - Space Example: Binary tree traversal
  
- **Linear Complexity** $O(n)$
  - Resource usage scales linearly with input size.
  - Time Example: Element search in an unordered list
  - Space Example: Allocating an array proportional to input size
  
- **Linearithmic Complexity** $O(n\log n)$
  - Resource usage grows at a rate between linear and quadratic. Often seen when combining linear and logarithmic operations.
  - Time Example: Efficient sorting algorithms like merge sort and quicksort
  - Space Example: Divide-and-conquer algorithms that decompose the problem
  
- **Quadratic Complexity** $O(n^2)$
  - Resources scale with the square of the input size.
  - Time Example: Algorithms with simple nested loops, e.g., bubble sort
  - Space Example: Creating a two-dimensional matrix based on input size
  
- **Exponential Complexity** $O(2^n)$
  - Resource usage doubles (or increases exponentially) with each additional unit of input.
  - Time Example: Generating all subsets of a set
  - Space Example: Recursive algorithms that double the size of the call stack for each input element

### Practical Applications

1. **Resource Management**: It helps in pre-allocating sufficient resources, especially in constrained environments.
2. **Reliability**: Provides a performance guarantee, crucial in time-sensitive tasks.
3. **Optimization**: Aids in identifying bottlenecks and areas for potential improvement, ensuring the algorithm is as efficient as possible.

### Code Example: Linear Search

Here is the Python code:

```python
def linear_search(arr, target):
    for i, num in enumerate(arr):
        if num == target:
            return i
    return -1
```

- **Worst-Case**: The target is not in the list, resulting in $O(n)$ time complexity.
- **Best-Case**: The target is the first element, with $O(1)$ time complexity.
- **Average-Case**: Simplifies to $O(n)$ as every element has an equal chance of being the target.
<br>

## 2. Explain the difference between _Big-O_, _Big-Theta_, and _Big-Omega_ notations.

Let's discuss the meanings and practical applications of three important notations:

### Big-O Notation: Upper Bound

Big-O defines the **worst-case scenario** of an algorithm's performance. It gives an upper limit, denoted as $O(g(n))$, for the order of growth of a function.

$$
0 \leq f(n) \leq k \cdot g(n), \quad \exists n_0, k > 0
$$

In simpler terms, if an algorithm has a time complexity of $O(n^2)$, it means that the worst-case runtime will grow at worst as $n^2$.

### Big-Theta Notation: Tight Bound

Big-Theta represents the **exact growth rate**, showing both an upper and lower limit for a function. It is denoted as $\Theta(g(n))$.

$$
0 \leq k_1 \cdot g(n) \leq f(n) \leq k_2 \cdot g(n), \quad \exists n_0, k_1, k_2 > 0
$$

An algorithm with a time complexity of, say, $n^2$, will have a worst-case performance that grows as $n^2$ and no worse than $n^2$.

### Big-Omega Notation: Lower Bound

Big-Omega provides the **best-case time complexity**, giving a lower limit for the function's growth rate. It is denoted as $\Omega(g(n))$.

$$
0 \leq k \cdot g(n) \leq f(n), \quad \exists n_0, k > 0
$$

If an algorithm's best-case time complexity is $k \cdot n$, it means that the best-case performance will grow at least as $k \cdot n$.


### Use Case

An algorithm requiring a **specific time-bound** to be effective might be suitably described using Big-Theta notation. Conversely, algorithms designed for different use cases, such as adaptive sorting algorithms, are better depicted with Big-Omega and Big-Oh notations.
<br>

## 3. Describe the role of _constants and lower-order terms_ in Big-O analysis.

Understanding the role of **constants** and **lower-order terms** in Big-O analysis helps to differentiate between performance that's good in practice versus good in theory. Evaluation of these factors leads to the most accurate Big-O classification for an algorithm, providing practical insights into its efficiency.

### Constants ($c$)

Constants are numerical multipliers in functions that represent an **exact number of operations** an algorithm performs. However, in the context of Big-O analysis, they are not typically included as they do not affect the overall **order of magnitude**.

For example, an algorithm might have a runtime of $7n^2$. It is still classified as $O(n^2)$, and the leading 7 is considered negligible in the context of **asymptotic analysis**. This aligns with the principle that for sufficiently large $n$, the multiplication by a constant becomes less significant.

### Lower-Order Terms

**Lower-order terms**, also referred to as "small-oh", correspond to the **lower-graded factors** in a given Big-O function and provide a **more detailed view** of the algorithm's behavior.

When dealing with multiple terms:

- **The dominant term** is retained for Big-O representation as it is the most influential for larger inputs.
- Lower-order terms and constants are omitted for the same reason.

For example, if an algorithm has a complexity of $3n^3 + 100n^2 + 25n$, the Big-O simplification is $O(n^3)$ because the term with $n^3$ is the most significant for large inputs.
<br>

## 4. Give examples of how _amortized analysis_ can provide a more balanced complexity measure.

Let's explore how **amortized analysis** can lead to more balanced complexity measures by considering the following algorithms:

### 1. Dynamic Array List

This data structure combines the benefits of an array's fast random access with dynamic resizing. While single insertions might occasionally trigger expensive resizing, most insertions are quicker. Through amortized analysis, the average cost per operation is $O(1)$.

#### Code Example: Dynamic Array List

Here is the Python code:

```python
import ctypes

class DynamicArray:
    def __init__(self):
        self.n = 0  # Not the actual size
        self.array = self.make_array(1)

    def insert(self, elem):
        if self.n == len(self.array):
            self._resize(2 * len(self.array))
        self.array[self.n] = elem
        self.n += 1

    def _resize(self, new_capacity):
        new_array = self.make_array(new_capacity)
        for i in range(self.n):
            new_array[i] = self.array[i]
        self.array = new_array

    def make_array(self, cap):
        return (cap * ctypes.py_object)()
```

### 2. Binary Search

Despite its primarily logarithmic time complexity, there are situations where **Binary Search** can exceed this efficiency through repeated operations that halve the search range. With amortized analysis, such "good" or "lucky" scenarios are considered, leading to a balanced time complexity measure of $O(\log n)$.

#### Code Example: Binary Search

Here is the Python code:

```python
def binary_search(arr, x):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

### 3. Fibonacci with Caching

Standard **Fibonacci** calculations exhibit $O(2^n)$ complexity, arising from the recursive tree's 2^n leaves. However, with caching, the same subproblems get solved only once, reducing the tree height and time complexity to a linear $O(n)$.

#### Code Example: Caching Fibonacci

Let's use Python:

```python
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]
```
<br>

## 5. Describe how the _coefficients of higher-order terms_ affect Big-O Notation in practical scenarios.

Although the **dominant term** typically defines a function's time complexity, coefficients of higher-order terms are not necessarily moot. Their impact can be tangible, especially in **real-world** applications.

### Interplay of Coefficients and Big-O

Let's consider a scenario where we need to compare time complexities based on Big-O notation.

1. **Low Order, Large Coefficient**: $3n + 500$ vs $6n + 10$ where $n$ is small:
   - Big-O suggests $O(n)$
   - In practice, `3n` tends to outperform `6n`, especially for diminutive `n`.

2. **Low Order, Small Coefficient**: $0.05n + 3$ vs $0.1n + 1.5$ where $n$ is large:
   - Big-O suggests $O(n)$
   - With sizeable `n`, the constant factors can still have an **observable** cumulative effect.

3. **High Order, Small Coefficient**: $0.00001n^2 + 10$ vs $0.0001n^2 + 1$ :
   - Big-O suggests $O(n^2)$
   - For large $n$, the leading term becomes so dominant that the impact of the coefficients is relegated.

4. **High Order, Large Coefficient**: $10n^2 + 5n + 2$ vs $2n^2 + 3n + 1$ for all $n$:
   - Big-O suggests $O(n^2)$
   - Coefficients can modify performance marginally, but not enough to alter the **asymptotic behavior**.

### Real-world Examples

Many algorithms exhibit these characteristics.

- Linear-time algorithms are often referred to as $O(n)$ even though they can be $4n + 2$ as a worst-case scenario. However, for large datasets, the difference between $4n$ and $0.5n$ can be substantial, as opposed to the theoretical constant factor of 4.
<br>

## 6. Explain how _probabilistic algorithms_ can have a different Big-O Notation compared to deterministic ones.

In this scenario, let's discuss the problems for which probabilistic algorithms can provide approximate solutions in **expected polynomial time**, even though a deterministic algorithm might require exponential time.

### Problem Statement: Graph Connectivity

The goal is to find out whether a graph $G$ is connected, i.e., there is a path between every pair of vertices.

####  Deterministic Algorithm  

The standard "Breadth-First Search" or "Depth-First Search" algorithms **requires linear time** in $O(|V|+|E|)$ to determine graph connectivity.

#### Probabilistic Algorithm

The "Randomized Incremental Algorithm" emulates a different strategy. It initially considers the graph to be disconnected and then adds edges one by one, checking for graph connectivity after each addition. This approach, on average, **operates in expected constant time** before the graph becomes connected and $O(|V|)$ when the graph becomes connected.

### Complexity Analysis

#### Deterministic Algorithm

The $|V| + |E|$ complexity persists for graphs with a large expected vertex count to vertex-edge count ratio.

#### Probabilistic Algorithm

- The first stage, where the graph is disconnected, **expects a constant time operation** (usually ends prematurely).
- The second stage, activated with a probability smaller than $(2/3)^{3 \sqrt{ 2 n} }$, which can be bounded above by a constant $c$, dictates an $O(c|V|)$ complexity.

Hence, in expected polynomial time, $O( n^2)$, the algorithm combines the preliminary constant time and the upper-bounded linear time.
<br>

## 7. Analyze the Big-O _time complexity_ of array operations.

Let's look at common array operations and their Big O-time complexities.

### Time Complexity in Arrays

#### Access $O(1)$

Accessing an array element by its index is a **constant-time operation**. 

#### Search $O(n)$

For an unsorted array, searching for a specific value might have a **worst-case time complexity** of $O(n)$ as each element may need to be inspected in order to locate the target.

For a sorted array, binary search can be implemented, reducing the complexity to $O(\log n)$.

#### Insert $O(n)$

Inserting an element at a specific index might require elements after that index to be 'shifted' to the next index. This shifting operation, especially in an array list, is a **linear-time process**.

Appending an element (insertion at the end) can generally be done in **amortized constant time**, unless the array needs to be resized, in which case it becomes a linear-time operation.

#### Delete $O(n)$

Deleting an element at a specific index will likely require subsequent elements to be shifted, which is a linear-time operation. 

If you delete the last element, it's a **constant-time operation*.

#### Other Operations

- **Sort**: Many popular sorting algorithms $e.g., quicksort, mergesort$ have an average and worst-case time complexity of $O(n \log n)$.
- **Transpose**: Rearranging elements or swapping adjacent elements, such as in the "transpose" operation, is a **constant-time process**.

- **Merge in a Sorted Order**: If both arrays are sorted into progressively larger or smaller elements, then it is possible — for many algorithms — to "merge" the two arrays into one array still sorted in a time linear to the total number of elements (remember, sorting entire datasets usually is a $O(n \log n)$ operation).

Remember, these complexities are general. The **actual performance can be influenced by the hardware** and specific programming language. By knowing these complexities, you can better understand and predict the efficiency of your code.
<br>

## 8. Discuss the Big-O _space complexity_ of using _linked lists_.

While **Linked Lists** offer $O(1)$ insertion and deletion operations, a key consideration comes with their **space complexity** due to the "per-node memory overhead". Each node occupies memory for both its data and a pointer, making their space complexity $O(n)$.

That means, in the worst case, they could occupy as much space as an $O(n)$-sized array, without offering the contiguous memory benefits of arrays. Different types of Linked Lists can affect space complexity, as smaller pointers in types like "Doubly Linked Lists" occupy more space, and auxiliary pointers in "Auxiliary Pointers Doubly Linked Lists" can further exacerbate the overhead. The standard for space is "Singly Linked List".

### Memory Overhead in Different Types of Linked Lists

- **Singly Linked List**: Each node has a single pointer, typically 4-8 bytes on a 64-bit system, in addition to the data. Hence, $O(1)$ pointers per node makes it $O(n)$ in terms of space.
  
- **Doubly Linked List**: Every node has two pointers, requiring either 8-16 bytes. While it's still $O(1)$ in terms of pointers, it does have comparatively higher overhead than a singly linked list.
  
- **Circularly Linked List**: Similar to singly and doubly linked lists, but the tail node points back to the head. It doesn't impact the space complexity.

- **XOR Linked List**: Uses bitwise XOR operations to store only one reference (as opposed to two in doubly linked lists). However, it's quite complex to implement and maintain, so it's used mainly for educational or theoretical purposes.

- **Auxiliary Pointer Singly Linked and Doubly Linked Lists**: These lists have an additional pointer, increasing the per-node memory requirements and thus the space complexity.
<br>

## 9. Compare the Big-O complexities of various _sorting algorithms_.

Let's look at the **Big-O complexities** of various fundamental sorting algorithms.

### Bubble Sort

**Complexity**:
- Best Case: $O(n)$ when the list is already sorted due to the flag $swapped$.
- Worst Case: $O(n^2)$ when each element requires $n-1$ comparisons and there are $n$ elements.

### Selection Sort

**Complexity**:
- Best Case: $O(n^2)$ because $n-1$ passes are still made to find the largest element.
- Worst Case: $O(n^2)$ for the same reason.

### Insertion Sort

**Complexity**:
- Best Case: $O(n)$ when the list is already sorted.
- Worst Case: $O(n^2)$ occurs when the list is sorted in reverse order. $n-1$ comparisons and assignments are made for each of the $n$ elements.

### Merge Sort

**Complexity**:
- Best Case: $O(n \log n)$ as it partitions the list equally and takes $n \log n$ time to merge.
- Worst Case: $O(n \log n)$ due to the same reasons.

### Quick Sort

**Complexity**:
- Best Case: $O(n \log n)$ when the chosen pivot divides the list evenly all the time.
- Worst Case: $O(n^2)$ when the pivot is always one of the smallest or largest elements. This is more likely if the list is already partially sorted.

### Heap Sort

**Complexity**:
- Best Case: $O(n \log n)$ - The heapify operation takes $O(n)$ time and the heap operations take $O(\log n)$ time.
- Worst Case: $O(n \log n)$ for the same reason.

### Counting Sort

**Complexity**:
- Best Case: $O(n+k)$ for $k$ distinct elements with $O(n)$ time complexity.
- Worst Case: $O(n+k)$ for $k$ distinct elements.

### Radix Sort

**Complexity**:
- Best Case: $O(nk)$ where $k$ is the number of digits in the maximum number. Actual complexity is $O(dn+k)$.
- Worst Case: $O(nk)$ similar to the best case, but it varies based on the distribution of data.

### Bucket Sort

**Complexity**:
- Best Case: $O(n+k)$ where $k$ is the number of buckets or partitions. It can be made $O(n)$ when $k=n^2$.
- Worst Case:  $O(n^2)$ when all elements fall into a single bucket.

### Shell Sort

**Complexity**:
- Best Case: Depends on the gap sequence, but typically $O(n \log n)$.
- Worst Case: $O(n (\log n)^2)$ typically but can vary based on the gap sequence.
<br>

## 10. Evaluate the Big-O _time complexity_ of _binary search_.

**Binary Search** is a Divide and Conquer algorithm primarily used on **sorted lists** to efficiently locate an element. It offers a time complexity of $O(\log n)$.

### Key Features

- **Notable Efficiency**: Binary Search outperforms linear search, which has a time complexity of $O(n)$, especially on longer lists.
- **Recursive or Iterative Implementation**: The algorithm can be implemented using a recursive or an iterative approach.
- **Memory Constraint**: Binary Search navigates a list in memory without needing additional data structures, making it ideal for large datasets with limited memory resources.

### Time Complexity Analysis

Binary Search's time complexity is evaluated using a **recurrence relation**, benefiting from the Master Theorem:

$$
T(n) = T\left(\frac{n}{2}\right) + 1
$$

where:
- $T(n)$ is the time complexity.
- The constant work done within each recursive call is $1$.
- The algorithm divides the list into sublists of length $\frac{n}{2}$.

### Pseudocode

Here is the Pseudocode:

```plaintext
Let min = 0 and max = n-1
While min <= max
    Perform middle calculation
    If arr[middle] = target
        : target found, return middle
    If arr[middle] < target
        : Discard the left sublist (set min = middle + 1)
    Else
        : Discard the right sublist (set max = middle - 1)
Return "not found"
```
### Time Complexity Breakdown

- **Loop Control**: Although not always true, the loop's primary control generally checks  `min <= max`. Each iteration, therefore, helps to halve the size of the sublist under consideration. 

- **Iteration Count**: The number of iterations, in the worst-case, typically determines how efficiently Binary Search can reduce the search space. Also, the overall complexity can be expressed as:

$$
1 + 2 + 4 + \ldots 2^k
$$

The last term ranges up to $2^k$, which may not be directly on point but should be close enough to assess the overall time-complexity of the algorithm.

### In-Place vs. Memoization/Tabulation

"Binary Search" employs an **in-place** approach. It navigates the input list without needing any auxiliary structures, demonstrating a space complexity of $O(1)$.
<br>

## 11. Determine the Big-O _time and space complexities_ of _hash table_ operations.

Let's discuss the **time** and **space** complexities associated with basic **Hash Table** operations.

### Key Operations

- **Search** (Find Element) - $O(1)$
    - The hash function identifies the location of an element within the unique hash table.

- **Insert** (Add New Element) - $O(1)$
    - The hash function determines placement, usually in constant time.

- **Delete** (Remove Element) - $O(1)$
    - Similar to "Insert", this step generally requires only constant time.

#### Resizing Mechanism

- **Resize** (Rehash for Dynamic Tables) - $O(n)$
    - Regarding amortized time, resizing arrays takes linear time but happens infrequently. Thus, the average time is amortized over many operations.

- **Rehash** (during Resizing) - $O(n)$
    - Re-inserting all elements happens linearly with the number of items in the table.

### Why is Search $O(1)$?

In an ideal setting, each key is mapped to a unique hash, and thus a **unique** table slot. When dealing with **collisions** (two keys hashing to the same slot), some algorithms outshine others. 

The time it takes to resolve a collision is crucial in forming the table's amortized bounds. Techniques include separate chaining and open addressing.

### Complexity Summary

- Worst-Case Time: $O(n)$ (All keys collide)
- Amortized Time: $O(1)$
- Worst-Case Space: $O(n)$
<br>

## 12. Discuss the Big-O complexities of _tree_ operations, including binary search trees and AVL trees.

**Binary Search Trees** (BST) are efficient for both search and insert operations, with average time complexity of $O(\log n)$. However, sub-optimal structures can lead to worst-case time complexity of $O(n)$. 

### Key Characteristics

- **Search (Best & Worst)**: $O(\log n)$ Best case: Root of the tree is the target. Each level eliminates half of the remaining nodes. Worst case: Tree is a single linked list.
- **Insert**: $O(\log n)$. Ensures the tree remains balanced.

### Code Example: BST Insertion

Here is the Python code:

```python
class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

def insert(root, value):
    if root is None:
        return Node(value)
    if value < root.value:
        root.left = insert(root.left, value)
    else:
        root.right = insert(root.right, value)
    return root
```

### AVL Trees

**AVL Trees** maximize search efficiency by remaining balanced. They guarantee worst-case time complexities of $O(\log n)$ for search and insert.

#### Rotations for Balancing

- **Single Rotation**: Utilized when an imbalance arises due to operations on either the left or right subtree of a node.
  - Left Rotation: Resolves a `left-heavy` situation.
  - Right Rotation: Resolves a `right-heavy` situation.
  
- **Double Rotation**: Employed when a node's imbalance is due to operations stemming from both subtrees.
  - Left-Right (or Right-Left) Rotation: Applies both a left and a right (or vice versa) rotation to restore balance.

#### Example of Rotation

Here is the visual representation:

Before Rotation:

```
  A(-2)
  /
B(0)
  \
   C(0) 
```

After rotation:

```
   B(0)
  / \
C(0) A(0)
```

#### Code Example: AVL Insertion

Here is the Python code for AVL Tree:

```python
class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value
        self.height = 1

def insert(root, value):
    if root is None:
        return Node(value)
    if value < root.value:
        root.left = insert(root.left, value)
    else:
        root.right = insert(root.right, value)

    root.height = 1 + max(get_height(root.left), get_height(root.right))
    balance = get_balance(root)
    
    if balance > 1 and value < root.left.value:
        return right_rotate(root)
    if balance < -1 and value > root.right.value:
        return left_rotate(root)
    if balance > 1 and value > root.left.value:
        root.left = left_rotate(root.left)
        return right_rotate(root)
    if balance < -1 and value < root.right.value:
        root.right = right_rotate(root.right)
        return left_rotate(root)
    
    return root

def get_height(node):
    return node.height if node else 0

def get_balance(node):
    return get_height(node.left) - get_height(node.right)

def right_rotate(z):
    y = z.left
    t3 = y.right
    y.right = z
    z.left = t3
    z.height = 1 + max(get_height(z.left), get_height(z.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    return y

def left_rotate(z):
    y = z.right
    t2 = y.left
    y.left = z
    z.right = t2
    z.height = 1 + max(get_height(z.left), get_height(z.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    return y
```
<br>

## 13. Analyze the Big-O complexity of _graph_ algorithms, including traversal and shortest path algorithms.

**Graph algorithms** vary in their computational requirements. The time complexity is often measured in Big O notation.

Here is the detailed breakdown:

### Graph Traversal

#### Depth-First Search (DFS)

- **Time Complexity**: $O(V + E)$ - Visiting all vertices and edges once.
- **Space Complexity**: $O(V)$ - Underlying stack depth.

#### Breadth-First Search (BFS)

- **Time Complexity**: $O(V + E)$ - Visiting all vertices and edges once.
- **Space Complexity**: $O(V)$ - Queue typically contains all vertices.

### Shortest Path Algorithms

#### Dijkstra's Algorithm

- **Time Complexity**: $O((V + E) \log V)$ - Efficient in practice on sparse graphs.
- **Space Complexity**: $O(V)$ - Using a priority queue.

#### Bellman-Ford Algorithm

- **Time Complexity**: $O(VE)$ - Slower but robust, suitable for graphs with negative edges or cycles.
- **Space Complexity**: $O(V)$ - Single-source shortest path tree.

- **Negative Cycle Detection**: $O(V \cdot E)$

#### Floyd-Warshall Algorithm

- **Time Complexity**: $O(V^3)$ - Finds all-pairs shortest paths.
- **Space Complexity**: $O(V^2)$
- Dynamic Programming formulation

#### A* Algorithm

- **Time Complexity**: $O(\log V)$ on average - Admissible and consistent heuristics guide the search. Failing to do so can lead to higher time complexities.
- Efficiency in practice often relies on a good heuristic function $h(n)$.

#### Johnson's Algorithm

- **Time Complexity**: $O(VE + V^2 \log V)$ - Combines Bellman-Ford and Dijkstra's algorithms with the help of potential functions, making it particularly efficient for sparse graphs.
<br>

## 14. Discuss time and space complexities of various _heap operations_.

**Heaps** are specialized trees that enable fast operations such as insertions, deletions, and min/max lookups. They are widely used in priority queues and sort algorithms. Heaps come in two varieties: the min-heap, where the smallest key is at the root, and the max-heap, which places the largest key at the root.

### Array Operations

- **Index**: $i$  
  nodes: $\left\lfloor \frac{i-1}{2} \right\rfloor$ parent, $2i+1$ left child, $2i+2$ right child
- **Parent, Left Child, Right Child**:
  - Parent: $\text{index} = \left\lfloor \frac{\text{index} -1}{2} \right\rfloor$
  - Left Child: $\text{index} = 2 \times \text{index} + 1$
  - Right Child: $\text{index} = 2 \times \text{index} + 2$

### Operations Complexity

- **Insertion**: $O(\log n)$  
  The element to be inserted is placed at the leaf level, and then "heapified" (moved up).
  - Time: This involves up to $\log n$ swaps to restore the heap's structure.

  - Space: The operation may incur up to $O(\log n)$ space due to its iterative nature.

- **Deletion**: $O(\log n)$  
  The root element is replaced with the last node, and then "heapified" (moved down).
  - Time: This includes up to $\log n$ swaps to preserve the heap property.

  - Space: It requires $O(1)$ space, as it performs in-place swaps.

- **Min/Max Lookups**: $O(1)$  
  The smallest or largest element, respectively, can be found at the root.
  - Time: This is a direct and constant time operation.

  - Space: It does not utilize additional space.

- **Extract Min/Max**: $O(\log n)$  
  Same as Deletion.
  - Time: Like deletion, the process may take up to $\log n$ swaps.

  - Space: It requires $O(1)$ space, as it performs in-place swaps.

### Code Example: Heap Operations

Here is the Python code:

```python
import heapq

# Create a min-heap
min_heap = []
heapq.heapify(min_heap)

# Insert elements:
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 1)

# Delete/ Extract Min
print(heapq.heappop(min_heap))  # Output: 1

# Min/Max Lookup
print(min_heap[0])  # Output: 3 (the only remaining element)

# Create a max-heap
max_heap = []
heapq._heapify_max(max_heap)

# Insert elements into max-heap
heapq._heappush_max(max_heap, 3)
heapq._heappush_max(max_heap, 1)

# Delete/ Extract Max
print(heapq._heappop_max(max_heap))  # Output: 3
```
<br>

## 15. Provide examples of _space-time tradeoffs_ in algorithm design.

**Time-space tradeoffs** characterize algorithms that can be optimized for either time or space, but generally not both.

### Examples of Space-Time Tradeoffs

#### Data Compression Algorithms

- **Time**: Using a simple algorithm for compression or decompression can save computing time. 
- **Space**: More sophisticated compression techniques can require additional space but reduce memory's overall size.

#### Index for Text Queries

- **Time**: A more in-depth indexing mechanism can speed up word lookups to improve search time.
- **Space**: It consumes extra memory to store the index.

#### Databases - Persistent Indices

- **Time**: Persisting indices on disk rather than recreating them frequently can improve query time.
- **Space**: Requires storage on disk.

### Code Example: Simple vs. Huffman Compression

Here is the Python code:

```python
# Simple compression algorithm (time optimized)
simple_compress = lambda data: ''.join(bin(ord(char))[2:].zfill(8) for char in data)

# Huffman compression algorithm (space optimized)
from heapq import heappop, heappush, heapify

def huffman_compress(data):
    freq = {}
    for char in data:
        freq[char] = freq.get(char, 0) + 1
    heap = [[f, [char, ""]] for char, f in freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    codes = dict(heappop(heap)[1:])
    return ''.join(codes[char] for char in data), codes
```
<br>



#### Explore all 30 answers here 👉 [Devinterview.io - Big-O Notation](https://devinterview.io/questions/data-structures-and-algorithms/big-o-notation-interview-questions)

<br>

<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

