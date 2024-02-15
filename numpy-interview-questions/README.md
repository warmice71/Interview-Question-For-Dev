# 70 Common NumPy Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/machine-learning-and-data-science/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fmachine-learning-and-data-science-github-img.jpg?alt=media&token=c511359d-cb91-4157-9465-a8e75a0242fe" alt="machine-learning-and-data-science" width="100%">
</a>
</p>

#### You can also find all 70 answers here 👉 [Devinterview.io - NumPy](https://devinterview.io/questions/machine-learning-and-data-science/numpy-interview-questions)

<br>

## 1. What is _NumPy_, and why is it important in _Machine Learning_?

**NumPy** (Numerical Python) is a fundamental library in Python for numerical computations. It's a versatile tool primarily used for its advanced **multi-dimensional array support**.

### Key Features

- **Task-Specific Modules**: NumPy offers a rich suite of mathematical functions in areas such as linear algebra, Fourier analysis, and random number generation.

- **Performance and Speed**:
  - Enables vectorized operations.
  - Many of its core functions are implemented in `C` for optimized performance.
  - It uses contiguous blocks of memory, providing efficient caching and reducing overhead during processing.

- **Broadcasting**: NumPy allows combining arrays of different shapes during arithmetic operations, facilitating streamlined computation.

- **Linear Algebra**: It provides essential linear algebra operations, including matrix multiplication and decomposition methods.

### NumPy Arrays

- **Homogeneity**: NumPy arrays are homogeneous, meaning they contain elements of the same data type.
- **Shape Flexibility**: Arrays can be reshaped for specific computations without data duplication.
- **Simple Storage**: They use efficient memory storage and can be created from regular Python lists.

### Performance Benchmarks

1. **Contiguous Memory**: NumPy arrays ensure that all elements in a multi-dimensional array are stored in contiguous memory blocks, unlike basic Python lists.

2. **No Type Checking**: NumPy arrays are specialized for numerical data, so they don't require dynamic type checks during operations.

3. **Vectorized Computing**: NumPy obviates the need for manual looping, making computations more efficient.

### Code Example: NumPy and Efficiency

Here is the Python code:

```python
# Using Python Lists
python_list1 = [1, 2, 3, 4, 5]
python_list2 = [6, 7, 8, 9, 10]
result = [a + b for a, b in zip(python_list1, python_list2)]

# Using NumPy Arrays
import numpy as np
np_array1 = np.array([1, 2, 3, 4, 5])
np_array2 = np.array([6, 7, 8, 9, 10])
result = np_array1 + np_array2
```
In the above example, both cases opt for element-wise addition, yet the NumPy version is more concise and efficient.
<br>

## 2. Explain how _NumPy arrays_ are different from _Python lists_.

**NumPy arrays** and **Python lists** are both versatile **data structures**, but they have distinct advantages and use-cases that set them apart.

### Key Distinctions

#### Storage Mechanism

- **Lists**: These are general-purpose and can store various data types. Items are often stored contiguously in memory, although the list object itself is an array of references, allowing flexibility in item sizes.
- **NumPy Arrays**: These are designed for homogeneous data. Elements are stored in a contiguous block of memory, making them more memory-efficient and offering faster element access.

#### Underlying Optimizations

- **Lists**: Are not specialized for numerical operations and tends to be slower for such tasks. They are dynamic in size, allowing for both append and pop.
- **NumPy Arrays**: Are optimized for numerical computations and provide vectorized operations, which can dramatically improve performance. Array size is fixed upon creation.

### Performance Considerations

- **Memory Efficiency**: NumPy arrays can be more memory-efficient, especially for large datasets, because they don't need to store type information for each individual element.
- **Element-Wise Operations**: NumPy's vectorized operations can be orders of magnitude faster than traditional Python loops, which are used for element-wise operations on lists.
- **Size Flexibility**: Lists can grow and shrink dynamically, which may lead to extra overhead. NumPy arrays are more memory-friendly in this regard.

#### Use in Machine Learning

- **Python Lists**: Typically used for general data-handling tasks, such as reading in data before converting it to NumPy arrays.
- **NumPy Arrays**: The foundational data structure for numerical data in Python. Most numerical computing libraries, including TensorFlow and scikit-learn, work directly with NumPy arrays.
<br>

## 3. What are the main _attributes_ of a _NumPy ndarray_?

A NumPy `ndarray` is a multi-dimensional array that offers efficiency in numerical operations. Much of its strength comes from its **resilience with large datasets** and **agility in mathematical computations**.

### Main Attributes

- **Shape**: A tuple representing the size of each dimension.
- **Data Type (dtype)**: The type of data stored as elements in the array.
- **Strides**: The number of bytes to "jump" in memory to move from one element to the next in each dimension.

### NumPy Examples:

#### Shape Attribute

```python
import numpy as np

# 1D Array
v = np.array([1, 2, 3])
print(v.shape)  # Output: (3,)

# 2D Array
m = np.array([[1, 2, 3], [4, 5, 6]])
print(m.shape)  # Output: (2, 3)
```

#### Data Type Attribute

```python
import numpy as np

arr_int = np.array([1, 2, 3])
print(arr_int.dtype)  # Output: int64

arr_float = np.array([1.0, 2.5, 3.7])
print(arr_float.dtype)  # Output: float64
```

#### Strides Attribute

The **strides** attribute defines how many bytes one must move in memory to go to the next element along each dimension of the array. If **`x.strides = (10,1)`**, this means that:


- Moving one element in the last dimension, we move **1** byte in memory --- as it is a **float64**.
- Moving one element in the first dimension, we move **10** bytes in memory.

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
print(x.strides)  # Output: (6, 2)
```
<br>

## 4. How do you create a _NumPy array_ from a regular _Python list_?

### Problem Statement

The task is to create a **NumPy array** from a standard Python list.

### Solution

Several routes exist to transform a standard Python list into a NumPy array. Regardless of the method, it's crucial to have the `numpy` package installed.

#### Using `numpy.array()`

This is the most straightforward method.

#### Implementation

Here, I demonstrate how to convert a basic Python list to a NumPy array with `numpy.array()`. While it works for most cases, be cautious with nested lists as they have significant differences in behavior compared to Python lists.

#### Code

Here's the Python code:

```python
import numpy as np

python_list = [1, 2, 3]

numpy_array = np.array(python_list)
print(numpy_array)
```

#### Output

The output displays the NumPy array `[1 2 3]`. 

#### Using `numpy.asarray()`

This is another method to convert a Python list into a NumPy array. The difference from `numpy.array()` is primarily in how it handles inputs like other NumPy arrays and nested lists.

#### When to Use `numpy.asarray()`

The function `numpy.asarray()` is beneficial when you're uncertain whether the input is a NumPy array or a list. It converts non-array types to arrays but leaves already existing NumPy arrays unchanged.

#### Using `numpy.fromiter()`

This method is useful when you have an iterable and want to create a NumPy array from its elements. An important point to consider is that the iterable is consumed as part of the array-creation process.

#### Using `numpy.arange()` and `numpy.linspace()`

If your intention is to create sequences of numbers, such as equally spaced data for plotting, NumPy offers specialized methods.

- `numpy.arange(start, stop, step)` generates an array with numbers between `start` and `stop`, using `step` as the increment.
  
- `numpy.linspace(start, stop, num)` creates an array with `num` equally spaced elements between `start` and `stop`.
<br>

## 5. Explain the concept of _broadcasting_ in _NumPy_.

**Broadcasting** in NumPy is a powerful feature that enables efficient operations on arrays of different shapes without explicit array replication. It works by duplicating the elements along different axes and then carrying out the operation through these 'virtual' repetitions.

### Broadcasting Mechanism

1. **Axes Alignment**: Arrays with fewer dimensions are padded with additional axes on their leading side to match the shape of the other array.

2. **Compatible Dimensions**: For two arrays to be broadcast-compatible, at each axis, their sizes are either equal or one of them is 1.

### Example: Adding Scalars to Arrays

When adding a scalar to an array, it's as if the scalar is broadcast to match the shape of the array before the addition:

```python
import numpy as np

arr = np.array([1, 2, 3])
scalar = 10
result = arr + scalar

print(result)  # Outputs: [11, 12, 13]
```

### Visual Representation

The example below demonstrates what happens at each step of the **three-dimensional** array addition `arr` + `addition_vector`:

```python
import numpy as np

arr = np.array(
    [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]
    ]
)

addition_vector = np.array([1, 10, 100])
sum_result = arr + addition_vector

print(f"Array:\n{arr}\n\nAddition Vector:\n{addition_vector}\n\nResult:\n{sum_result}")
```

The broadcasting process, along with the output, is visually depicted in the code.

### Real-world Application: Visualizing Multidimensional Data

NumPy broadcasting is invaluable in applications where visualizing or analyzing **multidimensional named data** is essential, permitting easy manipulations without resorting to loops or explicit data copying.

For instance, matching a three-dimensional RGB image (represented by a 3D NumPy array) with a 1D intensity array prior to modifying the image's pixels is simplified through broadcasting.
<br>

## 6. What are the _data types_ supported by _NumPy arrays_?

**NumPy** deals with a variety of data types, which it refers to as **dtypes**.

### NumPy Data Types

NumPy data types build upon the primitive types offered by the machine:

1. **Basic Types**: `int`, `float`, and `bool`.
   
2. **Floating Point Types**: `np.float16`, `np.float32`, and `np.float64`.

3. **Complex Numbers**: `np.complex64` and `np.complex128`.
   
4. **Integers**: `np.int8`, `np.int16`, `np.int32`, and `np.int64`, along with their unsigned variants.
   
5. **Boolean**: Represents `True` or `False`.

6. **Strings**: `np.str_`.

7. **Datetime64**: Date and time data with time zone information.

8. **Object**: Allows any data type.
   
9. **Categories and Structured Arrays**: Specialized for categorical data and structured records.

**NumPy** enables you to define arrays with the specific data types:

```python
import numpy as np

my_array = np.array([1, 2, 3])  # Defaults to int64
float_array = np.array([1.5, 2.5, 3.5], dtype=np.float16)
bool_array = np.array([True, False, True], dtype=np.bool)

# Specifying the dtype of string
str_array = np.array(['cat', 'dog', 'elephant'], dtype=np.str_)
```
<br>

## 7. How do you inspect the _shape_ and _size_ of a _NumPy array_?

You can examine the **shape** and **size** of a NumPy array using two key attributes: `shape` and `size`.

### Code Example: Shape and Size Attributes

Here is the Python code:

```python
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Access shape and size attributes
shape = arr.shape
size = arr.size

print("Shape:", shape)  # Outputs: (2, 3)
print("Size:", size)    # Outputs: 6
```
<br>

## 8. What is the difference between a _deep copy_ and a _shallow copy_ in _NumPy_?

In NumPy, you can create **shallow** and **deep** copies using the `.copy()` method. 

Each type of copy preserves ndarray data in a different way, impacting their link to the original array and potential impact of one on the other.


### Shallow Copy

A shallow copy creates a new array object, but it does not duplicate the actual **data**. Instead, it points to the data of the original array. Modifying the shallow copy will affect the original array and vice versa.


The shallow copy is a view of the original array. You can create it either by calling `.copy()` method on an array or using a slice operation.

Here is an example:

```python
import numpy as np

original = np.array([1, 2, 3])
shallow = original.copy()

# Modifying the shallow copy
shallow[0] = 100  # Modifications do not affect the original
print(shallow)  # [100, 2, 3]
print(original)  # [1, 2, 3]

# Modifying the original
original[1] = 200
print(shallow)  # [100, 200, 3]  # The shallow copy is affected
print(original)  # [1, 200, 3]
```

### Deep Copy

A deep copy creates a new array as well as creates separate copies of arrays and their data. **Modifying a deep copy does not affect the original array**, and vice versa.

In NumPy, you can achieve a deep copy using the same `.copy()` method but with the `order='K'` parameter, or by using `np.array(array, copy=True)`. Here is an example:

```python
import numpy as np

# For a 1D array:
original_deep = np.array([1, 2, 3], copy=True)  # This creates a deep copy
original_deep[0] = 100  # Modifications do not affect the original
print(original_deep)  # [100, 2, 3]
print(original)  # [1, 2, 3]

# For a 2D array:
original_2d = np.array([[1, 2], [3, 4]])
deep_2d = original_2d.copy(order='K')  # Deep copy with 'K'
deep_2d[0, 0] = 100
print(deep_2d)  # [[100, 2], [3, 4]]
print(original_2d)  # [[1, 2], [3, 4]]
<br>

## 9. How do you perform _element-wise operations_ in _NumPy_?

**Element-wise operations** in NumPy use broadcasting to efficiently apply a single operation to multiple elements in a NumPy array.

### Key Functions

- **Basic Math Functions**: `np.add()`, `np.subtract()`, `np.multiply()`, `np.divide()`, `np.power()`, `np.mod()`
- **Trigonometric Functions**: `np.sin()`, `np.cos()`, `np.tan()`, `np.arcsin()`, `np.arccos()`, `np.arctan()`
- **Rounding**: `np.round()`, `np.floor()`, `np.ceil()`, `np.trunc()`
- **Exponents and Logarithms**: `np.exp()`, `np.log()`, `np.log10()`
- **Other Elementary Functions**: `np.sqrt()`, `np.cbrt()`, `np.square()`
- **Absolute and Sign Functions**: `np.abs()`, `np.sign()`
- **Advanced Array Operations**: `np.dot()`, `np.inner()`, `np.outer()`

### Example: Basic Math Operations

Here is the Python code:

```python
import numpy as np

# Generating the arrays
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

# Element-wise addition
print(np.add(arr1, arr2))  # Output: [ 6  8 10 12]

# Element-wise subtraction
print(np.subtract(arr1, arr2))  # Output: [-4 -4 -4 -4]

# Element-wise multiplication
print(np.multiply(arr1, arr2))  # Output: [ 5 12 21 32]

# Element-wise division
print(np.divide(arr2, arr1))  # Output: [5.         3.         2.33333333 2.        ]

# Element-wise power
print(np.power(arr1, 2))  # Output: [ 1  4  9 16]

# Element-wise modulo
print(np.mod(arr2, arr1))  # Output: [0 0 1 0]
```
<br>

## 10. What are _universal functions_ (_ufuncs_) in _NumPy_?

In **NumPy**, a **Universal Function** (ufunc) is a function that operates element-wise on **ndarrays**, optimizing performance.

Whether it's a basic arithmetic operation, advanced math function, or a comparison, ufuncs are designed to process data fast.

### Key Features

- **Element-Wise Operation**: Ufuncs process each element in an ndarray individually. This technique reduces the need for explicit loops in Python, leading to enhanced efficiency.

- **Broadcasting**: Ufuncs integrate seamlessly with **NumPy's broadcasting rules**, making them versatile.

- **Code Optimization**: These functions utilize low-level array-oriented operations for optimized execution.

- **Type Conversion**: You can specify the data type for output ndarray, or let NumPy determine the optimal type automatically for you.

- **Multi-Threaded Execution**: Ufuncs are highly compatible with multi-threading to expedite computation.

### Ufunc Categories

1. **Unary Ufuncs**: Operate on a single ndarray.
   
   Example: $\exp(5)$

2. **Binary Ufuncs**: Perform operations between two distinct arrays.

   Example: $10 + \cos(\text{{arr1}})$

### Code Example: Unique Advantages of Using Ufuncs

- Ufuncs Empower Faster Computing:
  - Regex and String Operations: Ufuncs are quicker and more efficient compared to list comprehension and string methods.
  - Set Operations: Ufuncs enable rapid union, intersection, and set difference with ndarrays.

- Enhanced NumPy Functions:
  - Log and Exponential Functions: NumPy provides faster and more accurate methods than standard Python math functions.
  - Trigonometric Functions: Ufuncs are vectorized, offering faster calculations for arrays of angles.
  - Special Functions: NumPy features an array of special mathematical functions, including Bessel functions and gamma functions, optimized for array computations.

```python
import numpy as np

arr = np.array([1, 2, 3])

# Using ".prod()" reduces redundancy and accelerates functional operation.
result = arr.prod()
print(result)

# Accessing unique elements via ufunc "np.unique" is more streamlined and quicker.
unique_elements = np.unique(arr)
print(unique_elements)
```
<br>

## 11. How do you perform _matrix multiplication_ using _NumPy_?

### Problem Statement

The task is to explain how to perform **matrix multiplication** using **NumPy**.

### Solution

NumPy's `np.dot()` function or the `@` operator is used for both **matrix multiplication** and **dot product**.

#### Matrix Multiplication

Two matrices are multiplied using the `np.dot()` function.

- $C = A \times B$ where $A$ is a $2 \times 3$ matrix and $B$ is a $3 \times 2$ matrix.

$$
C = \begin{bmatrix} A_{11} & A_{12} & A_{13} \\ A_{21} & A_{22} & A_{23} \end{bmatrix} \times \begin{bmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \\ B_{31} & B_{32} \end{bmatrix}
$$

$$
C = \begin{bmatrix} A_{11} \times B_{11} + A_{12} \times B_{21} + A_{13} \times B_{31} & A_{11} \times B_{12} + A_{12} \times B_{22} + A_{13} \times B_{32} \\ A_{21} \times B_{11} + A_{22} \times B_{21} + A_{23} \times B_{31} & A_{21} \times B_{12} + A_{22} \times B_{22} + A_{23} \times B_{32} \end{bmatrix}
$$

#### Broadcasting in NumPy

NumPy has a built-in capability, known as **broadcasting**, for performing operations on arrays of different shapes. If the shapes of two arrays are not compatible for an element-wise operation, NumPy uses broadcasting to make the shapes compatible.

#### Implementation

Here is the Python code using NumPy:

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])

# Matrix Multiplication
C = np.dot(A, B)
# The result is: [[ 58  64] [139 154]]
```
<br>

## 12. Explain how to _invert a matrix_ in _NumPy_.

### Problem Statement

The goal is to **invert a matrix** using NumPy.

### Solution

In NumPy, you can use the `numpy.linalg.inv` function to find the inverse of a matrix.

#### Conditions:

1. The matrix must be square, i.e., it should have an equal number of rows and columns.
2. The matrix should be non-singular (have a non-zero determinant).

#### Algorithm Steps:

1. Import NumPy: `import numpy as np`
2. Define the matrix: `A = np.array([[4, 7], [2, 6]])`
3. Compute the matrix inverse: `A_inv = np.linalg.inv(A)`

#### Implementation

Here's the complete Python code:

```python
import numpy as np

# Define the matrix
A = np.array([[4, 7], [2, 6]])

# Compute the matrix inverse
A_inv = np.linalg.inv(A)
print(A_inv)
```

The output for the given matrix `A` is:

```
[[ 0.6 -0.7]
 [-0.2  0.4]]
```
<br>

## 13. How do you calculate the _determinant_ of a _matrix_?

### Problem Statement

The **determinant** of a matrix is a scalar value that can be derived from the elements of a **square matrix**.

Calculating the determinant of a matrix is a fundamental operation in linear algebra, with applications in finding the **inverse of a matrix**, solving systems of linear equations, and more.

### Solution

#### Method 1: Numerical Calculation

For a numeric $n \times n$ matrix, the determinant is calculated using **Laplace's expansion** along rows or columns. This method is computationally expensive, with a time complexity of $O(n!)$.

#### Method 2: Matrix Decomposition

An alternative, more efficient approach involves using **matrix decomposition** methods such as **LU decomposition** or **Cholesky decomposition**. However, these methods are more complex and are not commonly used for determinant calculation alone.

#### Method 3: NumPy Function

The most convenient and efficient method, especially for large matrices, is to make use of the `numpy.linalg.det` function, which internally utilizes LU decomposition.

#### Implementation

Here is Python code:

```python
import numpy as np

# Define the matrix
A = np.array([[1, 2], [3, 4]])

# Calculate the determinant
det_A = np.linalg.det(A)
print("Determinant of A:", det_A)
```

#### Output

```
Determinant of A: -2.0
```

### Key Insight

The determinant of a matrix is crucial in various areas of mathematics and engineering, including linear transformations, volume scaling factors, and the characteristic polynomial of a matrix, often used in Eigenvalues and Eigenvectors calculations.
<br>

## 14. What is the use of the `_axis_` parameter in _NumPy functions_?

The `_axis_` parameter in **NumPy** enables operations to be carried out along a specific axis of a multi-dimensional array, providing more granular control over results.

### Functions with `_axis_` Parameter

Many NumPy functions incorporate the `_axis_` parameter to modify behavior based on the specified axis value.

### Common Functions

- **Math Operations**: Functions such as `mean`, `sum`, `std`, and `min` perform element-wise operations or aggregations, allowing you to focus on specific axes.

- **Array Manipulation**: `concatenate`, `split`, and others enable flexible array operations while considering the specified axis.

- **Numerical Analysis**: Functions like `trapezoid` and `Simpsons` provide integration along a specific axis, especially useful for multi-dimensional datasets.
  
### Practical Examples

#### Mean Calculation

Suppose you have the following dataset representing quiz scores:

```python
import numpy as np

# Quiz scores for five students across four quizzes
scores = np.array([[8, 6, 7, 9],
                   [4, 7, 6, 8],
                   [3, 5, 9, 2],
                   [4, 6, 2, 8],
                   [5, 2, 7, 9]])
```

You can calculate the mean scores for each quiz with:

```python
# axis=0 calculates the mean along the first dimension (students)
quiz_means = np.mean(scores, axis=0)
```

#### Splitting Arrays

Consider you want to separate a dataset into two based on a specific criterion. You can do this using `split`:

```python
# Assign students into two groups based on the mean quiz score
group1, group2 = np.split(scores, [2], axis=1)
```

In this case, it splits the `scores` array into two arrays at column index 2, resulting in `group1` containing scores from the first two quizzes and `group2` from the last two quizzes.

#### Integration over Multi-dimensional Arrays

NumPy provides functions to integrate arrays along different axes. For example, using the `trapz` function can calculate the area under the curve represented by the array:

```python
# Define a 2D array representing a surface
surface = np.array([[1, 2, 3, 4],
                    [2, 3, 4, 5]])

# Perform integration along axis 0
area_under_curve = np.trapz(surface, axis=0)
```
<br>

## 15. How do you _concatenate_ two _arrays_ in _NumPy_?

### Problem Statement

The task is to combine two $\text{NumPy}$ arrays.  Concatenation can occur **horizontally** (column-wise) or **vertically** (row-wise).

### Solution

In `NumPy`, we can concatenate arrays using the `numpy.concatenate()`, `numpy.hstack()`, or `numpy.vstack()` functions.

#### Key Points

- `numpy.concatenate()`: Combines arrays along a specified **axis**.
- `numpy.hstack()`: Stacks arrays horizontally.
- `numpy.vstack()`: Stacks arrays vertically.

Let's explore these methods in more detail.

#### Implementation

Here is the Python code:

```python
import numpy as np

# Sample arrays
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Concatenation along rows (vertically)
print(np.concatenate((arr1, arr2), axis=0))  # Output: [[1 2] [3 4] [5 6] [7 8]]

# Concatenation along columns (horizontally)
print(np.concatenate((arr1, arr2), axis=1))  # Output: [[1 2 5 6] [3 4 7 8]]

# Stacking horizontally
print(np.hstack((arr1, arr2)))  # Output: [[1 2 5 6] [3 4 7 8]]

# Stacking vertically
print(np.vstack((arr1, arr2)))  # Output: [[1 2] [3 4] [5 6] [7 8]]
```
<br>



#### Explore all 70 answers here 👉 [Devinterview.io - NumPy](https://devinterview.io/questions/machine-learning-and-data-science/numpy-interview-questions)

<br>

<a href="https://devinterview.io/questions/machine-learning-and-data-science/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fmachine-learning-and-data-science-github-img.jpg?alt=media&token=c511359d-cb91-4157-9465-a8e75a0242fe" alt="machine-learning-and-data-science" width="100%">
</a>
</p>

