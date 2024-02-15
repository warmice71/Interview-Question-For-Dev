# 40 Important Bit Manipulation Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

#### You can also find all 40 answers here 👉 [Devinterview.io - Bit Manipulation](https://devinterview.io/questions/data-structures-and-algorithms/bit-manipulation-interview-questions)

<br>

## 1. What is a _Bit_?

The term **"bit"** stems from "binary" and "digit." As the basic unit of information in computing and digital communications, a bit can assume one of two values: 0 or 1.

### Binary System vs Decimal System

Computers operate using a **binary number system**, employing just two numerals: 0 and 1. In contrast, our day-to-day decimal system is **base-10**, utilizing ten numerals (0-9).

In the binary system:
- **Bit**: Represents 0 or 1
- **Nibble**: Comprises 4 bits, representing 16 values (0-15 in decimal)
- **Byte**: Contains 8 bits, representing 256 values (0-255 in decimal)

For instance, the decimal number 5 is depicted as $0101_2$ in binary.

### Bit Manipulation

Bits are pivotal in **bit manipulation**, a field encompassing operations like bit shifting, logical operations (AND, OR, XOR), and bit masking. These techniques find applications in data compression, encryption, and device control.

Considering two 8-bit numbers: $0010\,1010$ and $0000\,1100$. The logical AND operation gives:

$$
\begin{array}{c c c c c c c c c}
  & 0 & 0 & 1 & 0 & 1 & 0 & 1 & 0 \\
\text{AND} & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 \\
\hline
  & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
\end{array}
$$

### Integer Representation in Bits

An integer's representation typically occupies a **fixed** number of bits. On many systems, an integer uses **32 bits**. Thus, a 32-bit signed integer spans $-2^{31}$ to $2^{31} - 1$.

### Hardware Considerations

Although **bits** underpin computing, hardware designs can constrain their usage. A 32-bit CPU processes 32 bits simultaneously, requiring extra steps for larger numbers. This led to the adoption of "**double words**" and "**quad words**" to represent larger integers.
<br>

## 2. What is a _Byte_?

A **byte** is a foundational data unit in computing and telecommunications, capable of representing 256 unique values, ranging from 0 to 255. It consists of 8 **bits**, the smallest data storage units, which can be either 0 or 1.

### Bit Composition

Each bit in a byte has a place value, starting from the least significant bit (LSB) on the right to the most significant bit (MSB) on the left. Their place values are: 

| Place Value | Bit Position |
|-------------|--------------|
| 128         | 7            |
| 64          | 6            |
| 32          | 5            |
| 16          | 4            |
| 8           | 3            |
| 4           | 2            |
| 2           | 1            |
| 1           | 0            |


Setting all bits to 1 yields the byte's maximum value of 255.

### Converting Bytes to Decimal

To find the decimal equivalent of a byte, sum the place values of bits set to 1. For a byte with all bits as 1:

$$
1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255
$$

### Code Example: Byte to Decimal Conversion

Here is the Python code:

```python
def byte_to_decimal(byte_str):
    # Reverse the string for right-to-left calculation
    byte_str = byte_str[::-1]
    
    # Sum up place values for bits set to 1
    return sum(int(byte_str[i]) * 2 ** i for i in range(len(byte_str)))

# Example usage
byte_value = "11111111"  # All bits are 1
decimal_value = byte_to_decimal(byte_value)
print(decimal_value)  # Output: 255
```
<br>

## 3. Explain what is a _Bitwise Operation_.

**Bitwise operations** are actions applied to individual bits within binary numbers or data units like integers. These operations offer several advantages, including speed and memory efficiency, and are widely used in specific computing scenarios.

### Why Use Bitwise Operations?

- **Speed**: Executing bitwise operations is often faster than using standard arithmetic or logical operations.
  
- **Memory Efficiency**: Operating at the bit level allows the storage of multiple flags or small integers within a single data unit, optimizing memory usage.

- **Low-Level Programming**: These operations are crucial in embedded systems and microcontroller programming.

- **Data Manipulation**: Bitwise operations can selectively alter or extract specific bits from a data unit.

### Types of Operators

#### Logical Operators

1. **AND (`&`)**: Yields `1` if corresponding bits are both `1`; otherwise, `0`.
    - Example: `5 & 3 = 1`.
  
2. **OR (`|`)**: Yields `1` if one or both corresponding bits are `1`; otherwise, `0`.
    - Example: $5 | 3 = 7$.

3. **XOR (`^`)**: Yields `1` when corresponding bits differ; otherwise, `0`.
    - Example: $5 \oplus 3 = 6$.

4. **NOT (`~`)**: Inverts all bits.
    - Example: $~5$ becomes $-6$ in 2's complement.

#### Shift Operators

1. **Left Shift (`<<`)**: Moves bits to the left and fills in with `0`.
    - Example: $5 \text{ << 2 } = 20$.

2. **Right Shift (`>>`)**: Moves bits to the right and fills in based on the sign of the number.
    - Example: $5 \text{ >> 2 } = 1$.

3. **Zero Fill Right Shift (>>>)**: Shifts bits right, filling in zeros.
    - Example: $-5 \text{ >>> 2 } = 1073741823$.
    
#### Specialized Operations

- **Ones' Complement**: Similar to NOT but restricted to 32 bits.
    - Example: `(~5) & 0xFFFFFFFF`.

- **Twos' Complement**: Used for representing negative numbers in two's complement arithmetic.
    - Example: $~5 + 1 = -6$.

### Practical Applications

- **Flag Management**: Bits can represent on/off flags, and bitwise operators can be used to toggle these.

- **Data Compression**: These operations play a role in compression algorithms.
  
- **Encryption**: Bitwise manipulation is used in cryptographic algorithms.

### Code Example: Flag Manipulation

Here is the Python code:

```python
# Define flags with binary representation
FLAG_A, FLAG_B, FLAG_C, FLAG_D = 0b0001, 0b0010, 0b0100, 0b1000

# Set flags B and D
flags = FLAG_B | FLAG_D

# Check if Flag C is set
print("Flag C is set" if flags & FLAG_C else "Flag C is not set")
```
<br>

## 4. What are some real-world applications of _Bitwise Operators_?

**Bitwise operators** provide efficient means of manipulating variables at the bit level. This feature is integral to various applications like data compression, cryptography, and embedded systems.

### Real-World Use-Cases

#### Data Compression Algorithms
1. **Run-Length Encoding**: Identical consecutive characters are stored once followed by a count. This requires bitwise operators to efficiently manage the bit stream.
2. **Huffman Coding**: For implementing lossless data compression, this technique assigns shorter codes to frequently occurring characters, which is made possible through bitwise operations.

#### Cryptography and Data Security
1. **Bit Level Encryption**: Techniques like XORing bits are used in various encryption algorithms.
2. **Hardware Security**: In integrated chips, bitwise operations play a crucial role in providing secure key management systems.

#### Network Packet Analysis
1. **Packet Inspection**: Applications, especially in firewalls and routers, might necessitate bitwise operations for quick and low-level packet analysis.

#### Embedded Systems
1. **Peripheral Configuration**: The individual bits in control registers, often set using bitwise operations, help configure peripherals in microcontrollers and other embedded systems.
2. **Memory Mapped I/O**: Bitwise operations are instrumental in interfacing with embedded hardware through memory-mapped I/O.

#### Algorithm Optimization
1. **Bit Manipulation for Speed**: In specific situations, using bit-level operations can significantly enhance the efficiency of algorithms. This is especially true for resource-constrained devices.
2. **Integer Multiplication and Division in Limited-Bit Environments**: On systems with limitations on the size of numbers that can be represented, bit manipulation can be used to carry out basic arithmetic operations more efficiently.

#### Graphics and Image Processing
1. **Pixel Manipulation**: Adjusting color information or applying specific transformations may involve bitwise operations in image processing.
2. **Optimized Blending**: Quick and optimized alpha blending, common in graphic rendering, can be achieved using bitwise operations without the need for costly division and multiplication.

#### Data Integrity and Validation
1. **Flags in Data Structures**: Bitwise operations enable data integrity checks and the management of multiple flags within data structures while using limited memory.
2. **Parity Checks**: Detection of odd/even parity in small data segments, commonly in error-checking algorithms, employs bitwise methods.

#### Memory Optimization and Cache Management
1. **Memory Allocation**: In scenarios where individual bits are used for encoding specific information within memory allocation strategies, bitwise operations are fundamental.
2. **Cache Optimization**: Techniques like bit masking can be used to optimize cache performance by ensuring data alignment with cache lines.

#### User Interface and Input Management
1. **Keyboard Input Handling**: In certain contexts, handling multiple keyboard inputs or mapping specific keys can be simplified using bit manipulation.
2. **Graphics Display**: To save resources while still effectively managing color palettes in limited environments, bit manipulation is employed.

#### Dynamic Resource Management
1. **Memory and Resource Allocation**: In operating systems and embedded systems, bitwise operations provide a means of managing the allocation and deallocation of resources with precision and efficiency.

#### General Efficiency and Resource Utilization
1. **Memory Efficiency**: Bit fields in languages like C and C++ make efficient use of memory by grouping variables into compact memory units.
2. **Performance Enhancement in Math Operations**: Bit manipulation can be used for efficient multiplication, division, and modulo operations on binary integers.
3. **Finding Mismatches and Duplicates**: Bitwise Exclusive OR (XOR) operations ascertain duplicates or mismatches in data sets.

### Code Example: Run-Length Encoding

Here is the Python code:

```python
def run_length_encode(data):
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded.append((data[i - 1], count))
            count = 1
    encoded.append((data[-1], count))
    return encoded

def run_length_decode(encoded):
    decoded = ""
    for char, count in encoded:
        decoded += char * count
    return decoded

# Test the Run-Length Encoding and Decoding
input_data = "AAABBCCCCDDEEEE"
encoded_data = run_length_encode(input_data)
decoded_data = run_length_decode(encoded_data)
print("Original Data:", input_data)
print("Encoded Data:", encoded_data)
print("Decoded Data:", decoded_data)
```

In this example, the input string "AAABBCCCCDDEEEE" is run-length encoded and then decoded back to its original form using bit manipulation techniques.
<br>

## 5. What is a _bitwise AND_ operation and how can it be used to check if a number is _odd_ or _even_?

The **bitwise AND** operation is a fundamental concept in computer science and cryptography. When you apply this operation to two bits, the result is 1 if both bits are 1. At least one bit being 0 results in a zero output. This operation is often used in hashing and encryption algorithms.

In the context of determining whether a number is **odd** or **even**, the bitwise AND operation becomes useful.

### Bitwise AND to Check for Odd or Even

The basic way to figure out if a decimal number is even or odd, based on its binary representation, is to look at the least significant bit (the rightmost bit):

- If that bit is 1, the number is odd.
- If it's 0, the number is even.

The rule behind this method is that all even numbers end in `0` in binary, and odd numbers end in `1`.

### Logical Representation

- bitwise AND with 1: Returns 1 if rightmost bit is 1 (indicating odd number), and 0 if it's 0 (indicating even number).

#### Mathematical Foundation

When you do a **bitwise AND** with a number and 1, you get:

- 1 if both the numbers are 1.
- 0 if the other number is 0.
  
For an even number $n$, its **binary** form ends in `0`. When you take the **logical AND** with `1`,  you actually perform a **logical AND** with `0`, which results in `0`.

For an odd number $n$, its **binary** form ends in `1`. When you take the **logical AND** with `1`, the operation returns `1`.

### Example

- For n = 5 (binary: 101):

  `5 & 1` gives 1, indicating `n` is odd.

- For n = 10 (binary: 1010):

  `10 & 1` gives 0, indicating `n` is even.

### Python Code

Here is the Python code for the operation:

```python
def is_odd(n):
    return n & 1 == 1
```

The function `is_odd` checks whether `n` is an odd number by using bitwise AND with 1. If the result is 1, the number is odd.
<br>

## 6. Explain the _bitwise OR_ operation with an example.

The **Bitwise OR** operator works at the binary level. It combines two bit sequences. For each position, if **either bit is 1**, the result has a 1 in that position.

### Key Insight

- **Input**: Two binary numbers, say `a = 1010` and `b = 1100`.
- **Output**: Their bitwise OR, denoted by `a | b`, gives `1110`.

### Code Example: Implementing Bitwise OR

Here is the Python code:

```python
a = 10   # Binary: 1010
b = 12   # Binary: 1100

result = a | b
print(bin(result))  # Output: 0b1110 (14 in decimal)
```
<br>

## 7. How does the _bitwise XOR_ operation work, and what is it commonly used for?

The **bitwise XOR operator** (\^\) compares each bit of two integers, setting the resulting bit to 1 only when the two input bits are different.

### XOR in Action

- **Example**: 5 ($101$) $\text{XOR}$ 3 ($011$) = 6 ($110$)
  
- **Properties**: Commutative: $A \text{ XOR } B = B \text{ XOR } A$

### Practical Applications

- **Invert Elements and Undo Pairing**: Useful for error checking and data identification.

- **Text/Data Encryption**: Employed in ciphers that use bit manipulation for security.

- **Efficiently Swapping Values**: Useful for in-place memory operations without using temporary storage.

- **Color Calculations and Image Processing**: Commonplace in graphics processing for tasks like filtering.

- **Error Correction in Data Transmission**: Ensures data integrity during communication.

- **Modifying Individual Bits in a Register**: Efficiently flip specific bits without affecting others.

### Code Example: Swapping Numbers

Here is the Python code:

```python
a, b = 5, 7
a = a ^ b
b = a ^ b
a = a ^ b
print(a, b)  # Output: 7, 5
```
<br>

## 8. Demonstrate how to set, toggle, and clear a _specific bit_ in a number using _bitwise operators_.

Let's first discuss the different bitwise operations:

- **Setting a Bit** involves turning the bit on (to 1), if it's not already.
  
- **Toggling a Bit** changes the bit's state: 1 becomes 0, and vice versa.
  
- **Clearing a Bit**, on the other hand, turns the bit off (to 0).

And here is the table of operations:

| Bit State | Operations  | Result |
|-----------|-------------|--------|
| 0         | Set         | 1      |
| 0         | Toggle      | 1      |
| 0         | Clear       | 0      |
| 1         | Set         | 1      |
| 1         | Toggle      | 0      |
| 1         | Clear       | 0      |



### Bit Manipulation Operations

Here is the code:

In C++:

```cpp
#include <iostream>

// Set the I-th bit of N
int setBit(int N, int I) {
    return N | (1 << I);
}

// Toggle the I-th bit of N
int toggleBit(int N, int I) {
    return N ^ (1 << I);
}

// Clear the I-th bit of N
int clearBit(int N, int I) {
    return N & ~(1 << I);
}

int main() {
    int number = 10; // 0b1010 in binary
    int bitPosition = 1;
  
    // Set bit in position bitPosition
    int newNumber = setBit(number, bitPosition);  // Result: 14 (0b1110)
    std::cout << "Number after setting bit: " << newNumber << std::endl;

    // Toggle bit in position bitPosition
    newNumber = toggleBit(newNumber, bitPosition);  // Result: 10 (0b1010)
    std::cout << "Number after toggling bit: " << newNumber << std::endl;

    // Clear bit in position bitPosition
    newNumber = clearBit(newNumber, bitPosition);  // Result: 8 (0b1000)
    std::cout << "Number after clearing bit: " << newNumber << std::endl;

    return 0;
}
```

### Visual Illustration

Here are the steps visually:

#### Setting a Bit

```plaintext
Initial Number:      1010 (10 in decimal)
Bit Position:          1     
Shifted 1 to:        0010
Result (OR):         1110 (14 in decimal)
```

#### Toggling a Bit

```plaintext
Previous Result:     1110 (14 in decimal)
Bit Position:           1
Shifted 1 to:         0010
Result (XOR):        1010 (10 in decimal)
```

#### Clearing a Bit

```plaintext
Previous Result:   1010 (10 in decimal)
Bit Position:         1
Shifted Negation:  1101
Logical AND:       1000 (8 in decimal)
```
<br>

## 9. What is _bit masking_, and how would you create a _mask_ to isolate the _nth bit_?

**Bit masking** involves using **bitwise operations** to either clear or set specific bits in a binary number.

For instance, if you want to extract the 3rd bit of a number, you would use the **bit mask** `00001000` (in binary), which is decimal 8.

### Creating a Bit Mask to Isolate the $n^{th}$ Bit

To extract the $n^{th}$ bit from a number `num`, you can use a bit mask that has all 0s except for a 1 at the $n^{th}$ position.

You can generate this bit mask `mask` by left-shifting a 1 by $n-1$ positions. If $n=3$, for example, the resultant `mask` would be 4 in decimal or `00000100` in binary.

Here's an example, using $n=3$:

```python
def extract_nth_bit(num, n):
    mask = 1 << (n-1)
    return (num & mask) >> (n-1)
```

### Mask Action: Logical AND

To **extract** the bit, you perform a **logical AND** of the number with the mask. All bits in `mask` are zero, except for the $n^{th}$ bit, which preserves the corresponding bit in the original number. All other bits become zero.

### Mask Action: Bit Shift (Right)

After using the **logical AND**, the extracted bit is still in position 1 ($2^1$). By shifting it to the right one time, it will be in position 0 ($2^0$), i.e., as 0 or 1.

### Code Example: Extracting $3^{rd}$ Bit

```python
def extract_nth_bit(num, n):
    mask = 1 << (n-1)
    return (num & mask) >> (n-1)

# Using a number where the 3rd bit is 1
num = 13    # 1101 in binary
print(extract_nth_bit(num, 3))  # Output: 1
```
<br>

## 10. Explain how _left_ and _right shifts_ (_<<_ and _>>_) work in bit manipulation.

**Bit shifts**, controlled by the `<<` (left) and `>>` (right) operators, move bits in a binary number. Each shift direction and position has unique behavior.

### Direction vs. Operator

- **Direction**: Determines whether bits shift to the left or to the right.
- **Operator**: Symbolizes the actual shift in the code.


### Shift Direction and Unary Operators

- **Left Shift** (<<): Moves bits to the right, effectively multiplying the number by $2^n$.
- **Right Shift** (>>): Moves bits to the left and truncates the remainder, akin to integer division by  $2^n$ in most programming languages.

### Shift Operations on Binary Numbers

  Let's understand shift operations through examples:

  ```plaintext
  Original Number (in binary):  11001010
  ```

  #### Right Shift ($\text{>>}$)
  
  - 1-bit Right Shift
      ```plaintext
      1100101
      Binary: 1100101
      Decimal:  105
      ```

  - 2-bit Right Shift
      ```
      110010   
      Binary: 110010
      Decimal:  50 
      ```
  
  - 3-bit Right Shift
      ```
      11001  
      Binary: 11001
      Decimal:  25
      ```

  - 4-bit, full 8-bit, and 10-bit Right Shift
      All shift operations are readily achieved by further bit truncation. 

  **Note**: As your task requires involving multiplication and division, such shift operations lead to an understanding of these mathematical operations in binary number representation.

   -  Multiplication
      By performing a left shift, you are essentially multiplying the number by 2. 

      ```
      110010
      Binary: 1100100
      Decimal: 100
      ```

  -  Division
     Right shifts are akin to dividing a number by powers of 2. A 3-bit right shift divides the given number by  $2^3 = 8$ .

      ```plaintext
      13/2^3 = 13/8 = 1, remainder = 5
      Based on this example:
      11010
      Binary: 11010
      Decimal: 26
      ```

### Code Example: Right Shift to Divide by 2

Here is the Python code:

```python
number = 8
right_shifted = number >> 1  # This effectively divides by 2
print(right_shifted)  # Output will be 4
```
<br>

## 11. What is the difference between _>>_ and _>>>_ operators?

Let's look at two shift operators defined in Java: the **right shift (>>) operator** and the **unsigned right shift (>>>) operator**, and compare their functionality.

### Understanding the Operators

- **Right Shift (`>>`)**: Moves all bits of the specified numeric value to the right. It fills the leftmost bits with the sign bit (0 for positive, 1 for negative).

- **Unsigned Right Shift (`>>>`)**: Similar to the `>>` operator, but it always fills the left-moved positions with zeros, ignoring the sign bit.

### Visual Representation

```plaintext
      Decimal      Binary Bits
       10           00001010
       10 >> 1      00000101    - 5
       10 >>> 1     00000101    - 5
--------------------------
     -10           11110110
      -10 >> 1      11111011    - (-5)
      -10 >>> 1     01111011    - 251
```
<br>

## 12. Explain how to perform _left_ and _right bit rotations_.

**Bit rotation** refers to shifting the bits of a binary number to the left or right, and wrapping the bits around so that any that "fall off" one end are reintroduced at the other.

### Arithmetic Shift vs Logical Shift

In many programming languages, bit shifts are either arithmetic or logical.

- **Arithmetic shifts** are typically used for signed integers and preserve the sign bit, which means the bit shifted in from the most significant bit becomes the new least significant bit, and bits "shifted off" on the other end are discarded.
- **Logical shifts** shift all bits, including the sign bit, and always fill in the vacated bit positions with zeros.

#### Code Example: Logical and Arithmetic Shifts

Here is C++ code:

```cpp
#include <iostream>

int main() {
    // Explain logical shift
    int logicalShiftResult = -16 >> 3;
    
    // Explain arithmetic shift
    int arithmeticShiftResult = -16 << 3;
    
    return 0;
}
```
<br>

## 13. Write a function that counts the number of _set bits_ (1s) in an _integer_.

### Problem Statement

The task is to count the number of set bits in an integer - the **1**s in its binary representation.

### Solution

One approach is to check each bit and sum them. An optimized solution uses a technique called **Brian Kernighan's Algorithm**. 

It is based on the observation that for any number `x`, the value of `x & (x-1)` has the bits of the rightmost set `1` unset. Hence, repeating this operation until the number becomes `0` yields the set bit count.

#### Algorithm Steps

1. Initialize a count variable to `0`.
2. Iterate using a **while loop** until the number is `0`.
3. Within each iteration, decrement the number `n` by `n & (n-1)` and increment the `count` variable by `1`.

#### Complexity Analysis

- **Time Complexity**: $O(\text{{set bits count}})$, as it depends on the number of set bits.
- **Space Complexity**: $O(1)$

#### Implementation

Here is the Python code:

```python
def count_set_bits(n):
    count = 0
    while n:
        n &= (n-1)
        count += 1
    return count
```

Here is the C++ code:

```cpp
int countSetBits(int n) {
    int count = 0;
    while (n) {
        n &= (n - 1);
        count++;
    }
    return count;
}
```
<br>

## 14. Determine if a number is a _power of two_ using _bit manipulation_.

### Problem Statement

The task is to design an algorithm that determines whether a given number is a **power of two**.

### Solution

Using **bit manipulation**, we can apply the **logical AND** operator to swiftly identify powers of two.

- If a number is a power of two, it has exactly one bit set in its binary representation.
- Subtracting 1 from a power of two yields a binary number with all lower bits set.

Combining these properties, we obtain an expression that performs the essential check.

#### Algorithm Steps

1. Check if the number is **non-negative**.
2. Apply the **bitwise AND** operation between the number and its **one's complement** (the bitwise negation of the number).
3. Determine if the result is **zero**.

If the result is zero, we confirm the number is a power of two.

#### Complexity Analysis

- **Time Complexity**: $O(1)$
- **Space Complexity**: $O(1)$

#### Implementation

Here is the Python code:

```python
def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x-1)) == 0
```
<br>

## 15. Design a function that adds two numbers without using the '+' operator.

### Problem Statement

The task is to create a function that would add two numbers **without using the `+` operator**.

### Solution

There are several methods to add two numbers without using the `+` operator, each with its own trade-offs. One common approach is to use **bit manipulation**.

#### Algorithm Steps

1. **Perform XOR Operation**: 
   - Calculate the bitwise XOR of the two numbers. This produces a number where the set bits represent the positions at which the two numbers have different bits.

   ```
   0010 (2) 
   XOR 0100 (4) 
   ------------
   0110 (6)
   ```

2. **Perform AND Operation, then Left Shift**:
   - Perform bitwise AND of the two numbers and then left shift the result by `1`. This brings forward any 'carry' bits to the appropriate position.

   ```
   0010 (2) 
   AND 0100 (4)
   ------------
   0000 (0)
   ``` 

   After left shifting by 1:

   ```
   0000 (0)
   << 1
   ------------
   0000 (0)
   ```

3. **Recursion**: Apply the addition method to the new **XOR** result and the **AND-left-shifted** result. The recursion continues until there are no carry bits left.

   ```
   0110 (XOR output)
   0000 (Carry bits from AND-left-shifted operation)
   ```

   Next, we perform the addition method to `0110` and `0000`, which returns `0110`.

   The final result will be `0110` which equals 6, the sum of 2 and 4.

#### Complexity Analysis

- **Time Complexity**: $O(\log n)$ where $n$ is the larger of the two numbers.
- **Space Complexity**: $O(1)$

#### Implementation

Here is the Python code:

```python
def add_without_plus_operator(a, b):
    while b:
        # Calculate the carry bits
        carry = a & b

        # Use XOR to get the sum without carry
        a = a ^ b

        # Left shift the carry to add in the next iteration
        b = carry << 1

    return a

# Test the function
print(add_without_plus_operator(2, 4))  # Output: 6
```
<br>



#### Explore all 40 answers here 👉 [Devinterview.io - Bit Manipulation](https://devinterview.io/questions/data-structures-and-algorithms/bit-manipulation-interview-questions)

<br>

<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

