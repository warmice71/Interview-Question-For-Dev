# 28 Must-Know Trie Data Structure Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

#### You can also find all 28 answers here 👉 [Devinterview.io - Trie Data Structure](https://devinterview.io/questions/data-structures-and-algorithms/trie-data-structure-interview-questions)

<br>

## 1. What is a _Trie_?

**Trie**, often called a **prefix tree** and named after the term **reTRIEval**, is a tree-like data structure tailored for string operations. Instead of housing data within its nodes, a Trie utilizes its edges to encode information.

### Key Features

- **Fast String Operations**: Tries streamline operations such as search, insert, and delete for strings, offering an edge over arrays and hash tables in specific scenarios.
- **Efficient Prefix Searches**: Their hierarchical structure facilitates easy traversal, making operations like autocomplete and prefix-based searches efficient.
- **Memory Compactness**: While they can be more space-intensive than hash tables for smaller dictionaries, Tries can store large dictionaries compactly, suiting tasks like natural language processing, spell-checking, and IP routing.

### Core Components

#### Trie Nodes

The **root node** doesn't carry a character value but acts as the starting point. Each subsequent **child node** corresponds to a distinct character. Some nodes might denote the end of a string while simultaneously representing characters.

#### Node Pointers

Nodes in the Trie hold references to child nodes. For instance, when dealing with the English alphabet, an array of 26 elements can map characters to respective child nodes.

#### Visual Representation

![Trie Data Structure](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/trie%2F1200px-Patricia_trie.svg%20(1).png?alt=media&token=b216a7e0-1ec9-4a12-b394-80ae45ffaf5e&_gl=1*1h44uz2*_ga*OTYzMjY5NTkwLjE2ODg4NDM4Njg.*_ga_CW55HF8NVT*MTY5NzM2MjgxMi4xNTguMS4xNjk3MzYzMjgyLjEuMC4w)

### Complexity Analysis

- **Time Complexity**:
  - Insertion, Search, and Deletion: $O(m)$ on average, where $m$ is the length of the word.
  - Prefix Search: $O(p)$, where $p$ is the number of words sharing the prefix.

- **Space Complexity**: Roughly $O(n \cdot \sigma)$, where $n$ is the total word count and $\sigma$ represents the size of the alphabet.

### Code Example: Trie

Here is the Python code:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word_end = True
        
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word_end

# Usage example
trie = Trie()
for word in ["a", "to", "tea", "ted", "ten", "in", "inn"]:
    trie.insert(word)

print(trie.search("tea"))  # True
print(trie.search("tedd")) # False
```
<br>

## 2. What are the advantages of _Tries_ over _Hash Tables_?

In certain scenarios, **Tries** can outperform **Hash Tables** due to their unique characteristics.

### Benefits of Using Tries Over Hash Tables

- **Ordered Iteration**: Unlike hash tables, tries inherently maintain key order. This makes them suitable for applications, such as dictionaries, where sorted output is needed. For instance, in Python, you can obtain sorted words from a trie-based dictionary using depth-first traversal.

- **Longest-Prefix Matching**: Tries stand out when identifying the longest common prefix, a feature indispensable for tasks like text auto-completion or network routing. As an example, search engines can expedite query suggestions based on the longest matching prefix.

- **Consistent Insertion Speed**: Tries offer a stable average-case insertion performance. This consistent behavior is attractive for latency-sensitive tasks. Contrarily, hash tables might necessitate occasional, time-intensive resizing operations.

- **Superior Performance with Small Keys**: For small keys, such as integers or pointers, tries can be more efficient than hash tables. Their simpler tree structures remove the overhead related to intricate hash functions in hash tables.

### Complexity Comparison

#### Time Complexity

- **Hash Tables**: On average, they have $O(1)$ lookup time. However, this can deteriorate to $O(n)$ in worst-case scenarios.
- **Tries**: They consistently exhibit an $O(k)$ lookup time, where $k$ represents the string's length. This predictability can be a boon for latency-critical tasks.

#### Space Complexity

- **Hash Tables**: Typically occupy $O(N)$ space for $N$ elements.
- **Tries**: Their space requirement stands at $O(ALPHABET\_SIZE \times N)$, governed by the number of keys and the alphabet's size.
<br>

## 3. What are _Advantages_ and _Disadvantages_ of a _Trie_?

The **trie** is a tree-like data structure optimized for string operations. While it presents distinct advantages, it also comes with specific drawbacks.

### Advantages of the Trie

1. **Efficient String Operations**: The trie achieves $O(m)$ time complexity for standard operations like insert, search, and delete, outpacing many traditional data structures.

2. **Prefix Matching**: Designed for rapid prefix matching, the trie is especially effective in auto-completion and spell-check scenarios.

3. **Memory-Saving for Common Prefixes**: The trie stores shared prefixes only once, which is advantageous for datasets with repeated string beginnings.

4. **Effective with Repetitive Data**: It's ideal for identifying duplicates or repetitions in datasets.

5. **Lexicographical Sorting**: The trie structure allows for easy iteration over data in lexicographic order.

6. **Flexible Alphabet Support**: The trie isn't limited to alphanumeric data and can handle any consistent set of symbols, such as DNA sequences.

### Disadvantages of the Trie

1. **Higher Memory Usage**: Compared to some alternatives, the trie can consume more memory, especially for large datasets, due to its node structure.

2. **Insertion Speed**: Inserting characters individually can make the insertion process slower than bulk data insertion methods.

3. **Cache Inefficiency**: Non-contiguous memory storage of trie nodes might lead to more frequent cache misses.

4. **Implementation Complexity**: Its recursive nature and pointer-based setup can make the trie more intricate to implement than simpler structures.
<br>

## 4. How do you handle the case sensitivity problem in _Tries_?

**Tries** are commonly used in natural language processing to store and look up words efficiently. They are **well-suited** for English, which is a **case-sensitive** language.

### Case Sensitivity and Tries

- Tries can treat characters in several ways:
  - **Case-sensitive**: Distinguishes between upper and lower case.
  - **Case-insensitive**: Ignores case distinctions.
  - **Folded**: Maps all characters to a specific case.

- For case-sensitive operation, use tree branches to represent different cases. For each node, there'd be up to 26 children: 13 for lowercase letters and 13 for uppercase letters.
  
- Modern keyboards can differ in the number of keystrokes required for common operations on case-insensitive tries. This influences the popularity of these systems.

- The most widely used trie variant is not case-sensitive because it requires fewer memory and computational resources.

### Handling Case Sensitivity 

In the **alphabet**:
- **Num of Unique Characters**: $26$ (English Standard 26, French 23, Turkish 29, etc.)
  
In a trie's **node structures**:
- **Memory Requirement**: One Boolean flag plus 26 pointers, demanding 26 times more memory than the simplest case-less variant.

### Code Example: Case-Insensitive Trie with Python

Here is the Python code:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class CaseInsensitiveTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        word = word.lower()
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_word = True

    def search(self, word):
        current = self.root
        word = word.lower()
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_word
```
<br>

## 5. What are some _Practical Applications_ of a _Trie_?

The **Trie** data structure is effective for tasks related to **exact string matching** and **text auto-completion**. Let's look at it's real-world applications.

### Practical Applications

#### Text Input
- **Auto-Completion**: Predict words as a user types, seen in search bars or messaging apps.
- **Spell Checking**: Quickly identify misspelled words.
- **Recommendation Engines**: Offer suggestions based on user input or previous choices.

#### Data Retrieval
- **String Filtering**: Efficiently find exact matches, useful for profanity filtering or blacklists.
- **Contact Lookups**: Search contact lists using names or numbers faster.

#### Data Analysis
- **Web Crawlers**: Assist in web indexing and data extraction.
- **Pattern Matching**: Used in algorithms like Aho-Corasick and Rabin-Karp for matching patterns, including in DNA sequences.

### Code Example: Trie-Based Auto-Completion

Here is the Python code:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word_end = False

class AutoComplete:
    def __init__(self):
        self.root = TrieNode()

    def insert_word(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word_end = True

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._get_words_from_node(node, prefix)

    def _get_words_from_node(self, node, prefix):
        results = []
        if node.is_word_end:
            results.append(prefix)
        for char, child_node in node.children.items():
            results.extend(self._get_words_from_node(child_node, prefix + char))
        return results

# Sample Usage:
ac = AutoComplete()
ac.insert_word("apple")
ac.insert_word("app")
ac.insert_word("application")
print(ac.search_prefix("ap"))  # ['app', 'apple', 'application']
```
<br>

## 6. Describe how a _Trie_ can be used for implementing autocomplete functionality.

**Trie**, a tree-like data structure, is an optimal choice for building **autocomplete systems**. With each node representing a single character, **Tries** are especially efficient in dealing with text data.

### Characteristics

- **Partial Matching**: The search process returns results that start with the input characters.
- **Dynamic**: The trie adapts as text changes.

### Algorithm Steps for Autocomplete

1. **Trie Traversal**: Starting from the root, navigate the trie to the node representing the last character in the input string.
2. **Subtree Collection**: Using the node from step 1, collect all descendant nodes to achieve partial matching.
3. **List Generation**: Convert the nodes from step 2 to strings, typically utilizing depth-first search.

### Advantages

- **Speed**: Autocompletion, even in large datasets, is quick.
- **Context Sensitivity**: Users see suggestions that match the context of words they've already typed.
- **Duplication Handling**: No duplicate suggestions are provided.

### Code Example: Trie Structure

Here is the Python code:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    # Other utility methods for trie operations such as delete.

# Example Usage
t = Trie()
words = ["hello", "world", "hi", "python"]
for word in words:
    t.insert(word)
```

### Code Example: Autocomplete Functionality Using Trie

Here is the Python code:

```python
class TrieAutoComplete(Trie):
    def get_all(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        
        return self._gather_words(node, prefix)

    def _gather_words(self, node, prefix):
        results = []
        if node.is_word:
            results.append(prefix)
        for char, child_node in node.children.items():
            results.extend(self._gather_words(child_node, prefix + char))
        return results

# Example Usage for Autocompletion
t = TrieAutoComplete()
words = ["hello", "world", "hi", "python"]
for word in words:
    t.insert(word)

print(t.get_all("h"))  # Output: ["hello", "hi"]

```

In the given example, the `get_all` method uses the prefix to reach the relevant node in the Trie and then uses the private method `_gather_words` to collect all words that start with the given prefix through a depth-first search.
<br>

## 7. Explain the use of a _Trie_ in IP Routing to match IP prefixes.

**Tries** are advanced data structures, specifically designed for efficient prefix matching. This quality makes them not only powerful but also indispensable in IP routing.

### IPv4 IP Address Representation

An IPv4 address like `192.168.1.1` is commonly represented in **binary** using 32 bits, partitioned into four octets. Decimal-to-binary conversion yields:

- 192: 11000000
- 168: 10101000
- 1: 00000001
- 1: 00000001

Concatenating these octets gives the binary form: 11000000101010000000000100000001.

### Trie Representation for IP Addresses

Tries are $N$-ary trees—each node can have up to $N$ children (commonly $N = 2$ for binary tries).

A **Binary Trie** is a specialized kind of trie where each node has either two or no children. This binary arrangement makes them ideal for matching binary IP addresses.

Starting from the root (or root node), you compare each bit of the IP address. A `0` bit usually directs the traversal to the left (child node), while a `1` bit leads to the right.

When you reach a leaf node, the series of `0`s and `1`s you encountered forms the longest matching prefix. Also, if a given node has no children, and it is a leaf node, the path to that node forms the **IP Prefix**.

### Binary Trie Example for IP Addresses

Here is the visual representation of how a binary trie looks with three IP addresses.

1. $1100 0000$ 0000 0000 0000 0000 0000 0001
2. $1100 0000$ 1010 1000 0000 0000 0000 0001
3. $1100 0000$ 10 000000 0000 0000 0000 0000

### Longest-Prefix Matching Technique

Longest-prefix matching involves identifying the most specific IP address that matches a given destination address. 

Practical uses of longest-prefix matching arise in various network tasks, such as:
- **IP address assignments** in Network Address Translation (NAT).
- **VPN routing tables** to find the most suitable forwarding entry in a routing table.

The trie's inherent design for binary strings and its navigational logic of reading incoming bits characterize the steps necessary to accomplish longest-prefix matching using a trie structure.

### Binary Trie for Longest-Prefix Matching

- **Initialization**: Start From Trienode `root` which contains all bits (32-bits long) of the keys. Both keys and Nodes (except leaves) can be thought of as 32-bit long strings, the root thus doesn't represent any key, just the unused bits.**.
  
- **Comparison**: Proceed through the trie by `bitwise AND` of the current node and the next bit in the given IP address. `1 -> Right`, `0 -> Left` are the conditions the comparison affects.
 
- **Longest Matching Prefix** is recorded as the valid IP prefix until the path reaches a leaf node.
<br>

## 8. How is a _Trie_ used in text _spell-checking_ and correction systems?

A **Trie** is a powerful **tree-like data structure**, especially suited for text operations such as **spelling correction**.

In a **spell-checking** context, the Trie helps to identify misspelled words by recognizing sequences of letters that do not form valid words.

### Trie for Spelling

The **Trie's** structure supports spelling-related tasks. Here's how:

- **Path Matching**: Each node represents a letter. Traversing from the root down a collection of nodes can form a word. This setup aids in comparing input tokens with stored words.
- **Comprehensive Coverage**: The Trie covers the entire pool of words in its structure. Therefore, it can verify any word against all known words, making it ideal for spell-checking.

### Spelling Auto-Correction

The Trie employs techniques such as **levenstein distance** and **edit distances**. These algorithms examine the relationship between words determined by word case, word length, and term frequency (TF-IDF).

### Edit Distances and Typos

Edit distances, also known as **Levenshtein distances**, represent the number of single-edit steps required to convert one given word into another. Such edits can encompass:

1. **Substitute**: Swapping one letter for another.
2. **Insert**: Placing a new letter within the word.
3. **Delete**: Removing a letter.
4. **Transpose**: Reversing the positions of two adjoining letters.

Implementing these transformations and utilizing the Trie for efficient validation helps correct apparent typos or isolated spelling errors.
<br>

## 9. Name some _Trie Implementation_ strategies.

To **implement a trie**, various strategies are available:

### Array-based Implementation

In this approach, each **trie node** contains an **array** where each index corresponds to a character of the alphabet.

For example, `0` stands for 'a', `1` for 'b', and so on. While simple to understand, this method can **consume more memory**, especially if many of the array elements are unused.

#### Code Example: Array Implementation

Here is the Python code:

```python
class TrieNode:
    def __init__(self):
        # Array for 26 letters of the alphabet
        self.children = [None] * 26
        self.isEndOfWord = False

# Initialize the root node
root = TrieNode()
```

### Linked List Approach

Here, each **node** corresponds to a **character**, and the `child` attribute points to the **head** of another linked list, representing the next level of characters.

This approach can **save memory** when the number of child nodes is significantly less than the array size of the previous method.

#### Code Example: Linked List Implementation

Here is the Python code:

```python
class TrieNode:
    def __init__(self, key=None, value=None, parent=None, prev=None):
        self.key = key
        self.value = value
        self.parent, self.prev, self.next = parent, prev, None
        self.child = None

# Initialize the root node
root = TrieNode()
```

### Bitmap Strategy

This technique uses a combination of a **bitmap** and an **array of pointers**. The bitmap determines which characters are present at a given node. The array of pointers then has a size equal to the number of set bits in the bitmap.

While this strategy can significantly **reduce the memory footprint**, it can add complexity to some operations.

### Hybrid Techniques

Merging the strengths of the strategies mentioned above can yield a balanced solution. For instance, one could use a **bitmap** to compress the children array in the **array-based approach** or combine bitmaps with linked lists.

### Best Practices for Modern Applications

- **Leverage Libraries**: There are well-optimized, library-based trie structures that can be advantageous. Using established libraries can streamline the process and enhance efficiency.
- **Optimize Based on Use Case**: The ideal trie configuration will vary based on its intended application.
<br>

## 10. Write the code to insert a word into a _Trie_.

### Problem Statement

The task is to create the code for **inserting** a word into a **Trie data structure**.

### Solution

In a **Trie**, each node represents a single character of a word. Starting from the root, a path along the nodes denotes a valid word.

#### Algorithm Steps

1. Initialize a **current node** to the root of the Trie.
2. For each character in the word, check if the current node has a child node corresponding to that character. If not, create a new node and add it as a child of the current node.
3. Move the current node to the child node corresponding to the character and repeat step 2.
4. After iterating through the word, mark the current node (which now represents the last character of the word) as an **end of the word**.

#### Complexity Analysis

- **Time Complexity**: $O(L)$ where $L$ is the length of the word.
- **Space Complexity**: $O(L)$ since in the worst case, we need to add $L$ new nodes, each for a different character in the word.

#### Implementation

Here is the Python code:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True

# Usage
trie = Trie()
words = ["apple", "banana", "app", "ape", "bat"]
for word in words:
    trie.insert(word)
```
<br>

## 11. Implement a _Search_ function to determine if a word is included in a _Trie_.

### Problem Statement

The goal is to implement a **search function** that can determine whether a particular word is included in a **Trie** data structure.

### Solution

The decision to include a word in a **Trie** structure relies on whether it exists in the dictionary represented by the Trie. In a Trie, the presence of a word is directly associated with the presence of an endmarker (\$).

- **For Python and JavaScript**: the `search` function recursively traverses the Trie. 
- **For C++**: an iterative approach is often preferred for performance reasons, especially when dealing with very large Tries.

Both methods should identify whether a given word is part of the Trie, hence serving as an efficient spellchecker or dictionary for a set of words.

#### Complexity Analysis

- **Time Complexity**: 
  - $O(m)$ where $m$ is the length of the key. Each character in the word can be found in its corresponding Trie level, which makes this operation linear.
  
- **Space Complexity**: 
  - For Python and JavaScript, the Space Complexity is $O(m)$ as well, where $m$ is the length of the key, accounting for the call stack during the recursive traversal.
  - For C++, the Space Complexity can be considered $O(1)$ as there are no additional data structures influencing it. However, if we include the space taken by critical operations during the search, it will still be $O(m)$ for a particular word.

#### Implementation

Here is the Python code:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_word = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_word
```
<br>

## 12. Write a script to _Delete_ a word from a _Trie_.

### Problem Statement

Deleting a word in a **Trie** usually involves setting the `is_end_of_word` attribute of the last node of the word to `False`. This is necessary, even though the word may not be explicitly present. Consider, for instance, the words "a" and "and" — if "and" is deleted, "a" should remain a word.

### Solution

In addition to setting `is_end_of_word` to `False` in the last node of the word, if any nodes become **redundant** (i.e., have no other children or are part of another word), they should be removed from the tree as well.

#### Algorithm Steps

1. Start from the root node.
2. For each character $c$ in the word:
   - If the child node for $c$ doesn't exist, the word isn't present in the trie. Return without making any changes.
   - Navigate to the child node for $c$.
3. After reaching the last character of the word, which ends at node $node\_end$:
   - If $node\_end$ has no children, traverse back (from the end) and delete nodes with only one child, until the root or a node with multiple children is reached.
   - Set $node\_end.is\_end\_of\_word$ to `False`.

#### Complexity Analysis

- **Time Complexity**: $O(m)$ where $m$ is the length of the word. We may have to traverse back up the tree to delete nodes, but the maximum number of backtracks is limited by the number of nodes in the trie.
- **Space Complexity**: $O(1)$

#### Implementation

Here is the Python code:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def delete(self, word):
        if not self.root.children:
            return  # Trie is empty

        self._delete_helper(self.root, word, 0)

    def _delete_helper(self, node, word, index):
        if index == len(word):
            if node.is_end_of_word:
                node.is_end_of_word = False
                if not node.children:
                    del node  # Delete redundant nodes
                return not node.children  # True if node has no children

            return False

        char = word[index]
        if char not in node.children:
            return False

        should_delete_current_node = self._delete_helper(node.children[char], word, index + 1)

        if should_delete_current_node:
            del node.children[char]
            return not node.children  # True if node has no other children after deletion

        return False

# Example
trie = Trie()
words = ["hello", "world", "python", "programming"]
for word in words:
    trie.insert(word)

trie.delete("python")
print(trie.search("python"))  # Output: False
```
<br>

## 13. Compare _Trie_ vs. _Binary Search Tree_.

While **Tries** are specialized for tasks involving strings and are especially efficient for datasets with shared prefixes, **BSTs** are versatile, general-purpose trees that can store any ordered data.

### Time Complexity

#### Look-up

- **Trie**: This is determined by the length of the word/key being looked up. Hence, the time complexity is $O(m)$, where $m$ is the length of the key.
- **BST**: Efficient look-ups in balanced BSTs are $O(\log n)$, but if the BST becomes skewed, it degrades to $O(n)$.

#### Insertion and Deletion

- **Trie**: Insertion and deletion are typically $O(m)$, with $m$ being the key's length.
- **BST**: Insertion and deletion are $O(\log n)$ in a balanced tree. However, in the worst-case scenario (unbalanced tree), these operations can take $O(n)$ time.

### Space Complexity

- **Trie**: Often more space-efficient, especially when dealing with datasets having short keys with common prefixes. It can save considerable space by sharing common prefix nodes.
- **BST**: Every node in the BST requires storage for its key and pointers to its two children. This fixed overhead can make BSTs less space-efficient than tries, especially for large datasets.

### Specialized Operations

- **Trie**: Excels at operations like longest-prefix matching, making it an ideal choice for applications such as autocompletion, IP routing, and more.
- **BST**: While not specialized like tries, BSTs are more general-purpose and can handle a wider range of tasks.

### Maintenance and Balance

- **Trie**: Inherently balanced, making them relatively low-maintenance. This ensures consistent performance without the need for additional balancing algorithms.
- **BST**: To maintain efficient operation, BSTs often require balancing using specific algorithms or tree structures like AVL or Red-Black trees. Without periodic balancing, the tree could become skewed, leading to suboptimal performance.

<br>

## 14. How to choose between a _Hash Table_ and a _Trie_?

Both **Hash Tables** and **Tries** are data structures used for efficient data storage and retrieval.

While **hash tables** offer fast lookups and unordered data storage, **tries** are optimized for ordered tasks and text-related functions.

### Hash Tables: Quick Lookups

- **Definition**: Uses a hash function to map keys to unique values for fast retrieval.
- **Best For**: Unordered data and quick lookups.
- **Pros**: Fast O(1) lookups, memory efficiency.
- **Cons**: Lack of ordering, potential for collisions.

### Tries: Text-Focused Storage

- **Definition**: Organizes keys based on common prefixes and is often used for text-related tasks like auto-completion.
- **Best For**: Maintaining order and tasks involving text.
- **Pros**: Efficient prefix matching, ordered retrieval.
- **Cons**: Memory-intensive.

### Code Example: Word Lookup

Here is the Python code:

```python
# Hash Table
word_dict = {"hello": "English greeting", "hola": "Spanish greeting"}

# Trie
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False

root = TrieNode()
words = ["hello", "hola"]
for word in words:
    node = root
    for char in word:
        node = node.children[char]
    node.is_word = True
```
<br>

## 15. What is a _Suffix Trie_, and how does it differ from a standard _Trie_?

A **Suffix Trie** extends the functionality of a regular trie to capture all **suffices** of a string. Both Tries are structures well-suited for substrings and searching in text.

### Core Approach

Rather than simply storing individual letters, a **Suffix Trie** captures complete substrings from the input word. Each node in the **Suffix Trie** represents a substring, starting from the root that denotes the complete word and ending with terminal nodes signifying various suffixes.

#### Example: "BANANA"

In a regular trie, word "BANANA" would be represented as follows:

```plaintext
       root
        |
        B
       / \
      A   R
     / \
    N   E
   /
  A
```

In a **Suffix Trie** for "BANANA," the main root still represents the complete word. However, it has specialized edges, known as suffix links or paths, leading to nodes that correspond to **suffixes** of the word:

```plaintext
       root
        /  \
      A..  ..NA
     /       \
    NA        B..A
   /             \
  ANA             NA
 /                 \
NA                 A
 \               /
  A             NA
```

### Structural Differences

- **Leaf Nodes**: In a regular trie, leaves often serve as indicators for complete words. In a **Suffix Trie**, leaf nodes invariably denote the end of a suffix.

- **Internal Nodes**: These nodes in **Suffix Tries** hold multiple incoming edges, including those from the root, whereas in regular tries, they typically have a one-to-one edge-to-substring relationship.

- **Terminating Characters**: Regular tries often rely on designated characters like "end of word" flag (e.g., `'$'`) to mark the end of words. **Suffix Tries**, in contrast, determine word boundaries based on incorporating all suffixes of the input.

### Primary Utility

- **Trie**: Optimized for rapid prefix-based search, common in tasks such as autocompletion.
- **Suffix Trie**: Primarily used to facilitate swift pattern matching in texts, particularly in bioinformatics and data compression.

### Code Example: Regular Trie and Suffix Trie

Here is the Python code:

```python
# Regular Trie
class Node:
    def __init__(self):
        self.children = {}
        self.is_end = False

def insert_word(root, word):
    node = root
    for char in word:
        if char not in node.children:
            node.children[char] = Node()
        node = node.children[char]
    node.is_end = True

# Suffix Trie
class SuffixNode:
    def __init__(self):
        self.children = {}
        self.suffix_link = None

def insert_suffix(root, suffix):
    node = root
    for char in suffix:
        if char not in node.children:
            node.children[char] = SuffixNode()
        node = node.children[char]
    # Establishing the suffix link is algorithmically more complex
```
<br>



#### Explore all 28 answers here 👉 [Devinterview.io - Trie Data Structure](https://devinterview.io/questions/data-structures-and-algorithms/trie-data-structure-interview-questions)

<br>

<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

