# 55 Core Cryptography Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/software-architecture-and-system-design/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fsoftware-architecture-and-system-design-github-img.jpg?alt=media&token=521fd2a9-0d56-49c0-a723-9bd6ca081893" alt="software-architecture-and-system-design" width="100%">
</a>
</p>

#### You can also find all 55 answers here 👉 [Devinterview.io - Cryptography](https://devinterview.io/questions/software-architecture-and-system-design/cryptography-interview-questions)

<br>

## 1. What is _cryptography_, and what are its main goals?

**Cryptography** is an interdisciplinary field, harnessing techniques from mathematics, computer science, and electrical engineering to securely transfer and store data. This is achieved by converting **plain-text** into encrypted **cipher-text**, which is then transformed back into its original form.

### Key Elements

#### Encryption / Decryption

- **Encryption**: The process of converting **plain-text** into an **unreadable** form, known as **cipher-text**. This is typically carried out with the help of mathematical algorithms and a **secret key**.
  
- **Decryption**: The reverse process, which turns **cipher-text** back into **plain-text**. This requires the secret key.

#### Key Management

For secure encryption and decryption, keys need to be managed. This includes key generation, distribution, and revocation.

- **Symmetric Encryption**: uses a **single key** for both encryption and decryption (e.g., AES, DES).
- **Asymmetric Encryption**: uses a **pair of keys**, one for encryption and the other for decryption (e.g., RSA, ECC).

#### Integrity

Cryptography ensures that data has not been tampered with during transmission through techniques like digital signatures and message digests.

#### Non-Repudiation

Cryptographic methods like digital signatures aim to prevent the sender from denying the authenticity of a message.

#### Authentification

Cryptography provides mechanisms to ensure that the sender and recipient of a message are as they claim to be.

### Best Practices

- Embrace a layered approach, combining **encryption**, **authentication**, and **integrity** assurances.
- Implement **robust key management**, such as key update schedules and secure storage.
- Regularly review and update cryptographic methods.
- Use trusted technologies and APIs that are well-tested and continually audited.

By adhering to these principles, cryptographic systems can achieve their goals, providing a foundation of trust in digital interactions.
<br>

## 2. Explain the difference between _symmetric_ and _asymmetric cryptography_.

**Symmetric and asymmetric cryptography** each play a distinct role in securing digital communication, distinguished by their key management, computational complexity, and suitability for various tasks.

### Key Differences

- **Symmetric Cryptography**: Uses a shared secret key both for encryption and decryption. It's efficient and often used in real-time data transmission where both parties have the shared key.

- **Asymmetric Cryptography**: Leverages a pair of mathematically related keys: a public key for encryption and a private key for decryption. This approach offers enhanced security and is commonly used for initial key exchange and digital signatures.

**Best of Both**: Hybrid encryption combines the speed of symmetric systems for data encryption with the security of asymmetric systems for secure key exchange.

### Use Cases

- **Symmetric Cryptography**: Ideal for bulk data encryption such as disk encryption, file transfer, and video streaming.
- **Asymmetric Cryptography**: Suited for tasks like secure key exchange, digital signatures, and public communication through secure sockets layer (SSL) for websites. 

### Notable Algorithms

- **Symmetric**: Common ones include AES, DES, and RC4.
- **Asymmetric**: Widely-used algorithms are RSA, DSA, and ECC (Elliptic Curve Cryptography).

### Algorithm Characteristics

#### Symmetric Key Encryption
- Its Strength: Relying on the secrecy of the shared key.
- Scalability: Efficient for large data sets.
- Key Length: Typically requires shorter key lengths than asymmetric algorithms.

#### Asymmetric Key Encryption
- Its Strength: Based on the mathematical complexity of operations, like prime factorization in the case of RSA.
- Computational Load: Due to key size and operations, it's computationally more demanding.
- Key Length: Requires longer key lengths for equivalent security compared to symmetric key systems.

### Vulnerabilities

- **Symmetric Cryptography**: Key distribution is a challenge. Even if securely distributed initially, key management can become vulnerable over time through potential loss or unauthorized disclosure.
- **Asymmetric Cryptography**: While key distribution is more straightforward due to the public nature of one of the keys, the challenge here is in ensuring the authenticity of the public key. This is addressed through the use of digital certificates.
<br>

## 3. What is a _cryptographic hash function_, and what properties must it have?

**Cryptographic hash functions** are one-way functions that transform data (such as a password or a file) into a fixed-size string of characters. They are extensively used for many security functions, including password protection, digital signatures, message integrity, and more.

### Properties of Cryptographic Hash Functions

For a hash function to be considered **cryptographic**, it must satisfy the following five essential properties:

1. **Deterministic**: The same input will always result in the same hash output. This is crucial for functions like password verification.
   
2. **Quick to Compute**: The hash function should produce outputs rapidly, making them suitable for systems requiring frequent hashing.

3. **Pre-Image Resistant**: Given a hash output, it should be computationally infeasible to derive the original input. This property is crucial to safeguard sensitive information.

4. **Second-Preimage Resistant**: If an attacker has one input, finding a second distinct input that produces the same hash should be infeasible. This property is fundamental for ensuring data integrity.

5. **Collision Resistant**: The hash function should make it arduous to find two distinct inputs that yield the same hash output (a collision).

For each of these properties, it's essential to consider the "speed and strength" balance. Specifically, the function must be efficient without compromising on security.
<br>

## 4. Describe the concept of _public key infrastructure (PKI)_.

**Public Key Infrastructure (PKI)** is a comprehensive system designed to manage **public-key encryption** and provide mechanisms for secure digital communication. PKI is characterized by a hierarchical structure involving **Certificate Authorities** (CAs) and includes a system for **key and certificate management**.

### Core Elements of PKI

1. **Key Pair Generation**: Each user generates a unique key pair that consists of a public and a private key using an algorithm like RSA, DSA, or ECC.
2. **Digital Certificates**: A Certificate Authority issues digitally-signed certificates, binding the public key to user identity information such as name, email, organization, or domain name. Certificates include metadata concerning the issuer, expiration date, and the key itself.
3. **Public Key Repository**: A secure, centralized place for storing public keys, typically embodied by a Certificate Authority.
4. **Key Revocation Mechanism**: A system for revoking, or declaring a certificate invalid, which is essential if a private key is compromised or if a certificate expires. This is often managed via Certificate Revocation Lists (CRLs) or Online Certificate Status Protocol (OCSP).
5. **Protocols for Secure Communication**: PKI employs cryptographic protocols such as SSL/TLS to enable secure data transmission, and S/MIME for secure email.

### Security Characteristics

- **Confidentiality**: Data encrypted with the recipient's public key can only be decrypted with the matching private key.
- **Data Integrity**: Digital signatures generated with the sender's private key ensure the data's integrity. Public-key decryption verifies the signature's authenticity.
- **Authentication**: Verifying the digital certificate's authenticity establishes the user's identity.
- **Non-repudiation**: Digital signatures, confirmed through public key decryption, prevent a user from denying they sent a message.

### PKI in Practice

In a typical enterprise setup, PKI is responsible for the following:

- **User Authentication**: During logins and other information-sensitive actions.
- **Secure Email Communication**: Ensuring emails are sent and received securely.
- **Digital Signatures**: Verifying the authenticity of signed documents, like legal contracts.
- **Virtual Private Networks (VPNs)**: Used to securely connect remote devices or employees to the corporate network.
- **SSL/TLS Certificates for Websites**: Visualized in the form of secure padlocks in web browsers.
- **Code Signing Certificates**: Ensuring software comes from a trusted source.

### Code Example: Generating a Private/Public Key Pair

Here is the Python code:

```python
from Crypto.PublicKey import RSA

# Generate an RSA key pair
key = RSA.generate(2048)

# Fetch the private and public keys
private_key = key.export_key(passphrase='mysecret')  # Export the private key with password protection
public_key = key.publickey().export_key()

print(private_key.decode('utf-8'))
print(public_key.decode('utf-8'))
```
<br>

## 5. What is a _digital signature_, and how does it work?

A **digital signature**  ensures the authenticity and integrity of digital content, such as documents, emails, and code.

It provides three key components:

### Core Components

1. **Key Pairs**: A digital signature relies on a public-private key pair. The signer uses a private key to create the signature, while the verifier uses the corresponding public key to confirm the signature's authenticity.

2. **Signing Algorithm**: A mathematically robust algorithm, such as RSA, ECDSA, or EdDSA, is used for creating the signature.

3. **Verification Algorithm**: A complementary algorithm is employed to validate the signature using the provided public key.

### Validation Process

   The following steps are taken to authenticate a digital signature:

   1. **Obtain Public Key**: The public key, usually from a digitally signed certificate, is retrieved.
   2. **Compute Hash**: A hash function generates a unique fixed-size string from the target document.
   3. **Decrypt Signature**: The digital signature is mathematically reversed and decrypted to reveal specific data tied to the signed content.
   4. **Match Hashes**: The obtained hash should be consistent with the one recovered through signature decryption.

If all these steps align and pass the checks, the signature is considered valid, confirming the authenticity and integrity of the associated content.

### Key Security

The strength of **digital signatures** is closely linked to the security of the associated public-private key pair.

In practice, this means the following:

- The private key should be carefully guarded, and access restricted to authorized parties.
- The private key also needs to be unique to the entity using it. If the key is compromised, it could potentially be used to create fraudulent **signatures**.

### Digital vs. Electronic Signatures

While both types of signatures aim to accomplish the same goals, the methods they use and, to some degree, the legal status of their authenticity can differ.

- **Electronic Signatures** encompass a broader range of methods, such as a scanned image of a handwritten signature, or even a simple click on an "I agree" box.
  
- **Digital Signatures**, on the other hand, are a specific technology that uses cryptographic methods to achieve **authentication** and **integrity**.

In many jurisdictions, including the United States and the European Union, digital signatures hold a higher legal status. They are often used in contexts that require stringent security measures and high levels of risk management, such as financial transactions.
<br>

## 6. Can you explain what a _nonce_ is and how it's used in _cryptography_?

In cryptography, a **nonce** (Number used ONCE) is a unique, usually random, string of bits or values often used in secure network communications and session establishments like TLS/SSL.

### Purpose of a Nonce

The primary role of a **nonce** is to prevent a range of attacks, such as replay attacks and man-in-the-middle attacks. This is achieved because, by definition, the **nonce** can only be used once.

### Common Techniques

- **One-Time NiOnces (OTNs)** aim to be unique for each message.
- **Cryptographic Hash Functions** use a static **nonce**, known as a "salt", in operations like password hashing.

### Common Implementations

- **Internet Security Protocols** like TLS and older versions of SNMP use **nonces** for safeguarding against replay attacks.
- **Cryptography Libraries** often employ **nonces** in a variety of applications, like PRNGs (Pseudo-Random Number Generators) to seed random number generation.

### Code Example: Using Nonces for Uniqueness

Here is the Python code:

```python
import os

def generate_nonce(length=32):
    return os.urandom(length)

# Example Usage
print(generate_nonce())
```

Ensure you are using a secure method for generating **nonces**. The `os.urandom()` function is recommended as it uses system-level entropy.
<br>

## 7. What does it mean for a _cryptographic algorithm_ to be "computationally secure"?

When a cryptographic algorithm is deemed "computationally secure", this means it **stays resistant** under the reasonable constraints of computational resources. While absolute security (theoretical or "information-theoretic" security) is an ideal, it is often impractical to achieve. However, computational security offers a practical alternative.

### Limitations of Computational Security

1. **Putnam's Model**: This is the computer scientist Aaron Putnam. The limitation is on the computational resources like time and memory that a computer has. "No computer can run forever," That's his basic model.

2. **Polynomial-Time Computations**: A computational algorithm is considered efficient if it can be executed in time proportional to some polynomial in the size of the input.

3. **AVS Model**: This is Andrew Yao, an American computer scientist. In the 80s, he laid down what's called the "Yao's Millionaire Problem". The problem is, he mentioned "two multi-millionaires decide to know who is richer, but they don't want to tell anything about their actual wealth to each other."

These two discoveries further validate the concept of Computational Security.

### The Right Tool for the Job

In real-life applications, cryptographic schemes need to reconcile security with efficiency. While **unconditional security** might sometimes be unachievable, **computationally secure** algorithms offer a balanced compromise between security and performance.

One famous example is the RSA cryptosystem, which bases its security on the difficulty of factoring the product of two large prime numbers.

Researchers continue to explore fundamental aspects of computational security, making strides in understanding and refining its practical applications.
<br>

## 8. Describe the concept of _perfect secrecy_ and name an _encryption system_ that achieves it.

Perfect secrecy, also known as **unconditional security**, is a property in cryptography where the information-theoretic security guarantees that the ciphertext reveals absolutely no information about the plaintext, even to an attacker with unlimited computational resources.

### Mathematical Basis for Perfect Secrecy

Perfect secrecy is achieved when the a **cipher** is a stochastic process that meets the condition:

$$
P(\text{key}) = \frac{P(\text{key}, \text{plaintext})}{P(\text{plaintext})} = P(\text{key} \, | \, \text{plaintext})
$$

It is essential that for **every pair** of possible plaintexts $p_i$ and $p_j$, the probability of generating a ciphertext $c$ is the same given $p_i$ or $p_j$.

If the above condition holds for a cipher, **Kerckhoffs' Principle** ensures the security of the system, establishing that only the secrecy of the key can guarantee the inviolability of the plaintext.

### Use of One-Time Pad for Perfect Secrecy

The One-Time Pad (OTP), a symmetric-key algorithm, remains the sole practical example of a cryptosystem that offers **perfect secrecy**. A crucial property of the OTP is that its key must be:

- At least the same length as the message to be encrypted.
- Truly random.

The unbreakable nature of the OTP rests on these two critical provisions.

While **perfect secrecy** is a desirable attribute, practical encryption systems such as AES, DES, and RSA employ different security paradigms, often referred to as computational or **probabilistic security**, to provide robust data protection in real-world scenarios.
<br>

## 9. What are _substitution_ and _permutation_ in the context of _encryption algorithms_?

Modern encryption systems are built on two foundational concepts: **substitution** and **permutation**.

### Substitution

In **substitution ciphers**, plaintext characters are replaced with ciphertext characters based on a set of rules defined by a key.

This key is often any distinct scrambling of the alphabet. Classical examples include the **Caesar cipher** (shifts letters by a fixed number) and the **Atbash cipher** (reverses alphabet order).

#### Challenges

- **Security**: Most simple substitution ciphers are easy to crack through frequency analysis.
- **Longevity**: They are routinely taught as educational paradigms. 

#### Code Example: Caesar Cipher

Here is the Python code:

```python
def caesar_cipher(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - base + shift) % 26 + base)
        else:
            result += char
    return result
```

### Permutation

In **permutation ciphers**, the order of characters in the plaintext is changed according to a predetermined rule.

For example, the **rail fence cipher** writes characters in a zigzag pattern across multiple "rails" before reading them horizontally. Another classic example is the **transposition cipher** which shuffles characters in the plaintext according to a predetermined sequence.

Permutation ciphers, while effective, aren't quite as straightforward or brute-force-resistant as modern ciphers.

#### Code Example: Rail Fence Cipher

Here is the Python code:

```python
def rail_fence(text, rails):
    fence = [[None] * len(text) for _ in range(rails)]
    rail, delta = 0, 1
    
    for char in text:
        fence[rail] = char
        rail += delta
        if rail in {0, rails - 1}:
            delta = -delta
    
    return ''.join(char for row in fence for char in row if char is not None)
```
<br>

## 10. Explain the basic principle behind the _AES encryption algorithm_.

The **Advanced Encryption Standard** (AES) is a symmetric-key algorithm and a leading choice for securing data. It uses a **block cipher** to encrypt or decrypt fixed-sized data blocks.

The AES algorithm employs what is known as a **substitution-permutation network** (SPN), incorporating **multiple rounds of substitutions, permutations, and key additions** to enhance security.

### Key Algorithmic Functions

The AES algorithm employs several core functions to process data and keys, including:

- **Nibble Substitution**: Substitutes each nibble (4-bit sequence) with another using a fixed SBox (Substitution Box).
- **Shift-Row Transformation**: Involves cyclically shifting rows in the state array.
- **Mix-Columns Operation**: Applies a matrix multiplication to each column in the state array.
- **Key Expansion**: Generates round keys from the initial encryption key.

### Operating on State Array

AES works with what is termed the **state array**, a 4x4 array of bytes that represents the data block under encryption.

- Galois Field Transformation: Some operations in AES are carried out using the arithmetic of a finite field.
- Round Operations: AES processes the state array in multiple rounds. The number of rounds depends on the key size and is usually 10, 12, or 14 for 128, 192, and 256-bit keys, respectively.

### Steps in AES Encryption

1. **AddRoundKey**: An initial step where each byte of the state array is XORed with a corresponding byte from the round key.

2. **SubBytes**: Bytes in the state array are subjected to a substitution from the SBox (non-linear transformation).

3. **ShiftRows**: bytes in rows of the state array are cyclically shifted.

4. **MixColumns**: Applies a linear transformation to groups of bytes.

5. **Final Round**: The last round omits the MixColumns operation.

### AES Key Lengths and Rounds

- **128-bit Keys**: 10 rounds
- **192-bit Keys**: 12 rounds
- **256-bit Keys**: 14 rounds

### Code Example: AES Encryption

Here is the Python code:

```python
from Crypto.Cipher import AES

# Encryption Function
def encrypt(plaintext, key):
    cipher = AES.new(key, AES.MODE_ECB)  # Using ECB mode for simplicity in this example
    return cipher.encrypt(plaintext)

# Decryption Function
def decrypt(ciphertext, key):
    cipher = AES.new(key, AES.MODE_ECB)  # Using ECB mode for simplicity in this example
    return cipher.decrypt(ciphertext)

# Sample Usage
key_128 = b'ThisIsA128BitKey'
key_192 = b'ThisIsA192BitKeyThatHasExtraBits'
key_256 = b'ThisIsA256BitKeyThatIsEvenLargerAndHasExtraBits'
plaintext = b'TestingAES'

ciphertext_128 = encrypt(plaintext, key_128)
ciphertext_192 = encrypt(plaintext, key_192)
ciphertext_256 = encrypt(plaintext, key_256)

print(decrypt(ciphertext_128, key_128))  # Expected Output: b'TestingAES'
print(decrypt(ciphertext_192, key_192))  # Expected Output: b'TestingAES'
print(decrypt(ciphertext_256, key_256))  # Expected Output: b'TestingAES'
```
<br>

## 11. What is the _Data Encryption Standard (DES)_, and why is it considered insecure today?

**DES**, a symmetric key cryptosystem, was developed by IBM in the 1970s for financial and governmental use. It was later standardized by NIST.

### Key Structure

- **Length**: 56-bits (including 8 parity bits)
- **Rotation**: Two 28-bit halves are independently rotated 

### Encryption Process

1. **Initial Permutation**: Rearranges the 64-bit key.
2. **16 Rounds**: Each round approximates a "Feistel Function."
3. **Final Permutation**: Swaps halves.

### Insecurity Issues

- **Brute-Forcing**: With modern computing power, a 56-bit key can be reliably brute-forced.
- **Cryptanalysis**: Multiple attacks exploit DES's procedures, decreasing its effectiveness.

### Original Proposal vs Modern Approach

- **DES**: Proposed by IBM, adopted by the U.S. government, and standardized by NIST **(1977)**.
- **Monero**: Started as a Bytecoin fork **(2014)** and gradually evolved, introducing advanced security and privacy features.

### Cryptanalysis of DES

DES has Several weaknesses, that makes it susceptible to attacks

1. **Brute-Force**: With increased computational power, brute-forcing a 56-bit key became feasible. 
2. **Differential Cryptanalysis**: Introduced in the 1970s by D. Davies and W. Price, this method aimed to find statistical patterns in the output of a substitution-permutation network.
3. **Linear Cryptanalysis**: Described in 1993 by Lars Knudsen and Vincent Rijmen, linear cryptanalysis aims to identify linear approximations that may be applied to the algorithm and exploit them to retrieve information.
4. **Meet-In-The-Middle Attack**: This method utilizes the structural aspect of a Feistel network to reduce the complexity of a complete key search.

In 1997, the Electronic Frontier Foundation built a "DES cracker" that could brute-force a DES key in days, leading to DES's further erosion as a secure encryption method.

### Replacing DES: The Triple DES and AES Evolution

In response to DES's vulnerabilities, NIST selected Rijndael, named AES, to replace DES in 2001. Developed by Joan Daemen and Vincent Rijmen, **AES** provides robust security and is currently used worldwide.

Additionally, **Triple DES** (3DES or TDEA) was a stopgap solution. It applies three rounds in succession, leading to a 168-bit key. While more robust, 3DES **is inefficient** and has been largely replaced by AES in modern environments.
<br>

## 12. Describe the differences between _RSA_ and _ECC (Elliptic Curve Cryptography)_.

Both **RSA** and **ECC** are public key cryptographic systems, but they have distinctive characteristics in terms of speed, key size, and applicability.

### Key Length and Security

RSA traditionally requires significantly longer keys for equivalent security compared to ECC.

RSA key lengths tend to be a few thousand bits long, while **ECC keys** can be as short as 256 bits. A 256-bit ECC key is generally considered to provide security equivalent to a 3072-bit RSA key.


### Computational Complexity

- Generating the keys:
    - **RSA**: Time-consuming, particularly for large keys.
    - **ECC**: Requires less computational effort. 

### Speed

- Encryption/Decryption:
    - **RSA**: Slower, especially for larger keys.
    - **ECC**: Faster, thanks to shorter key lengths.

- Signature Creation/Verification:
    - **RSA**: Generally faster.
    - **ECC**: Typically faster, especially for smaller messages.

### Memory and Bandwidth Usage

- Smaller Keys Benefit Communication Efficiency:
    - **RSA**: More memory-intensive.
    - **ECC**: Uses less memory.

### Real-World Applicability

- **RSA**: Commonly used in legacy systems and for tasks like key exchange in SSL/TLS cryptographic protocols.
- **ECC**: Often favored in modern systems due to its efficiency, being widely used in SSL/TLS for key exchange as well as in the Bitcoin and Ethereum blockchains.
<br>

## 13. How does a _stream cipher_ differ from a _block cipher_?

Cryptography operations, such as **encryption** and **decryption**, can be achieved using either **stream ciphers** or **block ciphers**, each with its own unique characteristics.

### Key Distinctions

- **Data Unit**: Block ciphers process data in fixed-size blocks, while stream ciphers operate on individual **bits** or **bytes**, enabling real-time encryption.
- **Latency**: As block ciphers require a full block of data before processing, they introduce latency. Stream ciphers, in contrast, process data in **segments** of predefined length, minimizing latency.
- **Synchronization**: Block ciphers can pose synchronization issues when used for multiple encrypted data streams. Stream ciphers, on the other hand, employ techniques to ensure proper **synchronization**.
- **Error Propagation**: Stream ciphers confine transmission errors to the bit or byte where they occur, minimally affecting subsequent data. Block ciphers, however, invoke a **domino effect**, propagating errors throughout the block.
- **Complexity**: Block ciphers commonly involve intricate algorithms and key setups, making them relatively complex. Stream ciphers, in contrast, are **simpler** and often use lighter computational methods.
<br>

## 14. Can you describe the _Feistel cipher structure_?

**Feistel ciphers** are symmetric-key block ciphers that use a **divide-and-conquer** approach: the input block is split into two halves, then these halves interchangeably undergo multiple rounds of transformations. At each round, one half is passed through a **substitution function** and combined with the other half via an **XOR** operation.

The beauty of the Feistel structure lies in its **simplicity**, **flexibility**, and **security**. It's the heart of many well-known ciphers, including **Data Encryption Standard (DES)** and the derived cipher **Triple DES**.

![Feistel Cipher Structure](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/cryptography%2Ffeistel-cipher-diagram.png?alt=media&token=8ee25a94-ba2c-4fce-945a-ea520d33cc69)

### Core Components

#### Key Component

Each round of the Feistel structure requires a unique key for the substitution function. This **independent key setup** for each round is a distinctive feature of Feistel ciphers.

#### Rounds of Operations

A standard Feistel cipher typically performs 16 rounds. However, the number of rounds can be adjusted based on security requirements.

For decryption, the process is the same as encryption, but the order of key usage is reversed. This way, at each round, the key previously used for encryption is used for decryption.

### Rounds in Feistel Cipher

Each round of the Feistel process comprises the following stages:

1. **Function Application**: A substitution function $F(K_{i})$ (where $K_{i}$ is the key for the $i^{\text{th}}$ round) is applied to one-half of the block, usually termed the "left" or "right" half, denoted as $F(R_{i-1}, K_{i})$.
2. **XOR Operation**: The outcome of the function application is XORed with the other half of the block.
3. **Data Swap**: The two halves are then swapped to form the output of the $i^{\text{th}}$ round: $(L_{i-1}, R_{i-1}) = (R_i, L_i \oplus F(R_{i-1}, K_i))$.

These three stages are iterated for the specified number of rounds, resulting in the final output of the last round, $(L_{n}, R_{n})$, which is transformed back into a full block to produce the ciphertext or plaintext, using different transformations for encryption and decryption.

### Practical Example: DES

The most influential Feistel cipher, DES, operates on 64-bit blocks and utilizes a 56-bit key. Although deemed secure upon its introduction, its key size is considered insufficient compared to modern standards.

Specific elements of DES include the following:

- **Function**: DES employs an intricate substitution function known as the **S-box**.
- **Rounds**: It executes 16 rounds for both encryption and decryption, each with its key derived from the main 56-bit key.

### Security of Feistel Ciphers

The security of a Feistel cipher is **intrinsically linked** to the difficulty of executing two independent tasks: reproducing the round keys and inverting each round's function. This bifurcation helps in understanding the **why and how** behind the construction.

Beyond the core design, a Feistel cipher's robustness is contingent on both the strength of its **substitution function** and the **quality** of the mechanism generating these **round keys**.
<br>

## 15. What are the key differences between _DES_ and _3DES_?

**Data Encryption Standard (DES)** and its successor **Triple DES (3DES)** are both symmetric key ciphers that encrypt data in fixed block sizes. While 3DES is essentially a strengthened version of DES, the two algorithms differ in several key ways.

### Key and Block Size

- **DES**: 64-bit block size, 56-bit key.
- **3DES**: 64-bit block size, 112-bit or 168-bit (with three distinct keys) key.

By using multiple keys and applying DES multiple times in a specific order, 3DES achieves its enhanced key strength.

### Algorithm Modes

- **DES**: Basic DES employs Electronic Codebook (ECB) mode.
- **3DES**: Can use several modes, including CBC, which integrates an initialization vector (IV) to enhance security.

### S-Box Usage

- **DES**: Lightweight S-Box permutations are used in each round.
- **3DES**: In each instance using a different key, distinct, complex S-Box permutations are applied.

### Security and Speed

- **DES**: Prone to brute force attacks and no longer considered secure. 
- **3DES**: Significantly more robust due to its multiple keys and iterations. However, it's slower (especially with three keys) and less secure than advanced algorithms like AES. Because of this, NIST deprecated 3DES in 2017.

### Practical Usage

Today, AES is the preferred symmetric key cipher due to its efficiency and proven security. Nonetheless, 3DES and DES are useful in specific legacy systems, particularly when AES compatibility is unavailable.
<br>



#### Explore all 55 answers here 👉 [Devinterview.io - Cryptography](https://devinterview.io/questions/software-architecture-and-system-design/cryptography-interview-questions)

<br>

<a href="https://devinterview.io/questions/software-architecture-and-system-design/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fsoftware-architecture-and-system-design-github-img.jpg?alt=media&token=521fd2a9-0d56-49c0-a723-9bd6ca081893" alt="software-architecture-and-system-design" width="100%">
</a>
</p>

