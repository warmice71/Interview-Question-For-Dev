# Top 100 Web Security Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - Web Security](https://devinterview.io/questions/web-and-mobile-development/web-security-interview-questions)

<br>

## 1. What is _web security_, and why is it important?

**Web Security** encompasses strategies and technologies aimed at protecting internet-connected systems, including web applications and services from various threats. It's a paramount consideration for businesses to safeguard data and maintain user trust.

### Fundamental Security Principles

- **Confidentiality**: Ensuring that sensitive information is accessible only to authorized entities.
- **Integrity**: Preserving the accuracy and trustworthiness of data.
- **Availability**: Making resources and services accessible when needed.

### Web Security Components

#### Transport Layer Security (TLS)

**TLS** serves as the foundation for secure internet communication, ensuring encryption and data integrity through mechanisms like symmetric and asymmetric encryption.

#### Access Control

- **Authentication**: Verifies the identity of users through credentials or multi-factor methods.
- **Authorization**: Governs user access to resources and services based on their permissions.

#### Security Headers

**HTTP Security Headers** are HTTP response headers designed to enhance web application security. They provide strict web-security policies, protect against specific attacks, and help detect and mitigate potential security vulnerabilities.

- **X-Content-Type-Options**: Prevents content type sniffing.
- **X-Frame-Options**: Protects against clickjacking.
- **Content-Security-Policy**:  Mitigates cross-site scripting attacks and other code injection attacks.
- **X-XSS-Protection**: Activates the Cross-site scripting (XSS) filter in web browsers.

#### Data Validity and Sanitation

Properly validating and sanitizing input data from users is crucial in preventing injection and manipulation attacks.

- **Cross-Site Scripting (XSS)**: Attacks involving the execution of malicious scripts in a user's browser.
- **SQL Injection**: Exploits database handling code to execute unauthorized SQL commands.

#### Anti-CSRF Tokens

**Cross-Site Request Forgery (CSRF)** tokens mitigate unauthorized requests sent by trusted authenticated users.

#### Session Management

For maintaining user sessions securely, it's essential to consider session token generation, expiration, and storage best practices.

- **Secure Cookie Flags**: Additional flags like "Secure" and "HttpOnly" help protect against certain types of attacks like session hijacking and cross-site scripting.
- **Session Regeneration**: Regularly changing session tokens minimizes the window of opportunity for attackers.

### Code Example: Setting HTTP Security Headers

Here is the Python code:

```python
from flask import Flask

app = Flask(__name__)

# Example: Setting Content-Security-Policy Header
@app.after_request
def add_security_headers(response):
    response.headers.add('Content-Security-Policy', 
                          "default-src 'self'; script-src 'self' 'unsafe-inline';")
    return response

if __name__ == '__main__':
    app.run()
```
<br>

## 2. Can you explain what _HTTPS_ is and how it differs from _HTTP_?

**HTTPS** (HyperText Transfer Protocol Secure) is an extension of HTTP that integrates security protocols for more secure data communication. It's essential for **maintaining privacy** and protecting sensitive data during online activities such as financial transactions, form submissions, and user logins.

### Key Features

- **Data Encryption**: Uses cryptographic protocols like SSL/TLS to encode messages.
- **Certificate Validation**: Authenticates servers and, in some cases, clients.
- **Data Integrity**: Employs hash functions to confirm that the message content has not been tampered with.

### How HTTP and HTTPS Differ

- **Security Mechanisms**: While HTTP utilizes plaintext for data transfer, HTTPS adds security protocols like SSL/TLS.
- **Port Number**: HTTP typically uses port 80, while HTTPS uses port 443.
- **Transfer Speed**: HTTPS might be slightly slower due to encryption.
- **Default Browser Behavior**: Modern browsers can mark HTTP websites as "Not Secure."
- **SEO & Search Engine Strategy**: Secure websites rank higher.

### The SSL/TLS Handshake

When a `Client` connects to a `Server`:

1. **ClientHello**: The `Client` initiates the communication, stating its requirements and capabilities.
2. **ServerHello**: The `Server` selects the best cipher suite and informs the `Client`. It also provides its public key if using asymmetric encryption.
3. **Certificate**: The `Server` sends its SSL certificate along with additional certificates, including the Certificate Authority's (CA) public key for validation.
4. **Key Exchange**: The `Client` and `Server` compute a shared session key, necessary for symmetric encryption.
5. **Finished**: Both parties send a hash-based verification message, ensuring secure channel establishment.
6. **Secure Data Transmission**: The shared secret key is used for symmetric encryption.

### Common Security Terminologies in HTTP(S)

- **SSL/TLS**: The protocols responsible for secure connections.

- **Certificate Authority (CA)**: A trusted entity that issues SSL certificates.

- **Cipher Suite**: A collection of data encryption, authentication, and key exchange mechanisms used for the secure connection.

- **Handshake Protocols**: The client and server agree on a suitable security configuration during handshake.

- **Public Key Infrastructure (PKI)**: The framework that controls the use and distribution of digital certificates.

### Code Example: Basic HTTP Server

Here is the Python code:

```python
from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')

httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
```
Use Command `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365` to create self-signed SSL certificate.

### Code Example: Basic HTTPS Server

Here is the Python code:

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import ssl

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, secure world!')

httpd = HTTPServer(('localhost', 4443), SimpleHTTPRequestHandler)
httpd.socket = ssl.wrap_socket(httpd.socket, certfile='cert.pem', keyfile='key.pem', server_side=True)
httpd.serve_forever()
```
<br>

## 3. What are _SSL_ and _TLS_, and what role do they play in web security?

**SSL** (Secure Sockets Layer) and **TLS** (Transport Layer Security) are cryptographic protocols that create a secure, encrypted connection between a client and a server. This ensures that the data being transmitted across a network is secure and can't be easily intercepted or tampered with. This is especially important for sensitive data such as login credentials, personal information, and financial transactions.

### Key Features

- **Authentication**: Certificates are used to verify the identity of the server, ensuring that the client is communicating with the intended server and not an impersonator.
  
- **Confidentiality**: Data transferred between the client and the server is encrypted, obscuring its contents from unauthorized parties.

- **Integrity**: By using digital signatures, SSL/TLS ensures that data remains unchanged during transmission. If any tampering is detected, the connection is terminated.

- **Forward Secrecy**: Every SSL/TLS session uses a fresh, unique encryption key, so even if one key is compromised, it won't affect previous or future communication.

- **Compatibility**: SSL and TLS are designed to be backward and forward compatible, allowing older and newer systems to communicate securely.

- **Protection against known vulnerabilities**: Modern versions of TLS incorporate the latest security practices and can mitigate risks like session hijacking, man-in-the-middle attacks, and more.

### Encryption Mechanisms

SSL/TLS utilizes different encryption algorithms. In the handshake process, they are used for key exchange and to establish a secure communication channel, categorically:

- **Symmetric Key Encryption**: A single key is shared between the client and the server and is used for both encryption and decryption. Common algorithms include AES and RC4.

- **Asymmetric Key Encryption**: Also called public-key cryptography, it uses a key pair - one public, one private. Data encrypted with one can only be decrypted with the other. This type is generally used for the secure exchange of symmetric keys and for initial authentication. Common algorithms include RSA, Diffie-Hellman, and Elliptic Curve Cryptography (ECC).

- **Hash Functions**: Ensure data integrity and are used to create a message digest or checksum. Common algorithms include SHA (Secure Hash Algorithm) variations like SHA-256.

### SSL/TLS Handshake

Before data exchange begins, the client and server establish a secure connection through a **handshake**, which includes the following main steps:

1. **Hello Messages**: The client and server exchange messages to initiate the handshake.

2. **Certificate Exchange**: The server presents its SSL/TLS certificate, which contains its public key and is signed by a recognized Certificate Authority (CA).

3. **Key Exchange**: If needed, a secure key for the session is exchanged using asymmetric encryption.

4. **Symmetric Key Session Creation**: A unique session key is generated (such as the session key for the symmetric key cipher) and is securely shared between the client and server.

5. **Establish Cipher Suite**: A set of encryption, hash, and handshake algorithms to be used throughout the session is agreed upon.

6. **Handshake Completion**: Both parties confirm the completion of the handshake process.

### Evolution: SSL to TLS

While SSL was foundational for the development of secure web communication, it has been largely **deprecated** due to well-documented security vulnerabilities.
This led to the creation of its successor, TLS, which has undergone several versions, each improving upon its predecessor in terms of security and functionality.

### SSL and TLS in the Browser

To check a website's SSL/TLS status in Google Chrome, for instance, you could click on the padlock icon and view the certificate. A more holistic picture of a website's security can be viewed by clicking on the 'Not secure' or the padlock icon. This shows additional security details, such as whether the connection is secure or not and if there are any security issues.
<br>

## 4. How do _SSL certificates_ work, and what is the purpose of a Certificate Authority (_CA_)?

Let's walk through the purpose and mechanisms of **SSL certificates** and the role of **Certificate Authorities**.

### Secure Sockets Layer & Transport Layer Security

**SSL/TLS** protocols provide a secure channel between two communicating machines, ensuring privacy, integrity, and authentication.

These are the main components:

1. **Digital handshake**: Establishes the session, negotiates encryption algorithm, and generates session keys.
2. **Key Exchange**: Public-key cryptography to share a secret key for further symmetric encryption, ensuring speed and security.
3. **Encryption**: The shared secret key is used for symmetric encryption of data.
4. **Verification**: Certificates provide the means for authenticating parties.

### Certificate Authority (CA)

A **Certificate Authority** is a trusted entity, responsible for validating entities' identities before issuing certificates vouching for their authenticity.

The primary functions of a CA are:

1. **Authentication**: Verifying the identity of entities requesting a certificate.
2. **Issuance**: Creating and digitally signing certificates for validated entities.
3. **Revocation**: Providing mechanisms to invalidate certificates in case they're compromised.

### Certificate Components

A standard SSL certificate comprises the following sections:

1. **Issuer Information**: Details of the authority issuing the certificate.
2. **Validity Period**: Start and end dates for when the certificate is considered valid.
3. **Subject Information**: Identifying information for the entity the certificate is issued to, such as the domain name.
4. **Public Key**: The vital piece of encryption enabling secure communication.
5. **Signature**: A hash of the certificate data signed with the private key of the issuing authority.

### Certificate Formats

SSL certificates primarily exist in two formats:

1. **Base64-encoded** $\rightarrow$ Useful for communication between systems or for human readability.
2. **DER (Distinguished Encoding Rules)** $\rightarrow$ A binary format with a more concise footprint, suited for machine use.

### Obtaining SSL Certificates

SSL certificates can be obtained in multiple ways. The primary methods are:

1. **Self-signed Certificates**: Suitable for individual or local testing, where a CA's involvement might not be necessary. Not typically used in production due to trust issues.
2. **Publicly-trusted CA Certificates**: Issued by established CAs and widely accepted by browsers and systems. These require thorough identity verification ensuring the certificate owner is legitimate.
<br>

## 5. What is the difference between _encryption_ and _hashing_?

Both **encryption** and **hashing** are cryptographic methods to enhance data security. However, they serve different purposes, and these roles come with distinct functionalities and characteristics.

### Core Differences and Use Cases 

- **Encryption** secures data in a reversible manner using keys, making it understandable only with the right key. This is useful to keep data confidential.

- **Hashing**, on the other hand, is one-way—it converts input data into a unique, fixed-length string, called a **hash value**, which represents the original data. This is useful to verify data integrity and authenticate identities, especially for passwords where the original data should never be needed or known.

### One-Way Process (Hashing)

Data goes through a one-way transformation process:

- **Input**: Original data like a message or password.
- **Hash Function**: A mathematical algorithm that processes the input into a fixed-length output known as the hash value.

#### Use Case: Password Management

Instead of storing plain-text passwords, it's safer to store their hashed equivalents. When a user submits a password for comparison, the system hashes the input and checks if the generated hash matches the stored one. This strengthens data security, ensuring that even if the stored has value is compromised, the original password remains protected.

### Two-Way Process (Encryption)

- **Input**: Data
- **Encryption Algorithm**: Transforms the input into ciphertext.
- **Key**: Essential for encryption. Without the right key, it's nearly impossible to decrypt the ciphertext back into its original form.

#### Computing the Encrypted Value

Here is the Python code:

```python
from cryptography.fernet import Fernet

# Generate a key to be used for encryption/decryption
key = Fernet.generate_key()

# Initialize the Fernet cipher using the generated key
cipher_suite = Fernet(key)

# Text to encrypt
text = b"Hello, this is a secret message!"

# Encrypt the text using the cipher
cipher_text = cipher_suite.encrypt(text)

print(f"Cipher Text: {cipher_text}")

# Decrypt the text back to its original form
original_text = cipher_suite.decrypt(cipher_text)
print(f"Original Text: {original_text.decode()}")
```
<br>

## 6. Define the concept of a _secure session_ and explain how it is established.

A **secure session** refers to the safeguarded interactive exchange between a web server and user-through a consistent, encrypted connection. This mutual trust is established via **session key management**, subsequently encrypting and decrypting all data transmissions.

### Session Key Management

  The **session key** is the cryptographic linchpin of secure sessions. It's a relatively short-lived symmetric key used for encrypting the data during the session. Both parties—the web server and the user's browser—obtain the session key to sustain the confidential data exchange.

      - The web server generates the session key and maintains sole ownership.
      
      - The session key is sent to the user's browser using asymmetric encryption (public-key cryptography). The user's public key is employed to encrypt the session key.
      
      - Subsequent communications during the session are enveloped with the session key.

### Key Generation

  - **Randomness**: The session key's security hinges on its unpredictability. Suitable cryptographic algorithms and high-entropy sources, such as hardware RNGs or truly random user-generated input, are essential for the key's randomness. Consistent hashing can provide a more streamlined key distribution.

  - **Key Size**: Longer key lengths bolster the session's security. 128-bit keys, for example, are generally deemed secure against common attacks.

### Session Key Establishment:

  A web server and a user together ensure that they each **have the session key**. This method's objective is for both parties to possess the key without compromising it at any time.

#### Diffie-Hellman Key Exchange

The Diffie-Hellman algorithm enables two parties to formulate a shared secret without directly transmitting cryptographic keys. Each party devises a private key and an auxiliary public key, and following a series of exchanges, they both compute the shared secret. This secret then functions as the session key.

```plaintext
  Here is the Key Exchange
  
  User:
  Private Key: a
  Public Key: g^a mod p
  
  Server:
  Private Key: b
  Public Key: g^b mod p
  Shared Secret: (g^a)^b mod p = g^(ab) mod p
```

#### Perfect Forward Secrecy (PFS)

Perfect Forward Secrecy ensures that mechanical developments or bargained session keys at a later stage won't compromise the entire session or any earlier interchanges. Instead, it uses a freshly minted session key for each new session.

PFS is particularly beneficial for long-term security, making certain that past sessions won't be breached, even if the current session becomes compromised.

### Additional Validation

To further bolster the security and confidence in the session establishment process, a series of mechanisms are often added:

  - **Certificates**: Verify the web server's identity, as well as establishing trust.
  
  - **Token-Based Systems**: Convert temporary tokens into sessions to keep user data secure even further.
<br>

## 7. What are some common web security vulnerabilities?

Understanding common web security vulnerabilities is crucial in developing robust and secure web applications. Common security vulnerabilities include:

### Injection

**Description**: Unsanitized user input gets interpreted as commands by the backend database, opening the door to attackers for executing arbitrary SQL commands.

**Example**: A user enters `1; DROP TABLE users--` leading the database to execute unintended commands.

#### Mitigation

- **Avoid Dynamic Queries**: Prefer parametrized queries or ORM tools.
- **Input Sanitization**: Use prepared statements, stored procedures, or input validation.

#### Code Example: SQL Injection Vulnerable Code

Here is the Python code:

```python
username = request.POST['username']
password = request.POST['password']

# Bad practice: Using string formatting
cursor.execute("SELECT * FROM users WHERE user='%s' AND password='%s'" % (username, password))
```

### Cross-Site Scripting (XSS)

**Description**: Vulnerability where web applications accept and display user-provided content (typically in HTML or JavaScript form) without escaping/sanitizing. This allows attackers to inject scripts, potentially compromising end-user systems or stealing data.

**Example**: Attacker inputs `<script>malicious_code()</script>` in a comment field. When a user views the comment, the script gets executed.

#### Mitigation

- **Sanitize User-Input**: Encode/escape user-provided content before rendering.
- **HTTP Headers & CSP**: Set proper `Content-Security-Policy` headers to restrict resource load from external sources.

#### Code Example: XSS-Prone Code

Here is the HTML code:

```html
<form action="/comments" method="POST">
    <textarea name="comment" placeholder="Enter your comment"></textarea>
    <button type="submit">Post Comment</button>
</form>
```

### Broken Authentication

**Description**: Weaknesses in the mechanisms used to authenticate the application's users, often arising from poor password management, insecure storage, or unprotected authentication tokens.

**Example**: Default or weak admin credentials, insufficient session management, or **Cross-Site Request Forgery** (CSRF) tokens being predictable or repetitive.

#### Mitigation

- **Strong Password Policies**: Enforce complexity and suggest periodic changes.
- **Secure Session Handling**: Use cookies with `HTTPOnly` and `Secure` attributes and non-predictable session IDs.

### Security Misconfigurations

**Description**: Inaccurate or incomplete server, application, or database configurations that may give attackers unauthorized access or provide sensitive data.

**Example**: Default settings for admin interfaces, verbose error messages with stack traces in production, or storage buckets set to public instead of private.

#### Mitigation

- **Review and Monitor Settings**: Regularly audit configurations.
- **Principle of Least Privilege**: Restrict access to resources based on necessity.

### Insecure Direct Object References (IDOR)

**Description**: When an application exposes internal implementation objects to users, potentially allowing malicious users to manipulate or access unauthorized data.

**Example**: An attacker changes a URL parameter meant to display their profile to access someone else's profile.

#### Mitigation

- **Use Indirect References**: Employ mappings or unique identifiers not directly linked to internal references.
- **Access Control Lists (ACLs)**: Implementing fine-grained access control mechanisms.

### Cross-Site Request Forgery (CSRF)

**Description**: Occurs when an attacker tricks a victim into performing actions on a website where the victim is authenticated. This could be anything from clicking on a URL to a website's form submission, via another website, without the victim's consent.

**Example**: An attacker embeds a hidden form in a malicious website that submits a request to a banking site that the victim is authenticated to.

#### Mitigation

- **Use Anti-CSRF Tokens**: Use tokens unique to authenticated users, or  employ CORS (Cross-Origin Resource Sharing) headers.  

#### Code Example: Vulnerable CSRF Code

Here is the HTML code:

```html
<form action="/make-transfer" method="POST">
    <input type="text" name="amount" />
    <input type="text" name="toAccount" />
    <button type="submit">Transfer</button>
</form>
```

### Unvalidated Redirects and Forwards

**Description**: Websites featuring links or buttons that redirect users can be manipulated by attackers to forward users to malicious sites through **URL Redirection** or **Open Redirect** vulnerabilities.

**Example**: A website has a feature for login-free access to a partner site, and an attacker constructs a link to the affected site disguised as a link to the intended site.

#### Mitigation

- **Whitelisting or Direct Mapping**: Validate user input and check links against a whitelist of known URLs or enforce direct mappings.

#### Code Example: Redirect Prone to Attack

Here is the Python code:

```python
redirect_url = request.GET['redirect']
# Bad practice: Directly redirecting based on user input
return redirect(redirect_url)
```
<br>

## 8. Can you explain the _Cross-Site Scripting (XSS)_ attack and how to prevent it?

In the context of web applications, **Cross-Site Scripting** (XSS) refers to the injection of malicious scripts into otherwise benign and trusted websites or web applications.

### What an XSS Attack Does

By executing **arbitrary code** within a victim's browser, an XSS attack can:

- Steal Sensitive Information: Obtain cookies, session tokens, or even keystrokes.
- Deface Websites
- Redirect Users: Forwards them to malicious sites.
- Spread Worms: Perpetuates the attack by infecting other users.

### Types of XSS Attacks

1. **Stored XSS**: Scripts are stored in a database or other data source and then displayed to users on web pages.
2. **Reflected XSS**: The injected script is reflected off a web server, such as in an error message, and then executed within the user's browser.
3. **DOM-based XSS**: The attack occurs solely within the client-side HTML, bypassing typical server-based security measures.

### Key Sources of Vulnerability

1. **User Input**: All user inputs, including form fields and query parameters in URLs, are potential XSS sources.
2. **Client-Side Data**: Data from GET requests, the document URL, and anchor elements (`<a>`) can also be vectors for XSS.

### Key Defensive Measures

- **Output Encoding**: Convert special characters into their corresponding HTML entities, so they are displayed as text rather than executed.
- **HTTP Headers**: Utilize Content Security Policy (CSP) and the `X-XSS-Protection` header to bolster defense.
- **Input Validation**: Use strict input criteria and sanitize the input to prevent any script injection.
- **Session Cookies**: Implement the `HttpOnly` and `Secure` attributes to secure cookies.

### Vulnerable Code Example

Here is the PHP code:

```php
echo $_GET['greeting'];
```

A potential attack URL:

```url
example.com?greeting=<script>alert('Hello, world!')</script>
```

### Mitigation and Strong Input Validation

#### Key Anti-XSS PHP Functions:

- **htmlspecialchars()**: Converts special characters to HTML entities. The second parameter is essential to utilize, as it specifies the encoding mechanism.

  Example:

  ```php
  $output = htmlspecialchars($_GET['input'], ENT_QUOTES, 'UTF-8');
  ```

- **filter_var()**: particularly with `FILTER_SANITIZE_STRING`.

  Here is the PHP code:

  ```php
  $clean_input = filter_var($_GET['input'], FILTER_SANITIZE_STRING);
  ```

  However, be mindful that `FILTER_SANITIZE_STRING` might remove characters Unicode characters that are not within the 00-7F range.

- **ctype**: Functions such as `ctype_alpha()` can help verify if only alphabetic characters are present in a string.

  Example:

  ```php
  if (ctype_alpha($_GET['username'])) {
      // Username contains only alphabetic characters
  }
  ```
  
### Best Practices for Prevention

- **Code Review**: Regularly scrutinize code for potential vulnerabilities.
- **Security Modules**: Consider integrating third-party security modules.
- **User Education**: Ensure users are aware of the risks and assist them in recognizing secure communication practices.
- **Sandboxed Environments**: Test or execute untrusted code in secure, isolated environments.
<br>

## 9. What is _SQL Injection_ and how can you defend against it?

**SQL Injection** is one of the most dangerous and prevalent attack vectors. It involves maliciously altering SQL statements through web applications. This tactic can lead to data theft, data manipulation, and even unauthorized system access.

### Common Methods for SQL Injection Attackers

- **Identification of Vulnerable Input Points**: This often involves forms or URL parameters.
- **Injection Through User Input**: Attackers provide manipulated input, such as altering SQL queries with additional conditions or as commands.
- **Data Exfiltration or Manipulation**: The goal could be anything from obtaining sensitive data to tampering with the system.

### Guarding Against SQL Injection

- **Parameterized Statements**: Use placeholder parameters in SQL statements and pass the user-provided values separately.
- **Stored Procedures**: Centralize SQL logic, and call specific procedure names instead of constructing ad hoc SQL queries.
- **ORMs and Data Access Libraries**: These tools often provide a way to interact with the database without directly writing SQL.
- **Input Sanitization**: Validate and sanitize user input to mitigate injection risks.

### Code Example: Parameterized Query in Python

Here is the Python code:

  ```python
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Unsafe, vulnerable to SQL injection
query = "SELECT * FROM users WHERE username='%s' AND password='%s'" % (username, password)
cursor.execute(query)

# Safe, using parameterized query
query = "SELECT * FROM users WHERE username=? AND password=?"
cursor.execute(query, (username, password))

# Close
conn.close()
  ```
<br>

## 10. Describe what _Cross-Site Request Forgery (CSRF)_ is and how to prevent it.

**Cross-Site Request Forgery (CSRF)**, often called session riding or one-click attack, tricks users into executing unwanted actions while authenticated on another site.

### Mechanism

- A **victim** accessing a malicious site remains authenticated on a trusted site in a separate tab or browser session.
  
- The **attacker** loads an image, an iframe, or a script tag with a request to the trusted site, which, due to the victim's active session, gets executed without the user's consent.

This breach occurs because web applications **fail to validate** the source of incoming requests.

### Prevention Strategies

1. **Same-Origin Policy**: Modern browsers restrict how a web page can interact with resources from other domains, mitigating many CSRF scenarios. While generally effective, this approach has some limitations.

2. **Synchronizer Token Pattern**: The web application embeds a unique, random token in each HTML form and/or HTTP request. Upon submission, the server checks for this token's presence and its consistency with the user's session. This technique, known as a **CSRF token**, provides more robust defense.

3. **Double-Sided Cookies**: In addition to the user's session cookie, the web server generates a second cookie, often accessible only via HTTP headers. This additional layer of security becomes invaluable if the primary session cookie is compromised.

4. **User Confirmation**: Triggering critical actions only through a direct user interaction, such as a button click and not an automated request, deters common CSRF exploits.

5. **Always Use HTTPS**: HTTPS incorporates encryption and digital certificates to provide a secure channel for data communication, strengthening the defense against CSRF and numerous other attack vectors.
<br>

## 11. Explain the _Same-Origin Policy_ and its importance in web security.

The **Same-Origin Policy** (SOP) is a foundational web security concept that guards against **Cross-Origin Attacks** and data exposure, ensuring a safer and more private browsing experience.

### Core Principle

**SOP** mandates that a web browser should restrict interactions between documents from different origins to prevent potential security breaches. 

### Identifying 'Origin'

An **origin** refers to the combination of :

- **Protocol**
- **Domain**
- **Port**

For two webpages to share a common origin and thereby avoid SOP restrictions, all these components must match.

### Management of Cross-Origin Interactions

Browsers enable controlled cross-origin access using mechanisms such as:

- **Cross-Origin Resource Sharing (CORS)**: Servers can specify web resources that are accessible to various origins, relaxing the SOP's strictures.

- **Iframes**: These are windows embedded within a web page, useful for integrating third-party content, such as social media feeds or ads. Modern iframes are configurable, enabling or restricting cross-origin behavior.

### Importance for Web Security

The SOP is a **cornerstone of web security** because it:

- **Safeguards User Data**: SOP prevents malicious websites from arbitrarily accessing sensitive user information present on legitimate websites in the same browser session.

- **Mitigates Cross-Site Scripting (XSS)**: By limiting scripts and other dynamic content to their origin, SOP thwarts numerous XSS attack vectors.

- **Curbs Cross-Site Request Forgery (CSRF)**: SOP helps in eliminating unauthorized, cross-site requests that can manipulate a user's session on a different origin.

- **Averts Cookie Theft**: Data like user session tokens stored in cookies can't be pilfered across origins, bolstering user authentication.

- **Sustains Web Application Confidentiality**: Supports the confidentiality of web applications and assists in the prevention of information leakage.

### SOP Limitations

While SOP is a powerful and fundamental security tool, it has some limitations, such as:

- **Potential Misperception**: A website with numerous subdomains might confuse users on what exactly constitutes the 'same origin,' leading to accidental data exposure.
- **Loopholes in Older Browsers**: SOP's efficacy can vary across browsers. Earlier versions might not enforce it uniformly, enabling potential security breaches.
- **Incompatibility with Decentralized Systems**: Certain dispersed systems, like those using blockchain and decentralized file sharing, struggle with SOP's constraints, impeding their seamless operation.
<br>

## 12. What is _Clickjacking_, and what measures can prevent it?

**Clickjacking**, also known as a UI redress attack, is a form of web security threat in which a user is **tricked into clicking on something different from what they perceive, often leading to unintended actions**. This issue usually arises from cross-site scripting (XSS) vulnerabilities on a website.

### Common Clickjacking Techniques

- **Iframe Overlays**: Attackers place a transparent, positioned iframe on top of an innocuous element to intercept clicks.
- **CSS-Based Overlays**: By manipulating z-index and div positioning, attackers can make legitimate buttons appear beneath seemingly innocuous elements.
- **Drag & Drop**: Simulates user actions to trick users into dragging sensitive information onto a malicious target.

### Defense Mechanisms Against Clickjacking

#### X-Frame-Options

The **`X-Frame-Options`** HTTP response header mitigates clickjacking by controlling frame embedding. It has three possible settings:

1. **DENY**: Disallows any form of framing, ensuring sites load in stand-alone mode.
2. **SAMEORIGIN**: Limits framing to the same origin (domain), shielding against external embedding.
3. **ALLOW-FROM uri**: Permits limited framing from a specific URI, allowing conditional embedding.

#### Implementation:

- **HTTP Header**:

  ```http
  X-Frame-Options: SAMEORIGIN
  ```
  
- **Meta Tag**:

  This meta tag should be placed in the `<head>` section of HTML pages.

  ```html
  <meta http-equiv="X-Frame-Options" content="SAMEORIGIN">
  ```

#### Content Security Policy (CSP)

**CSP** is a powerful header that defines approved sources for different types of content, offering granular control over what can be loaded on a page.

- **frame-ancestors Directive**: Limits frame embedding. Setting it to `'self'` or a specific domain can help prevent clickjacking.

  ```http
  Content-Security-Policy: frame-ancestors 'self'
  ```

#### JavaScript-Based Defense

- **Frame-Busting Scripts**: Adopts JavaScript to ensure a page doesn't load in a frame, thereby preventing clickjacking.

  For instance, incorporate the below code into the `<head>` section:

  ```html
  <script>
    if (self !== top) {
      top.location = self.location;
    }
  </script>
  ```

- **Pointer Lock API**: More elaborate, this API can be leveraged to confine mouse movements to a specified region on a web page.

   Using the Pointer Lock API might not be a general solution; however, the lock will prevent an iframe from moving its related elements during clickjacking attempts.

  startPointerLock()

### Ensure Safe Browsing Experience with Clickjacking Protection

When an application or website deploys robust defense mechanisms like **X-Frame-Options**, **Content Security Policy**, and **JavaScript-based tactics**, it substantially reduces the risk of clickjacking, ensuring a secure and trustworthy browsing experience for users.
<br>

## 13. How can _web cookies_ compromise security, and how do you safeguard against these risks?

**Cookies** can pose security risks like **session hijacking**, **cross-site scripting**, and **data exposure**. But these can be mitigated with specific approaches.

### Risks and Mitigations

#### Session Hijacking

- **Risk**: Unauthorized access if a session ID is stolen.
- **Safeguard**: Use HTTP-only and Secure flags. Regularly change session IDs.

#### Cross-Site Scripting (XSS)

- **Risk**: Attackers inject malicious scripts, stealing user data or sessions.
- **Safeguard**: Sanitize inputs; utilize `HttpOnly` and `Secure` flags; employ cookie prefixes, e.g., `__Host-` and `__Secure-`.

#### Data Exposure

- **Risk**: Sensitive data sent in cookies can be intercepted.
- **Safeguard**: Use cookies **only** for essential session data.

### Best Practices

- **Limit Cookie Data**: Avoid transmitting sensitive user information.
- **Use Secure and HTTP-Only**: Set these attributes to protect against unauthorized data access and transmission over unsecured channels.
- **Regularly Update Session Identifiers**: For added security, update session identifiers periodically.
- **Encrypt Cookies**: If essential data has to be stored, ensure it's encrypted.
- **Cookie Prefixes**: Use `__Host-` and `__Secure-` to provide additional security.

### Implementation Example: Setting up Secure Cookies

Here is the Python code:

```python
from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/')
def index():
    resp = make_response(render_template('index.html'))
    resp.set_cookie('user_id', '1234', secure=True, httponly=True, samesite='Strict')
    return resp

if __name__ == '__main__':
    app.run()
```
<br>

## 14. What is a _Man-in-the-Middle (MitM) attack_ and how can it be prevented?

**Man-in-the-Middle (MitM)** attacks occur when a malicious actor intercepts communication between two parties. This breach can lead to data theft or falsification of information.

### Common MitM Scenarios

- **Wi-Fi Networks**: Unsecured public Wi-Fi networks are prime targets for MitM attacks.
- **Spoofed Sites**: Attackers create fake websites to capture login credentials.
- **Email**: Encrypted emails can be intercepted, read, and modified.

### Preventive Measures

1. **Digital Certificates**: Use SSL/TLS on servers and ensure proper certificate validation on clients.
2. **Encryption**: Apply end-to-end encryption to communication channels.
3. **Public Key Infrastructure (PKI)**: Employ validated digital certificates for secure identification and data integrity.
4. **Multi-Factor Authentication (MFA)**: Add layers of identity checks beyond passwords.
5. **Strong Password Policies**: Mandate complex passwords and encourage regular updates.
6. **Awareness Programs**: Train users to spot suspicious behavior and potential MitM indicators.
7. **Secure Applications**: Use authenticated, verified, and updated software. Ensure regular security audits.

### Code Example: SSL/TLS in Node.js

Here is the Node.js code:

```javascript
const https = require('https');
const fs = require('fs');

const options = {
  key: fs.readFileSync('key.pem'),
  cert: fs.readFileSync('cert.pem')
};

https.createServer(options, (req, res) => {
  // Handle secure requests
}).listen(443);
```

### Additional Layers with Transport Layer Security (TLS)

- **Certificate Pinning**: Associating a specific certificate with an application to prevent unauthorized replacements.
- **HTTP Strict Transport Security (HSTS)**: Directing web browsers to interact with the server using only HTTPS.

### Network Mechanisms to Counter MitM Threats

- **Ethernet Authentication**: IEEE 802.1X allows for network access control, reducing MitM risks.
- **Wi-Fi Protected Access 2 (WPA2)**: Enhanced security protocols over WEP and WPA, bolstering wireless security.
- **Intrusion Detection Systems (IDS) and Intrusion Prevention Systems (IPS)**: Active monitoring for suspicious network activity.
<br>

## 15. Describe the concept of _session management_ in web security.

**Session management** is a pivotal mechanism in web security. It maintains state between web pages and securely manages users' identities and permissions. It employs a combination of cookies, tokens, and server-side data to monitor user activities across a single session.

### Importance of Session Management for Web Security

- **Data Protection**: Sensitive user details are less exposed, mitigating the risk of unauthorized access.
- **Access Control**: It ensures that only authenticated users access protected resources.
- **User Privacy**: By limiting tracking to the current session, it enhances user privacy.

### Common Session Management Mechanisms

- **Cookies**: Text files stored on the user's device that include session identification.
- **URL Rewriting**: Append session IDs to URLs.
- **Hidden Form Fields**: Store session IDs in web forms.
- **HTTP Headers**: Utilize custom headers for session tracking.
- **Sessions**: Server-side storage, maintained between server and client via a unique session ID.

### The Risks Associated with Poor Session Management

- **Session Hijacking**: Unauthenticated entities assume control over a session.
- **Session Fixation**: Attackers manipulate the session to gain unauthorized access.
- **Cross-Site Request Forgery (CSRF)**: Malevolent sites instigate user action on another site where the user is authenticated.

### Best Practices for Secure Session Management

- **SSL/TLS Encrypted Connections**: Deploy HTTPS to secure data in transit.
- **Time-Out Sessions**: End inactive sessions after a predefined period.
- **Randomized Session IDs**: Generate unique, hard-to-guess session IDs.
- **Server-Side Validation**: Crosscheck session details on the server for added security.
- **Cookies with Secure and HTTPOnly Flags**: Cookies should only be sent over secure connections and remain inaccessible to JavaScript.

### Code Example: Setting a Secure Cookie

Here is the Java code:

```java
Cookie sessionCookie = new Cookie("sessionID", UUID.randomUUID().toString());
sessionCookie.setSecure(true);  // Ensures the cookie is only sent over HTTPS
response.addCookie(sessionCookie);  // Add the cookie to the response
```

In a Java web application, the `response` object is used to send the cookie to the client.

### Common Web Vulnerabilities Related to Session Management

- **Session Prediction/Inference**: Predictable session IDs increase the vulnerability to attacks.
- **Misconfigured Cross-Origin Resource Sharing (CORS) Headers**: Poorly configured headers might allow unauthorized websites to access a user's session data.
- **Transport Layer Security (TLS) Misconfigurations**: Weak or outdated TLS settings may compromise session security.
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - Web Security](https://devinterview.io/questions/web-and-mobile-development/web-security-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

