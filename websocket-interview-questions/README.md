# 100 Must-Know Websocket Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - Websocket](https://devinterview.io/questions/web-and-mobile-development/websocket-interview-questions)

<br>

## 1. What is WebSocket protocol and how does it differ from HTTP?

**WebSocket** and **HTTP** serve as communication protocols in web development, but they have different structures, behaviors, and best-fit applications.

### Core Differences

- **Unidirectional vs. Bidirectional**: HTTP operates unidirectionally, sending requests from the client to the server and receiving responses. In contrast, WebSockets support full-duplex communication, enabling data flow in both directions.

- **Connection Establishment**: HTTP initiates a connection purely through a client's request, and the server responds. On the other hand, WebSockets rely on a handshake mechanism for connection initiation, facilitating ongoing communication without the need for separate individual HTTP requests.

- **Header Overhead**: HTTP is heavier, primarily due to the necessity of headers in every request and response, containing metadata for the communication. WebSockets, after the initial handshake, carry fewer overheads.

- **Data Types**: Though both protocols facilitate the exchange of text or binary data, WebSockets excel in handling standardized data structures, like JSON and message framing.

### Operation Mechanism

- **HTTP**: It uses the familiar request-response model. When a client initiates interaction, it sends a request, and the server processes the request before responding. The connection is usually short-lived.

- **WebSockets**: After the initial handshake through an HTTP Upgrade message, the connection remains active, enabling data to travel in both directions with low latency. Once established, WebSockets typically persist.

### Protocol Stack Integration

- **HTTP**: It primarily sits at the application layer of the OSI model.

- **WebSockets**: It builds atop the HTTP protocol for the initial connection establishment and then operates at the application layer.

### Best-Candidate Use Cases

- **HTTP**: Most suitable for stateless, request-response scenarios, such as loading web pages, submitting forms, and downloading files.

- **WebSockets**: Ideal for applications where real-time, bidirectional communication is vital, spanning scenarios like online gaming, collaborative editing tools, and stock trading platforms.
<br>

## 2. Explain how the WebSocket handshake works.

The **WebSocket handshake** enables HTTP to evolve into a persistent, full-duplex communication channel by upgrading the initial HTTP request into a WebSocket connection. Here is a step-by-step explanation of the process.

### Handshake Process

1. **Client Request**: A WebSocket-compatible **client** initially sends a standard HTTP request to the server, presenting an `Upgrade` header.

    ```http
    GET /chat HTTP/1.1
    Host: server.example.com
    Upgrade: websocket
    Connection: Upgrade
    Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==
    Origin: http://example.com
    Sec-WebSocket-Protocol: chat, superchat
    Sec-WebSocket-Version: 13
    ```

2. **Server Response**: Upon receiving the client's request, the server evaluates it for WebSocket compatibility. If valid, the server responds with an `HTTP 101` status code and the `Upgrade` header.

    ```http
    HTTP/1.1 101 Switching Protocols
    Upgrade: websocket
    Connection: Upgrade
    Sec-WebSocket-Accept: HSmrc0sMlYUkAGmm5OPpG2HaGWk=
    ```

3. **Security Key Verification**: Both the client and server use cryptographic functions to confirm handshake integrity. The server appends a predefined `magic string` to the client's key and then computes the SHA-1 digest. If the calculated hash matches the `Sec-WebSocket-Accept` header, the handshake succeeds.

    ```javascript
    const magic = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
    const serverKey = crypto.createHash('SHA1')
        .update(clientKey + magic,'binary')
        .digest('base64');
    ```

     **Client Key**: `Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==`  
     **Server Computed Key**: `Sec-WebSocket-Accept: HSmrc0sMlYUkAGmm5OPpG2HaGWk=`

4. **Bi-directional Communication**: Upon successful verification, both the client and server transition into **full-duplex** mode, enabling concurrent data transmission in both directions.
<br>

## 3. What are some common use cases for WebSockets?

**WebSockets** fills several communication gaps experienced in traditional web environments, enhancing real-time interactivity.

### Use Cases

**Chat Applications**: Delivers real-time messaging with reduced server overhead. Particularly useful in group chats and when delivering notifications.

**Gaming**: Supports real-time, multiplayer game interactions such as moves, chats, and scores.

**Interactive Dashboard**:  Provides seamless live updates for data visualization and reporting, useful for financial, IoT, and analytics platforms.

**Live Customer Support**: Ensures instant direct interaction between customer support representatives and users.

**Collaborative Tools**: Facilitates real-time teamwork in productivity apps, such as Google Docs for simultaneous editing or Whimsical for shared whiteboards.

**Real-Time Editors**: Enables shared editing of text, code, and media in real time, like Google Docs and CodeSandbox.

**Interactive Maps**: Offers responsive real-time map updates, essential for GPS and logistics apps.

**Stock Market Tracker**: Displays live stock data, fluctuating prices, company news, and more, vital for traders and financial analysts.

**Real-Time Communication**: Powers features like VoIP, video conferencing, and screen sharing in communication apps like Slack and Zoom.

### Code Example: Sending Real-Time Currency Data

Here is the JavaScript code:

```javascript
// Establish WebSocket connection
const ws = new WebSocket('wss://currency-data-stream.com');

ws.onopen = function () {
    console.log('WebSocket connected.');
    // Subscribe to Euro updates
    ws.send(JSON.stringify({ action: 'subscribe', target: 'EUR' }));
};

// Handle incoming data
ws.onmessage = function (event) {
    const data = JSON.parse(event.data);
    console.log('Received currency data:', data);
    
    // Update UI with live data
    updateCurrencyUI(data);
};

// Simplified UI update function
function updateCurrencyUI(data) {
    // Update relevant UI elements with live currency prices
}
```
<br>

## 4. What are the limitations of WebSockets?

**WebSockets** are a powerful tool for bidirectional communication between clients and servers in real-time web applications, but they do have their limitations.

### Key Limitations

- **Firewall Interference**: Some firewalls or network setups might block WebSocket connections. This can interfere with the smooth functioning of real-time web applications.
  
- **Latency vs. Throughput**: WebSockets are optimized for low-latency data transmission. If your application needs high-throughput data transfer, this can lead to suboptimal performance.

- **Connection Overhead**: The initial handshake and setup of a WebSocket connection might lead to overhead. This could be a concern for applications that require frequent short-lived connections.

- **Noisy Neighbor Effect**: When multiple applications on the same server are using WebSockets, they might compete for server resources, potentially leading to poor performance for all applications. 

- **State Management**: WebSockets maintain a persistent connection, which could lead to challenges in managing server state, especially in cases of server restarts or updates.

### Protocol Flexibility

In many scenarios, the bidirectional communication required by WebSocket might not be necessary, making the protocol overkill for the task at hand. In such situations, **HTTP** (especially in the form of **HTTP/2**, which supports multiplexing, header compression, and server push) can be a more efficient choice. Moreover, some older systems or browsers might not support WebSocket, making HTTP a more universal choice.

### Deployment Complexity

Although WebSocket-enabled tools and libraries are widely available, integrating, debugging, and securing WebSocket connections might add to the deployment complexity. In certain cloud environments or containerized setups, additional configurations might be needed to support WebSocket connections.

### Resource Consumption

Server resources are consumed more persistently with WebSockets due to the maintenance of the open connection. With HTTP, a server is informed about the completion of a specific exchange (request/response), allowing it to free up resources more promptly. This continuous resource consumption in WebSocket connections can lead to inefficient server resource management.

### Mobile and Battery Impact

Persistent WebSocket connections can impact battery life on mobile devices. Maintaining an active data connection can be particularly demanding, especially in scenarios where bandwidth is limited or intermittent.

### Security and Infrastructure Compatibility

While the WebSocket protocol itself is secure, accessing WebSocket endpoints via unsecured means, such as via **unencrypted HTTP**, can pose security concerns. In such cases, secure alternatives like HTTPS should be used.

Firewalls and load balancers that are not configured to handle WebSocket connections can also cause limitations. Typically, for WebSocket connections to function correctly, these network devices must be able to carry and decipher WebSocket traffic.

### Service Stability

For serverless applications, continuously maintaining WebSocket connections might be challenging or impractical. Many serverless services work best with stateless communication models, where messages are passed along, processed, and promptly returned without the need for ongoing, persistent communications.

### Code Complexity and Additional Libraries

Using raw WebSocket APIs can be more labor-intensive compared to integrating specific libraries designed for real-time web tasks. These specialized libraries could come bundled with extra features such as auto-reconnection and message-queueing.

### Browser Compatibility

Although widely supported in modern browsers, old or less common browser versions might lack, or have imperfect support for, the WebSocket protocol. This can sometimes necessitate the inclusion of fallback mechanisms, typically, additional code that switches to alternative transport methods like long polling or server-sent events when direct WebSocket communication is not available.

### Cross-Domain Restrictions

By default, WebSocket connections, like most modern web operations, are subjected to cross-origin restrictions. Server administrators can choose to whitelist domains or use Cross-Origin Resource Sharing (CORS) to broaden accessibility. In some network setups, especially more restrictive ones, none of these methods might work, leading to connectively difficulties.

### Diagnosis and Testing

Debugging WebSocket connections might not be as straightforward compared to typical HTTP transactions. Specialized tools, like browser consoles or network traffic analyzers, might be essential to identify and rectify issues.

### Compliance and Legal Considerations

In certain sectors or regions, regulations like the General Data Protection Regulation (GDPR) in the EU or data privacy laws in the US might impose restrictions or requirements related to data persistence, which could impact the use of WebSocket connections due to their persistent nature. It's important to verify the compliance of the complete stack, including the use of WebSocket connections, with such regulations.

### Rate Limiting and Authentication

Managing and enforcing rate limits and authentication mechanisms in WebSocket connections can sometimes be less evident when compared to traditional HTTP requests, necessitating additional attention and specific strategies for each to ensure stability and security.
<br>

## 5. Can you describe the WebSocket API provided by HTML5?

WebSocket is a **communication protocol** that provides full-duplex, low-latency communication over a single, persistent connection. Developed as a part of the HTML5 specification, it enables **bi-directional** real-time communication.

### Key Aspects of WebSockets

- **Protocol Upgrade**: The WebSocket protocol is based on a standard handshake mechanism, initiated using the underlying HTTP or HTTPS protocols. This allows for enhanced security and firewall traversal capabilities.

- **Dual Data Channels**: Data, in either text or binary form, flows **simultaneously** in both directions. 

- **Native Integration with the Browser**: The WebSocket protocol is directly supported by modern web browsers, obviating the need for third-party plugins.

### WebSocket Workflow

1. **Handshake**: 
   - The process begins with an **HTTP-based handshake**, where the server and the client mutually agree to upgrade the connection to WebSocket.
   - The **upgrade request** (from the client) and **response** (from the server) contain specific headers for the WebSocket protocol.
   - If the server accepts the upgrade, the connection transitions to a full-duplex WebSocket, marking the end of the handshake.

2. **User Sessions**: Both clients and servers maintain a persistent session, eliminating the need for frequent re-establishment of connections.

3. **Data Transmission**: After the handshake, text or binary messages can be exchanged, and either end can initiate the traffic.

4. **Termination**: The connection can be terminated by either the client or server explicitly, or due to issues like timeouts or network disruptions.

### Key Use-Cases

- **Real-Time Web Applications**: Provides a streamlined vehicle for transmitting up-to-the-second data between the server and the client, crucial for various use-cases like live sports scores, news tickers, and others.

- **Interactive Gaming**: WebSocket's low latency and bidirectional nature make it a natural fit for real-time, multi-player gaming experiences.

- **Collaborative Tools**: Ensures seamless, instantaneous data sharing in collaborative tools such as shared document editors or real-time communication platforms.

- **Monitoring and Dashboard Applications**: Enables live visualizations and data updates for monitoring dashboards.

### Code Example: Establishing a WebSocket Connection

Here is the JavaScript code:

```javascript
// Create a new WebSocket
const socket = new WebSocket('ws://www.example.com/service');

// Define handlers for different events
socket.onopen = function(event) {
  console.log('WebSocket is open now');
};

socket.onmessage = function(event) {
  console.log('Message received:', event.data);
};

socket.onclose = function(event) {
  if (event.wasClean) {
    console.log('Connection closed cleanly');
  } else {
    console.error('Connection abruptly closed');
  }
  console.log('Close code:', event.code, 'Reason:', event.reason);
};

socket.onerror = function(error) {
  console.error('WebSocket error:', error);
};

// Sending data once the connection is open
socket.onopen = function(event) {
  socket.send('Hello from the client side!');
};
```
<br>

## 6. Explain the WebSocket frame format.

The **WebSocket frame** design is quite structured, typically consisting of at least an  **initial header**, sometimes an **extended header**, and then the **payload**.

### Initial Header

The initial header byte is the first byte of each WebSocket message and contains both the **opcode** (which specifies the type of message) and the **fin flag** signifying whether the message is the last in a sequence.

The format is:

```plaintext
FIN  RSV1 RSV2 RSV3  OPCODE
1 bit 1 bit 1 bit 1 bit  4 bits
```

- **FIN (1 bit):** Indicates whether this is the final fragment of a message (1) or if more frames will follow (0).
- **RSV1-3 (each 1 bit):** Reserved for extensions, which are responsible for setting these bits to zero.
- **OPCODE (4 bits):** Specifies the type of data in the payload. The available OPCODES are: 

| OPCODE | Message Type | Description |
|---|:---:|---|
| 0 | Continuation | The frame contains a part of a fragmented message |
| 1 | Text | The frame contains a UTF-8 encoded text message |
| 2 | Binary | The frame contains binary data |
| 8 | Close | The frame is requesting the connection to be closed |
| 9 | Ping | Used to confirm if the connection is still open |
| 10 | Pong | Used to reply to a Ping message |

### Possible Extended Header

For larger payloads, the initial header is followed by an **extended payload length** of either 16 bits (if the payload length is between 126 and 65535 bytes) or 64 bits (if it's more than 65535 bytes).

### Payload

The last section of the frame is the payload. Mathematically, the size of the payload depends on whether the **masked** boolean is set.

- **Masked (1 bit):** A Boolean value indicating whether the payload is masked.

    Masking prevents potential attacks where one browser might attempt to open a WebSocket to another service and inspect the traffic. If this value is set, the masking key appears after the extended payload length field.

- **Masking Key (0 or 4 bytes):** This key is present if the Masked bit is set. The key, which is included in the headers, is used to encode and decode the payload contents.

- **Payload Data (x+y octets):** If the Masked bit is set, this data is masked. If it's not set, the data is the original, unaltered payload.

###  Example Interpretation

Let's consider a binary message with a FIN flag set to 1, an opcode of 0x02, an unmasked payload length of 75 bytes, and a payload of 'A' repeated 75 times.

The initial header byte would be **10000010** (0x82), indicating a final message ('FIN' set to 1) of binary type ('opcode' 0x02).

This is followed by the exact payload specified. Given the absence of the masked bit, the payload of bytes 'A' (0x41) repeated 75 times would directly follow the header in this example.
<br>

## 7. How do WebSockets handle communication through proxies and firewalls?

**WebSockets** offer full-duplex communication and low-latency connections, but they face different challenges with **proxies** and **firewalls**.

### Proxies

- **HTTP/1.1 Issues**: Proxies initially designed for HTTP/1.1 may not recognize WebSocket upgrade requests. To rectify, WebSocket connections start as HTTP requests before being upgraded to WebSocket connections.

- **Addressing Restrictions**: Proxy servers might limit or reject WebSocket connections incompatible with standard HTTP ports.

- **Connection Consolidation**: To minimize overhead, proxies can combine multiple backend servers into one frontend server using connection multiplexing. With WebSockets, this can cause data mixing and interfere with the WebSocket handshake process, resulting in failed connections.

### Firewalls

- **Single Port Streamlining**: Certain firewalls may only allow traffic on standard HTTP ports (80) or HTTPS ports (443). While WebSockets can coexist with these ports, deployments on non-standard ports might face firewall restrictions.

- **Content Inspection**: Some firewalls examine and filter the content of data transmission. WebSockets use binary or text messages, making it challenging for these firewalls to perform content analysis and filtration effectively.

### Addressing Challenges

WebSockets adapt to these challenges primarily through their mechanism of starting as an HTTP or HTTPS connection before being upgraded, and by leveraging the favorable aspects of **Tunneling and Encapsulation**.

#### Tunneling and Encapsulation

- **Secure Transport**: WebSockets can be within a secure, encrypted SSL/TLS tunnel. This encapsulation hides the WebSocket-specific traffic within the SSL/TLS layer, allowing WebSockets to bypass firewall content inspections that only focus on unencrypted traffic.

- **Tunnel Relevance**: Firewalls designed to ensure secure, encrypted communications might allow tunneled traffic if it's robust ware-hygiene. This means the firewall can let WebSockets pass through, still benefiting from encryptions provided by the SSL/TLS tunnel.

- **Security Relevance**: Deploying WebSockets within an encrypted tunnel shields them from various security threats associated with direct internet traffic.

### Code Example: WebSockets Through Proxies and Firewalls

Here is the Python code:

```python
import websocket

def on_open(ws):
    print("Opened connection")

websocket.enableTrace(True)
ws = websocket.WebSocketApp("wss://www.example.com",
                            on_open=on_open,
                            on_message=on_message,
                            on_close=on_close)
ws.run_forever()
```
<br>

## 8. What are the security considerations when using WebSockets?

When using WebSockets, be mindful of various **security considerations** to protect both your server and client endpoints.

### Key Security Concerns

- **Cross-Origin Security**: Without proper configuration of the server, WebSockets can be vulnerable to Cross-Origin attacks.

- **Data Validation and Escaping**: Always ensure that data exchanged over WebSockets is validated and **properly escaped** to guard against client and server-side vulnerabilities such as Cross-Site Scripting (XSS).

- **DDoS Protection**: Due to the nature of WebSockets as a persistent connection, they can be exploited to carry out DDoS (Distributed Denial of Service) attacks. Appropriate measures need to be in place to mitigate this risk.

- **Secure Communication**: While WebSockets are inherently more secure than HTTP due to encryption, both client and server endpoints need to utilize secure communication.

- **Session Management**: Traditional stateless strategies like token-based authentication might not be sufficient with WebSockets since connections are persistent. As a result, session management in WebSockets is different and might require additional attention.

- **Rate Limiting and Access Controls**: Implement appropriate rate limiting and access controls to avoid abuse.

- **CORS Misconfigurations**: Misconfiguring Cross-Origin Resource Sharing (CORS) headers can lead to security vulnerabilities. Be meticulous in setting up these headers.

- **Payload Encryption**: Sensitive data transferred over WebSockets must be encrypted.

### Code Example: WebSockets and Cross-Origin Security

Here is the JavaScript code:

```javascript
// Server
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', function connection(ws, req) {
  const origin = req.headers.origin;
  if (isAllowedOrigin(origin)) {
    ws.send('You are granted websocket access');
  } else {
    ws.close();
  }
});

function isAllowedOrigin(origin) {
  // Insert your logic to validate allowed origins, e.g., a list of trusted origins
  return origin === 'http://example.com';
}
```

```javascript
// Client
const ws = new WebSocket('ws://localhost:8080');

// Handle responses from the server
ws.onmessage = function(event) {
  console.log(event.data);
};
```

In the server code, `isAllowedOrigin` is a server-side function to validate the Requests' Origin header to guard against Cross-Origin attacks. A similar mechanism must be in place for Origin validations in actual deployment configurations.
<br>

## 9. How would you detect and handle WebSocket connection loss?

Detecting and handling **WebSocket** disconnections involves monitoring the connection state, identifying the cause of disconnection, and implementing strategies for reconnection.

### Connection Monitoring

1. **Heartbeats**: Establish a periodic ping-pong mechanism between the server and the client to ensure the connection is alive.
2. **Server-Side Monitoring**: Use tools like active socket counters in **Node.js** or `WebSocketSession` in **SpringBoot** to track the number of open sockets.
3. **Client-Side Tracking**: Register event listeners for open, close, error, and other relevant socket events using `onopen` and `onclose` in **JavaScript** or equivalent methods in other frameworks.

### Identifying Disconnection Causes

1. **Explicit Closure**: The client or server can intentionally close the WebSocket.
2. **Network Interruption**: Unplanned disruptions or server issues can lead to disconnection.
3. **Inactivity Timeout**: The socket can close due to prolonged inactivity.

### Reconnection Strategies

1. **Persistent Reconnection**: Keep trying to reconnect with the server, either indefinitely or for a set number of attempts.
2. **Exponential Backoff**: Delays between reconnection attempts increase exponentially to avoid overwhelming the server.
3. **Scheduled Reconnection**: Use mechanisms like **cron** jobs or scheduled tasks to initiate reconnection attempts at specific intervals.

### Code Example: WebSocket Reconnection

Here is the JavaScript code:

```javascript
let ws;
let reconnectInterval = 2000;
let maxReconnectInterval = 30000;

function connect() {
  ws = new WebSocket('ws://localhost:8080');
  
  ws.onopen = () => {
    console.log('WebSocket connected!');
    reconnectInterval = 2000;
  };

  ws.onclose = (event) => {
    if (event.code === 1000) {
      console.log('WebSocket was closed intentionally.');
      return;
    }
    console.log('WebSocket disconnected. Attempting to reconnect.');
    setTimeout(connect, reconnectInterval);
    reconnectInterval = reconnectInterval < maxReconnectInterval ? reconnectInterval * 2 : maxReconnectInterval;
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    ws.close();
  };
}

connect();
```
<br>

## 10. Explain the role of ping/pong frames in WebSockets.

**Ping and Pong frames** in WebSockets are used to ensure a reliable, two-way real-time communication channel between the client (e.g., a web browser) and the server.

### Why Use Ping and Pong Frames in WebSockets?

Traditionally, network protocols like HTTP, which WebSockets build upon, follow a **request-response model**. However, many modern applications, such as chat services or online games, require a constant, bidirectional flow of data without waiting for a single continuous message to be completed.

1. **Connection Keep-Alive**: Pinging helps maintain an active connection. This mechanism is especially useful in networks with firewalls or proxies, which may terminate inactive connections.
  
2. **Timeout Detection**: Pong responses are expected within a certain time frame. If not received, the client or server can take corrective action, like closing the connection.

3. **Resource Conservation**: Using small, lightweight ping messages reduces overhead compared to regular data payloads.

### Ping Frame Structure

Ping frames consist of an **op-code** and an optional **application data payload**. The op-code for ping frames is hexadecimal value `0x09`.

Here is the hex representation of a ping frame:

\[0x89] [Length 0]

And here is the textual representation:

```
89 00 
```

In this case, the ping frame has a length of 0 and no application data.

### Pong Frame Structure

The pong frames indicate a **successful reception** of a ping frame. They also contain an **op-code** and an optional **application data payload**. The op-code for pong frames is hexadecimal value `0x0A`.

Here is the textual and hex representation of a pong frame:

```
8A 00 
```

As with the ping frame, it has a length of 0 and no application data.

### Security Considerations

Ping and Pong frames can play a role in detecting man-in-the-middle attacks, such as a server impersonating the client or vice versa.

### Code Example: Sending Pings with a  WebSockets Client

Here is the Python code:

```python
import asyncio
import websockets

async def example():
    uri = "wss://echo.websocket.org"
    async with websockets.connect(uri) as websocket:
        await websocket.ping()
        print("Ping sent successfully.")

asyncio.get_event_loop().run_until_complete(example())
```
<br>

## 11. How does WebSocket ensure ordered delivery of messages?

**WebSocket** employs a protocol that guarantees both the **reliability** and the **order of message delivery**, termed full-duplex communication.

### Full-Duplex Communication

WebSocket's status as a full-duplex communication technology, rather than only half-duplex like HTTP, allows it to send and receive data simultaneously without making use of multiple connections.

Consider a real-time chat application: a WebSocket connection lets users send and receive **instantaneous messages** concurrently.

### Abidance to The TCP Protocol

Underpinning WebSocket, the **TCP protocol** ensures data integrity and sequence preservation via a collection of mechanisms:

- **Segmentation**: TCP groups small chunks of data into segments, each bearing a sequence number.
- **Reassembly**: Segments are put back together at the receiving end, following their specified order.

This means that messages, even from a single client, are delivered in the same order in which they were sent.

### WebSocket Frames

Internally, WebSockets fragment messages into **frames** when necessary (e.g., for large payloads).

The `FIN` bit indicates if this frame is the final one or if the message continues with additional frames. Upon receipt, the combined frames are sequentially reconstructed to restore the original message.

### Code Example: WebSocket Frame Reassembly

Here is the JavaScript code:

```javascript
let combinedMessage = '';  // We'll concatenate our message fragments here

// Assume this callback receives a 'frame' object representing an incoming frame
socket.onmessage = function(frame) {
  // Combine frame's payload with any previous fragments
  combinedMessage += frame.payload;

  if (frame.FIN) {  // Check if the current frame is the last one
    // Perform further actions with the reconstructed, full message
    console.log('Received complete message:', combinedMessage);

    // Reset the storage for the next incoming message
    combinedMessage = '';
  }
};
```
<br>

## 12. Can WebSockets be used for broadcasting messages to multiple clients? If so, how?

Yes, **WebSockets** can facilitate **real-time bidirectional communication** and broadcast messages to multiple clients.

Each **WebSocket server can broadcast messages** by:

- **Queuing**: Storing messages for each client until the client is available.
- **Routing**: Sending targeted messages to specific clients or groups. This can be done by specialized frameworks built on top of WebSockets.

### The Broadcasting Mechanism

- **Unidirectional**: WebSockets, by design, operate as bidirectional channels for individual client-server pairs.
  
- **Client Loop**: To achieve multi-client message broadcasting, the server iterates through a list of active clients.

Let's look at a Python example.

### Code Example: Broadcasting with WebSockets

Here is the Python code:

```python
# server.py
import asyncio
import websockets

active_clients = set()

async def handle_client(websocket, path):
    active_clients.add(websocket)
    try:
        async for message in websocket:
            for client in active_clients:
                # Send the message to all active clients
                await client.send(message)
    finally:
        active_clients.remove(websocket)

start_server = websockets.serve(handle_client, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

```python
# client.py
import asyncio
import websockets

async def listen_for_messages():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            print(message)

asyncio.get_event_loop().run_until_complete(listen_for_messages())
```
<br>

## 13. What is the difference between WebSockets and Server-Sent Events (SSE)?

**WebSockets** and **Server-Sent Events** (SSE) both facilitate server-to-client communication in web applications, but they have distinct functionalities.

### Key Distinctions

#### WebSockets

- **Characteristics**: WebSockets are bidirectional, meaning that both the server and the client can send messages at any time.
- **Protocol**: WebSocket uses a full-duplex communication protocol.
- **API Support**: WebSockets are often implemented using JavaScript libraries and can be used in many modern web frameworks.
- **Use Case**: Real-time interactive applications that necessitate full-duplex communication, like chat or multiplayer games, benefit from WebSockets.

#### Server-Sent Events (SSE)

- **Characteristics**: SSE is unidirectional; the server is the primary sender of messages.
- **Protocol**: SSE uses the traditional HTTP protocol, arguably making it simpler to grasp.
- **API Support**: Supports a limited set of events, for instance, `open`, `message`, and `error`. Safari only started supporting SSE in 2020. Due to these restrictions, its use is often niche.
- **Use Case**: Ideal for scenarios where data needs to be sent from the server to the client in a standardized JSON format, such as financial data or news updates.
<br>

## 14. Explain how a WebSocket connection is closed.

A **WebSocket connection** can be explicitly closed by either the client or the server, or it can be closed unexpectedly due to network or server issues.

### Closure Scenarios

#### CloseEvent

When the connection is closed, a `CloseEvent` is created. It has two main attributes:

- `code`: A numeric code indicating the reason for closure.
- `reason`: A string reason for the closure.

Both the client and the server can initiate and interpret these **closure events**.

### Closing From the Client

1. **Regular Closure**: The client initiates the closure using the `close()` method on the WebSocket instance. 

2. **Aborted**: Situations such as network issues or an immediate call to `abort()` result in closing from the client.

### Closing From the Server

1. **Regular Closure**: The server decides to close the connection. It sends a close frame to the client, and upon successful transmission, the server's handshake is considered complete.

2. **Unsuccessful Handshake**: If the server deems the client's request invalid or not authorized, it closes the connection without sending a handshake accept frame.

### Automatic Closure

**WebSocket closure** can occur due to various factors, such as internet connectivity loss, server termination, or server-side timeout.

### Code Example: WebSocket Close Event

Here is the JavaScript code:

```javascript
// Establishing socket connection
const webSocket = new WebSocket('ws://www.example.com/socketserver');

// Adding event listener
webSocket.onclose = (event) => {
  console.log('Socket closed:', event);
};

//Functions to Manually Close the Connection
const closeModalButton = document.getElementById('closeModal');
closeModalButton.addEventListener('click', () => {
  webSocket.close(1000, 'User closed the modal');
});
```
<br>

## 15. What fallback mechanisms can be used if WebSockets are not supported by a browser or server?

When **WebSockets** are not supported, there are a range of alternatives and fallback mechanisms to maintain a real-time connection between a client and a server. Each mechanism has its pros and cons, catering to various specific needs and constraints. Therefore, the best approach is often to employ a **combination of methods** to achieve the desired level of functionality and support across different platforms.

### Polling

- **Mechanism**: The client regularly sends HTTP requests, polling for new data.
- **Pros**: Simple to implement, widespread browser support, compatible with most network setups.
- **Cons**: Increased latency due to regular requests, potential for data duplication or throttling.

### Long Polling

- **Mechanism**: A client request stays open until data is available, the server responds, and the connection closes. The client then opens a new request.
- **Pros**: Low latency, efficient with limited data transmissions.
- **Cons**: Complexity in managing long-lived requests, might not work well with certain server configurations.

### HTTP Streaming

- **Mechanism**: The server sends a continuous stream of data, keeping the connection open for as long as necessary.
- **Pros**: Efficient and low-latency; suitable for real-time updates.
- **Cons**: Can be challenging to implement across different server technologies.

### Server-Sent Events (SSE)

- **Mechanism**: The server delivers a unidirectional, long-lived stream of updates, primarily used for server-to-client communication.
- **Pros**: Simple to use, built for one-way data flow, automatic handling of reconnections.
- **Cons**: Not a bidirectional channel like WebSockets; may not be compatible with some browser versions.

### AJAX (Traditional and HTTP/2 Push)

- **Mechanism**: With traditional methods, the client initiates an HTTP request; with HTTP/2 Push, the server proactively sends data to the client.
- **Pros**: Ubiquitous support, especially HTTP/2 Push, can offer low latency.
- **Cons**: Traditional AJAX can be inefficient for real-time updates, while HTTP/2 Push may need server support and not be as widely compatible yet.

### WebHooks

- **Mechanism**: The server pushes data to an endpoint previously registered by the client.
- **Pros**: Efficient and scalable; doesn't require persistent client connections. Suitable for scenarios like notifications and callback-based systems.
- **Cons**: Setting up WebHooks requires coordination between the server and client endpoints.

### Encrypted and Secure Connections

- **Mechanism**: Securely encrypt communication between client and server to protect data privacy and integrity.
- **Pros**: Essential for safeguarding sensitive data.
- **Cons**: Might come with a slight overhead due to encryption and decryption processes.

### Reconnection Strategies

- **Mechanism**: Implement mechanisms like automatic reconnections or prompts for manual reconnection for clients facing connection issues.
- **Pros**: Ensures continuity of connection, enhanced user experience.
- **Cons**: Can introduce complexity especially in ensuring data integrity after reconnecting.

Here is the JavaScript code:

```javascript
// Short Polling
setInterval(() => {
  // Send an HTTP request to fetch updates
}, 1000);

// Long Polling
function longPoll() {
  // Send an HTTP request and keep the connection open
  // When the server responds, process the data and initiate another long poll
}
longPoll();

// HTTP Streaming (iframe example)
const iframe = document.createElement('iframe');
iframe.style.display = 'none';
iframe.src = 'http://example.com/streaming-endpoint';
document.body.appendChild(iframe);

// Server-Sent Events
const eventSource = new EventSource('http://example.com/sse-endpoint');
eventSource.onmessage = (event) => {
  // Process the received event data
};
eventSource.onerror = (error) => {
  // Handle any errors and reconnect if necessary
};
// When you're done, call eventSource.close() to terminate the connection.

// Traditional AJAX
setInterval(() => {
  fetch('http://example.com/updates-endpoint')
    .then((response) => response.json())
    .then((data) => {
      // Process the received data
    });
}, 1000);

// HTTP/2 Push (Client)
// Assuming server supports HTTP/2 Push and configured correctly
// In response headers: "Link: </updates-endpoint>; rel=preload"
fetch('/updates-endpoint').then((response) => {
  // Data will already be available in the response due to server push
});
```
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - Websocket](https://devinterview.io/questions/web-and-mobile-development/websocket-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

