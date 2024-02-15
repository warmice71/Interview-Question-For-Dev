# 32 Common Reactive Systems Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 32 answers here 👉 [Devinterview.io - Reactive Systems](https://devinterview.io/questions/web-and-mobile-development/reactive-systems-interview-questions)

<br>

## 1. What are the defining characteristics of a _Reactive System_ according to the _Reactive Manifesto_?

The **Reactive Manifesto** lays out key characteristics that define reactive systems.

###  Core Characteristics

1. **Responsiveness**: React to events and failures in a timely manner. Ensure application and network latency is minimal.
  
2. **Resilience**: Due to potential failures in distributed operations, reactive systems should remain responsive and consistent. They should recover and maintain operations in less than optimal times.

3. **Elasticity**: Able to manage varying volumes of traffic by scaling resources accordingly.

4. **Message-Driven**: Communication among system components is achieved through asynchronous message passing rather than more resource-intensive and less flexible mechanisms such as shared memory or method calls.

###  Additional Characteristics

5. **Back-Pressure**: Systems should apply back-pressure to prevent resource exhaustion by striving to keep the speed of message production and message processing in balance.

6. **Resource Efficiency**: They should use resources in a way that ensures the most efficient possible outcome. This means recognizing that **resources are limiting and shared** and managing them accordingly.    

###  Code Example: Back-Pressure in Akka Streams

Here is the Java code:

```java
final Source<Integer, NotUsed> source = Source.range(1, 100);
final Flow<Integer, Integer, NotUsed> flow = Flow.of(Integer.class).map(i -> i * 2);
final Sink<Integer, CompletionStage<Done>> sink = Sink.foreach(System.out::println);

source.via(flow).to(sink).withAttributes(Attributes.createLogLevels(Logging.DebugLevel(), Logging.InfoLevel()))
    .run(materializer);
```

Here is the Scala code:

```scala
val source = Source(1 to 100)
val flow = Flow[Int].map(_ * 2)
val sink = Sink.foreach[Int](println)

source.via(flow).to(sink).withAttributes(ActorAttributes.dispatcher("my-dispatcher")).run()
```
<br>

## 2. How is _back-pressure_ implemented in _Reactive Systems_ to manage data flow?

**Back-pressure** establishes control over data flow, ensuring asynchronous processes operate within defined resource limitations. It is a foundational concept in reactive systems, where components gracefully handle and communicate load imbalances.

### Key Components for Back-Pressure

- **Data Producer**: Generates items to be processed.
- **Data Consumer**: Receives and processes items.
- **Communication Channel**: Connects producer and consumer for data transfer.

### Communication Paradigms

- **Push Model**: - Producers actively deliver data.
- **Pull Model**: - Consumers dictate the pace of data arrival by requesting.

### Back-Pressure Strategies

- **Buffering**: Temporarily stores data to regulate the flow between producer and consumer. Beyond a certain threshold, producers are signaled to slow down or stop.

- **Dropping**: When buffer limits are reached, newer items replace older ones. This strategy is useful when older items become less relevant.

- **Throttling**: Dynamically controls how much data a producer can send. Common techniques include rate limiting and data batching.

- **Dynamic Scaling of Resources**: Solutions like auto-scaling cloud environments allow for quick adaptation to processing demands, ensuring resources match system requirements.

### Code Example: Throttling With Observable

Here is the C# code:

```csharp
var source = Observable.Interval(TimeSpan.FromMilliseconds(200));

var throttled = source
    .Throttle(TimeSpan.FromSeconds(1));

var subscription = throttled.Subscribe(
    Console.WriteLine,
    ex => Console.WriteLine("Error: " + ex.Message),
    () => Console.WriteLine("Completed"));

Console.ReadLine();
```

In this example, an observable sequence is created that emits an integer every 200 milliseconds. The **Throttle** operator is then applied to ensure items are emitted not more frequently than once per second.

### Trade-Offs and Considerations

- **Resource Utilization**: Buffering, while offering immediate handling of peaks, can lead to resource saturation if sustained for prolonged periods.
  
- **Latency**: Using buffers to handle peaks introduces a delay, which might not be acceptable in certain real-time scenarios.

- **Dropping Items**: Can simplify the system and reduce resource usage under heavy loads, but comes with the trade-off of potentially discarding important data.

- **Simple vs. More Complex Strategies**: Depending on the system and its requirements, simpler strategies might suffice. More intricate methods like adaptive processing might introduce unnecessary complexity.
<br>

## 3. Contrast _Elasticity_ with _Scalability_ with specific examples as they pertain to _Reactive Systems_.

**Scalability** is a system's ability to handle increasingly higher workloads. This can be horizontal, involving the addition of more nodes $N$, or vertical, meaning the increase of a node's capability.

**Elasticity**, on the other hand, is a dynamic, on-demand allocation and deallocation of resources to meet fluctuating workloads. Typically, this pertains to cloud-based solutions where resources are provisioned or de-provisioned in real-time based on demand to **minimize costs**.

In **Reactive** systems, both scalability and elasticity are essential. The system should be able to handle varying loads, commonly seen in modern applications, such as social media platforms.

### Examples in Reactive Systems

#### Horizontal Scalability

- **In Action**: A social media platform adds more database servers in response to an ever-increasing number of users.
- **Key Mechanism**: Load Balancers distribute incoming traffic across multiple application instances.

#### Vertical Scalability

- **In Action**: A system that processes financial transactions gets an increase in hardware resources to meet higher demand.
- **Key Mechanism**: The entire workload is handled by a single, more powerful machine rather than being distributed.

#### Elasticity in Virtual Machines (VMs)

- **In Action**: A cloud-based application, hosted on virtual machines, adjusts the number of VM instances based on current traffic patterns.
- **Key Mechanism**: Cloud management systems monitor resource usage and spawn or terminate VMs as necessary.

#### Elasticity in Containers

- **In Action**: A microservice-based system deployed with containers adjusts the number of running containers for each service based on traffic load.
- **Key Mechanism**: Container Orchestration tools like Kubernetes dynamically manage the number of containers to match the current load.
<br>

## 4. Describe a strategy a _Reactive System_ might use to maintain _responsiveness_ during a component failure.

**Reactive Systems** are designed to handle failures gracefully using techniques such as **Location Transparency**, Dynamic Message Routing and Load Balancing.

### Key Mechanisms for Failure Handling

- **Location Transparency**: Systems like Akka ensure that `actors`, which are key processing units in the system, are isolated from their actual physical location. The system abstracts the system location from the endpoint that communicates with the actor, allowing it to be resilient to location-specific failures.
  
- **Dynamic Message Routing**: In distributed systems, where individual components can fail, it's crucial that systems redirect traffic or messages from the failing component to a healthy one. Akka, for example, ensures this with its actor-based model. If an actor fails for some reason, its supervisor can detect this and redirect its messages to a backup or alternative actor.

- **Adaptive Load Balancing**: Systems like Akka can intelligently distribute incoming load or messages across all available actors. If a target actor becomes unresponsive, Akka can detect this and re-route messages to other available actors.

- **Eventual Consistency**: Rather than enforcing immediate and precise consistency, certain distributed architectures allow for eventual consistency. This means that, after a system recovers from a failure, it might take some time to completely reconcile the data or state across the system. Tools like CRDTs can assist in achieving this eventual consistency.

- **Graceful Degradation**: When failing under heavy loads, systems might choose to reduce their performance in a graceful manner, making sure not to collapse completely. This especially helps in maintaining stability during periods of excessive traffic or component failure.

- **Asynchronous Operations**: By performing tasks asynchronously and potentially non-blocking on event-driven systems, components can stay responsive even under heavy load or when waiting for operations to complete.

### Code Example: "Reactive System Failure Handling"

Here is the Java code:

```java
import akka.actor.*;
import akka.routing.*;
import akka.japi.pf.ReceiveBuilder;

public class FailureHandling extends AbstractActor {
    private final ActorRef router;

    public FailureHandling() {
        router = getContext().actorOf(FromConfig.getInstance().props(), "router");

        receive(ReceiveBuilder.
            matchAny(this::handle).
            build());
    }

    private void handle(Object message) {
        // Send messages to the router
        router.tell(message, getSender());
    }

    static public void main(String[] args) {
        // Instantiate the ActorSystem and the FailureHandling actor
        ActorSystem system = ActorSystem.create("FailureHandlingSystem");
        ActorRef failureHandling = system.actorOf(Props.create(FailureHandling.class), "failureHandling");

        // Use the `fireMissiles` method to simulate an incoming message to the `FailureHandling` actor
        failureHandling.tell("fireMissiles", ActorRef.noSender());
    }
}
```
<br>

## 5. How does _message-driven architecture_ contribute to the _resilience_ of _Reactive Systems_?

Message-driven architecture is fundamental to building **resilient** and **efficient** reactive systems. It facilitates clear separation between system components, allowing them to operate autonomously, in isolation, and at their own pace.

### Key Benefits of a Message-Driven Architecture

- **Loose Coupling**: Dependencies between system components are reduced, enabling them to evolve independently.

- **Asynchronous Communication**: Components can send and receive messages independently, enabling parallel processing and potentially allowing for higher throughput.

- **Back Pressure Handling**: The receiver controls the message flow, allowing systems to operate within their resource limits.

- **Location Transparency**: Message Decoupling enables systems to function consistently, regardless of the physical location of the components.

- **Resilience to Transient Failures**: The queuing of messages allows systems to mitigate temporary disruptions.

### Code Example: Message Passing

Here is the Scala code:

```scala
import akka.actor.{Actor, ActorSystem, Props}

case class Message(text: String)

class MyActor extends Actor {
  def receive = {
    case msg: Message => println(s"Received: ${msg.text}")
  }
}

val system = ActorSystem("MyActorSystem")
val myActor = system.actorOf(Props[MyActor], "MyActor")

myActor ! Message("Hello, Actor!")
```

In the code snippet, the `!` symbol is used for message passing. The actor system ensures delivery and processing of the message, showcasing key characteristics of a Message-Driven Architecture.
<br>

## 6. Identify a _resiliency strategy_ in _Reactive Systems_ and explain how it minimizes the impact of failures.

**Reactive Systems** incorporate numerous design strategies to manage failures gracefully. One such strategy is **Event Sourcing**, which helps systems regain their state after failure.

### Key Components

- **Commands**: Actions or intentions sent from users or systems to modify state.
- **Events**: Immutable facts representing state changes resultant from commands.
- **Event Store**: A durable data store that logs all published events.

### How Event Sourcing Works

1.  **Event Recording**: The system records every state change as an immutable event in the event store. This feature ensures data integrity while making it simpler to audit and debug state changes.

2.  **State Reconstruction**: To generate its current state, the system re-processes all historical events from the event store. This process is dynamic and transparent, enabling the system to adapt to changes in processing logic over time.

3.  **Fast Append-only Stores**: Most modern implementations use data stores optimized for high-throughput, low-latency append operations. This feature streamlines the process of recording events.

### Resiliency & Failover Mechanisms

-   **Operational Recovery**: After a failure, the system reverts to a consistent state by replaying logged events. This rollback mechanism safeguards against downtime caused by failed state modifications.

-   **Isolation & Integrity**: Each recorded event is independent and self-contained, assuring that a failure affecting one event doesn't compromise the entire system or the consistency of other events.

-   **Audit & Debug Capability**: Full event logs enable precise identification of the point in time where issues occurred. This aids in forensic analysis and can be valuable in complying with auditing standards.

-   **Ease of Temporal Reversion**: The append-only nature of event stores makes retractions or rectifications of unintended state changes possible. This feature is invaluable in mitigating the impact of human or application errors.
<br>

## 7. Why are _non-blocking I/O operations_ a necessity in _Reactive Programming_, and what problems do they solve?

**Reactive programming** necessitates **non-blocking I/O operations** to manage continuous streams of asynchronous data. Without them, systems can become slow and unresponsive.

### Background: Blocking I/O Operations

In traditional synchronous models, a process awaiting I/O is **blocked**, meaning it's inactive and consuming resources without performing any useful work. This leads to poor resource utilization and performance degradation, especially in environments like user interfaces and servers.

### The Problem

Consider the scenario where data is arriving at unpredictable, potentially high rates. If the system used blocking I/O, efficiency and performance could be marred by these issues:

- **Resource Wastage**: Threads are an example of costly resources that might be inappropriately occupied because of blocked I/O operations.
- **Queuing & Backpressure**: Mechanisms to manage data overflow could be missing or inefficient.
- **Responsiveness**: Without a non-blocking approach, a system can become unresponsive, struggling to process new requests amidst lingering I/O operations.

### Code Example: Blocking Write Operation

Here is the Java code:

```java
public void blockingWrite(String text) {
    try {
        // This operation blocks until the entire text is written to disk.
        fileWriter.write(text);
    } catch (IOException e) {
        e.printStackTrace();
    }
}
```

### The Solution: Non-Blocking I/O Operations

**Non-blocking I/O** operations free up resources during waiting times, enabling more efficient multitasking without the overheads linked to blocking operations.
**Asynchronous Data Handling** better aligns with the nature of real-world, unpredictable data sources, such as network requests and UI events.

### Code Example: Non-Blocking Write Operation

Here is the Java code:

```java
public void nonBlockingWrite(String text, Consumer<Boolean> onComplete) {
    fileWriter.write(text, result -> {
        if (result) {
            // Perform additional operations when the write finishes.
            onComplete.accept(true);
        } else {
            onComplete.accept(false);
        }
    });
}
```

### Combined Benefits of Non-Blocking I/O and Reactive Programming

- **Resource Efficiency**: Only engaged when necessary, conserving resources.
- **Responsiveness**: Uninterrupted, near-real-time processing of data.
- **Concurrency & Multitasking**: Optimized for multitasking without thread overutilization or contention.
- **Scalability**: Can manage high data volumes without overwhelming the system.
- **Predictability**: Well-defined control mechanisms like backpressure to manage data flows.
<br>

## 8. In what ways can _Domain-Driven Design (DDD)_ principles enhance the design of a _Reactive System_?

**Domain-Driven Design (DDD)**, with its focus on problem domains, makes a natural pairing with **Reactive Systems**, which emphasize responsiveness, elasticity, and message-driven interactions.

By combining the two methodologies, designers can build complex, reactive systems that better encapsulate problem domains.

### Key Concepts

#### Bounded Contexts

- **DDD**: Defines clear boundaries for the domain model, ensuring contextual integrity.
- **Reactive**: Supports isolation and decoupling, key prerequisites for bounded contexts.

#### Event-Driven Collaboration

- **DDD**: Leverages domain events to facilitate communication between elements in the domain model.
- **Reactive**: Emphasizes event-driven interactions, aligning with DDD's event sourcing and business event concepts.

#### Context Maps 

- **DDD**: Outlines the relationships between bounded contexts, guiding the system's modularity and integration.
- **Reactive**: Enforces loose coupling for better resilience and scalability, aligning with the need to integrate across contexts.

#### Platform-Agnostic Design

- **DDD**: Encourages the domain model to be independent of technical considerations, focusing instead on solving domain problems effectively.
- **Reactive**: Enhances adaptability and compatibility by avoiding technological entanglements, a principle known as reactive system autonomy.

### Example: e-Commerce System

Consider a DDD-based e-commerce system:

#### Bounded Contexts

The system may be divided into several bounded contexts, such as inventory management, order processing, and customer relations. Each context has its own domain objects and logic.

- **DDD**: Clear boundaries ensure that each domain context operates autonomously, promoting integrity and separation of concerns.
- **Reactive**: Provides an architecture that enables the independent scaling and resilience of these contexts, improving system responsiveness.

#### Event-Driven Collaboration

When an order is placed, the order processing context might publish a "new order" event, which triggers actions in the inventory and customer relations contexts.

- **DDD**: Emphasizes the use of domain events for cross-context communication, promoting loose coupling and making interactions asynchronous.
- **Reactive**: Aligns well with the need for asynchronous, message-based collaboration.

#### Context Maps

The various bounded contexts in the e-commerce system are connected according to specific integration patterns, as defined by the DDD context map.

- **DDD**: The context map establishes integration strategies, such as shared kernel or customer/supplier relationship, ensuring that different parts of the system integrate appropriately.
- **Reactive**: Loose coupling between contexts, guided by the context map, allows independent scaling and fault-tolerance.

#### Platform-Agnostic Design

The e-commerce domain is represented as closely to its real-world counterpart as possible, abstracting away any technological specifics.

- **DDD**: Puts the focus on the e-commerce domain, capturing its intricacies and addressing its challenges.
- **Reactive**: Ensures that the system adapts to varying loads and failure scenarios, with interaction patterns tuned to the specific domain's needs.

### Practical Tips

- **Clear Communication**: Use DDD's ubiquitous language and domain events to establish seamless communication between different parts of your system.
- **Cautious Integration**: Validate interactions across bounded contexts using reactive principles like back-pressure to prevent context overload.
- **Flexible Abstraction**: Combining DDD's focus on the problem domain and reactive principles empowers system architects to build adaptable, powerful systems tailored to real-world challenges.
<br>

## 9. Provide an example of a system that is reactive without fulfilling all the _Reactive Manifesto_ traits. Why does it qualify?

According to the **Reactive Manifesto**, to be considered a reactive system, it must:

- **Be Responsive**: Respond in a timely manner.
- **Be Resilient**: Stay responsive in the face of failure.
- **Be Message-Driven**: Embrace asynchronous, non-blocking communication.

A system might exhibit some, but not all, of these characteristics.

### Example: Asynchronous Email Service

This is **an example** that shows how a system can still be reactive without fully meeting all the reactive manifesto traits.

#### The System's Components 

1. **Service Entry**: Exposes an HTTP endpoint to accept email requests.
2. **Email Queue**: Buffers and processes email requests.
3. **Email Service**: Orchestrates and sends emails using external email providers like AWS SES or SendGrid.

### Reasons for Qualifying as Reactive

#### System Not Fully Meeting Resilience

- **Lack of Isolation**: If the email service fails, the entire system might appear unresponsive to incoming requests, especially if there's no built-in retry mechanism.

### Code Example: Asynchronous Email Service

Here is the Python code:

```python
from flask import Flask, request
from queue import Queue
import requests
import threading

app = Flask(__name__)
email_queue = Queue()

def send_email(email_data):
    # External Service Call (e.g., AWS SES)
    response = requests.post(
        '<EMAIL_PROVIDER_API_ENDPOINT>',
        data=email_data
    )
    return response

def email_worker():
    while True:
        email_data = email_queue.get()
        response = send_email(email_data)
        if response.status_code == 200:
            print("Email sent successfully!")
        else:
            print("Failed to send email. Response:", response.text)
        email_queue.task_done()

@app.route('/send-email', methods=['POST'])
def enqueue_email():
    email_data = request.json
    email_queue.put(email_data)
    return "Email queued for delivery.", 202

if __name__ == '__main__':
    email_thread = threading.Thread(target=email_worker)
    email_thread.daemon = True
    email_thread.start()
    app.run()
```
<br>

## 10. Describe how a _Reactive System_ would differently address a _transient failure_ versus a _network partition_.

A **reactive system** proactively and responsively manages issues to ensure optimal performance and user experience. Let's **explore** how it addresses **transient failures** and **network partitions**.

### Transient Failure

- **Cause**: Short-lived disruptions in communication or system components.

#### Reactive System Approach

- **Action**: Mitigates transient failures by accommodating quick and temporary disruptions.

### Network Partition

- **Cause**: Prolonged and potentially widespread communication breakdowns, making certain system elements unreachable.

#### Reactive System Approach

- **Action**: Uses techniques such as circuit breakers, timeouts, and retries to handle prolonged outages and ensure functional reliability.

### Code Example: HTTP Request with Timeouts

Here is the Python code:

```python
import requests

# Use a timeout to handle transient failures, ensuring fast response.
try:
    response = requests.get('http://example.com', timeout=5)
    response.raise_for_status()
except requests.Timeout:
    # Handle timeout
except requests.RequestException as ex:
    # Handle other connection-related issues
```
<br>

## 11. Discuss a particular _transport layer technology_ you would recommend for _Reactive Systems' asynchronous communication_ and why.

For effective and reliable asynchronous communication in **Reactive Systems**,  libraries like RxJava for Java or ReactiveX for various platforms are Go-To choices. They provide a suite of functionality to streamline concurrent and asynchronous programming.

### Key Components

- **Observables**: Representing data sources, they emit items, optionally transform these and complete or error out. They can model synchronous or asynchronous data sources.
- **Subscribers**: Where the emitted items are consumed. They handle the onNext, onError, and onComplete signals.
- **Schedulers**: Manage the execution context of observables, for example, by dictating whether an observable or observer should run on a computational thread or an I/O thread.

### Key Features

####  Back Pressure Mechanism

Support is offered to manage situations where producers outpace consumers, ensuring system stability.

```java
observable
    .observeOn(Schedulers.computation(), 10)  // Limit buffer to 10 items
    .subscribe(consumer);
```

#### Error Handling

Errors are appropriately routed and managed, ensuring system resilience.

```java
observable
    .onErrorResumeNext(errorHandler)
    .subscribe(consumer);
```

#### Hot and Cold Observables

- **Hot**: Data flows regardless of subscribers, making them useful for broadcast situations.
- **Cold**: New subscribers get their own stream of data, ensuring that computations are not repeated.

```java
// Cold observable
Observable<String> coldObservable = Observable.just("Data 1", "Data 2", "Data 3");

// Hot observable
Observable<String> hotObservable = Observable. from( aDataSource );
```

#### Scheduling and Threading

Developers can readily specify the thread on which certain operations within the observable take place, simplifying multi-threaded programming.

```java
observable
    .subscribeOn(Schedulers.io())
    .observeOn(AndroidSchedulers.mainThread())  // Android main thread
    .subscribe(consumer);
```

#### Delay and Timer Operators

Operations can be delayed or executed at predetermined intervals or times.

```java
observable
    .delay(3, TimeUnit.SECONDS)
    .timeout(10, TimeUnit.SECONDS)  // Time bound operations
    .subscribe(consumer);
```

#### Combining Observables

Data from multiple sources can be coordinated, a powerful feature for scenarios like parallel API calls.

```java
Observable.merge(observable1, observable2, observable3)
    .subscribe(consumer);
```

#### Caching

For hot observables especially, enables the caching of items, handy when multiple consumers need the same data set.

```java
Observable.just(1, 2, 3)
    .cache()
    .subscribe(consumer1);
```

#### Resource Management

RxJava provides constructs like disposables to facilitate the release of resources, ensuring efficient and safe operations.

```java
Disposable subscription = observable.subscribe(consumer);
// Later, to clean up
subscription.dispose();
```

### Levels of Complexity

- Rx Libraries: They are the most comprehensive and intricate tool. If you are dealing with a complex, resource-intensive use case, **Rx Libraries** offer an array of features to meet your demands, including imperative, functional, and loose-reactive programming paradigms. It's the most Otavanced. Orchestration of concurrent or sequential data streams, support for error handling. Unsubscribe or dispose of depending resources are some of the advantages of The Rx Libraries.

- **Loom**: When considering asynchronous coordination in Java, Loom, with its Project Loom emphasis on virtual threads and improved concurrency control, is a worthy successor to libraries like RxJava. Designed to simplify concurrency, particularly threading, in Java. Virtual threads, also known as lightweight threads, are a core characteristic of Loom.
<br>

## 12. Explain the role of _Reactive Systems_ in processing _continuous data streams_, providing industry use cases.

**Reactive Systems** are designed to handle **constantly evolving data streams** and are deployed across various industries for real-time data processing.

### Key Characteristics

- **Event-Driven**: Systems react dynamically to incoming stimuli, which typically consist of discrete events.
- **Non-Blocking** I/O: They don't wait for operations to complete, thus ensuring concurrent, efficient data processing.
- **Asynchronous**: Responsiveness is maintained through asynchronous handling of events. This is often achieved using queues or callback patterns.

### Industry Use-Cases

#### Web Development

- **Interactive Web Applications**: Tailored for user input, real-time updates, and immediate feedback.

#### FinTech

- **Algorithmic Trading**: Used for analyzing market data and executing trades in fractions of a second.
- **Real-Time Fraud Detection**: Essential for identifying potential fraudulent activities as they occur.

#### Telecommunications

- **Call Centers**: To manage incoming calls and provide prompt service.

#### IoT & Smart Devices

- **Connected Devices**: These involve applications like smart home technologies where devices need to interact and make decisions based on sensor inputs.
- **Real-Time Analytics**: For tracking and analyzing data streams from connected devices, a typical use case is smart energy management systems.

#### Gaming

- **Real-Time Multiplayer Games**: To ensure a smooth gaming experience and synchronized interactions among players.

#### Healthcare

- **Remote Patient Monitoring**: For continuously tracking patient data like heart rate or blood sugar levels in real time.

#### E-Commerce

- **Inventory Management**: To keep track of real-time stock and manage orders efficiently.
- **Real-Time Bidding**: Used in online auctions for ad placements to manage bids in real time.

#### Data Analytics

- **Stream Processing**: For real-time data ingestion and analytics, such as monitoring social media for trending topics or analyzing server logs for anomalies, among others.

### Example: A Real-Time Chat Application

- **User Interactions**: Each message sent or received is an event prompting a specific action, like updating the chat interface in real time.
- **Data Flow**: The chat messages are part of a continuous stream of data that needs to be displayed to the users in real time.
- **Client-Server Coordination**: The client and server communicate asynchronously and process events non-blockingly through WebSockets or similar technologies.
- **Responsiveness**: Immediate updates are crucial for a seamless chat experience.
<br>

## 13. What is _Event Sourcing_, and how does it benefit _Reactive Systems_?

**Event Sourcing**, as a data storage pattern, captures changes to application state as a sequence of time-stamped events. Each event, often represented via Domain-Driven Design principles, contains discrete state mutations.

### Advantages of Event Sourcing for Reactive Systems

- **Temporal Focus**: Maintaining a chronological log establishes a clear historical context, especially beneficial for audit trails, reproducibility, and regulatory compliance.
- **Asynchronous Integrity**: Decoupling state mutations improves horizontal scalability and that of compliance-critical operations.
- **Tracing and Debugging**: Events are self-contained, enabling deeper insights into fault contexts and streamlined debugging.

### How Does it Work?

- **Write Operations**: Instead of directly modifying state, applications persist events to an append-only log, often a distributed message broker.
- **Read Operations**: The current application state is computed by 'replaying' events. The status is the logical consequence of the sequence of events.

### Code Example: Event Sourcing

Here is the Python code:

```python
class BankAccount:
    def __init__(self):
        self.balance = 0
        self.events = []

    def withdraw(self, amount):
        if self.balance >= amount:
            self.balance -= amount
            self.events.append({"type": "withdrawal", "amount": amount})
            return True
        return False

    def deposit(self, amount):
        self.balance += amount
        self.events.append({"type": "deposit", "amount": amount})

    def replay_events(self):
        for event in self.events:
            if event["type"] == "withdrawal":
                self.balance += event["amount"]
            elif event["type"] == "deposit":
                self.balance -= event["amount"]

    def get_balance(self):
        self.replay_events()
        return self.balance
```
<br>

## 14. Differentiate between _hot_ and _cold reactive streams_ with examples of use cases.

**Hot** and **Cold** streams refer to production strategies in **reactive programming** that affect data emission and subscription behavior.

### Key Distinctions

- **Cold Streams**: Data gets produced upon subscription.
- **Hot Streams**: Data is produced independent of subscriptions.

### Characteristics

#### Cold Streams

- **One-to-One**: Each subscriber generates fresh data.
- **On Demand**: Producers emit data only when someone listens.

#### Hot Streams

- **Many-to-Many**: Multiple subscribers share the same data.
- **Broadcasting**: Producers emit data regardless of subscribers.

### Common Use Cases

#### Cold Streams

- **Pull-Oriented Data**: Best for scenarios where subscribers request data.
- **Stateless Sources**: Such as static collections or HTTP requests to REST endpoints.

#### Hot Streams

- **Push-Oriented Data**: Ideal for real-time data and event-driven interactions.
- **Stateful Sources**: Applications requiring shared context, e.g., sensors in a building.

### Practical Example: Weather Updates

- **Cold Stream**: Requesting a weather report that's generated in the moment.
  
  This mirrors an on-demand request.

- **Hot Stream**: Receiving continuous weather updates, maintaining an ongoing subscription.

  This reflects a live, always-on link.
<br>

## 15. Compare _synchronous request-response_ communication with _reactive message-driven_ communication in terms of _scalability_.

Let's discuss the scatter-gather pattern

The **Scatter-Gather** pattern is a common messaging pattern used in reactive systems to improve system efficiency by parallelizing workloads through "scatter" and then consolidating the results through "gather."

This is similar to the divide and conquer approach, where a single, larger task is split into multiple smaller sub-tasks, then processed either concurrently or distributed.

### Key Components

1. **Request Distributor**: The initial point that receives the client request and is responsible for dividing or "scattering" the request further.
  
2. **Scatter**: The distribution of the request over potentially numerous processing nodes. This phase can happen sequentially or be parallelized, depending on the context.

3. **Individual Processors**: Optional step where independent sub-tasks are processed. These can be executed concurrently.

4. **Gather**: The re-joining or aggregating of responses from individual processors to form a cohesive overall response.

5. **Response Publisher**: The component responsible for issuing the final, consolidated response back to the system or client.

### Benefits

- **Parallelism**: Individual tasks can be processed concurrently, reducing system response times.
- **Distributed Computing**: Well-suited for cloud environments and scenarios that benefit from distributed processing.
- **Fault Tolerance**: The design allows for graceful handling of component failures.

### Code Example: Scatter-Gather With Web Workers


Here is a simplified code example:

#### index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scatter-Gather</title>
</head>
<body>
    <button id="scatterGatherBtn">Run Scatter-Gather</button>
    <script src="scatterGather.js"></script>
</body>
</html>
```

#### scatterGather.js

```javascript
document.getElementById('scatterGatherBtn').addEventListener('click', async () => {
    const dataToProcess = [5, 10, 15, 20, 25];
    const results = await scatterGather(dataToProcess, processItem);
    console.log("Results:", results);
});

async function scatterGather(data, processorFunction) {
    const numWorkers = 2;
    const chunkSize = Math.ceil(data.length / numWorkers);
    const workers = Array.from({length: numWorkers}, (_, i) => {
        const start = i * chunkSize;
        const end = start + chunkSize;
        return new Worker('worker.js'), [data.slice(start, end)];
    });

    const results = await Promise.all(workers.map(worker => {
        return new Promise((resolve, reject) => {
            worker.onmessage = (event) => resolve(event.data);
            worker.onmessageerror = (error) => reject(error);
        });
    }));

    return results.flatMap(res => res);
}

async function processItem(item) {
    // Simulated processing delay
    await delay(1000);
    return item * 2;
}

function delay(ms) {
    return new Promise(resolve => {
        setTimeout(resolve, ms);
    });
}
```

#### worker.js

```javascript
self.importScripts('mathUtils.js');

self.onmessage = async (event) => {
    const data = event.data;
    const processedData = data.map(processItem);
    self.postMessage(processedData);
};
```

#### mathUtils.js

```javascript
function processItem(item) {
    return item ** 2;
}

function delay(ms) {
    return new Promise(resolve => {
        setTimeout(resolve, ms);
    });
}
```

This example demonstrates a simple scatter-gather pattern using Web Workers in a browser environment. When the button in `index.html` is clicked, the `scatterGather` function divides the array of numbers ([5, 10, 15, 20, 25]) into two chunks and assigns each chunk to a Web Worker for processing. Once both workers complete their tasks, the results are gathered and logged to the console.
<br>



#### Explore all 32 answers here 👉 [Devinterview.io - Reactive Systems](https://devinterview.io/questions/web-and-mobile-development/reactive-systems-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

