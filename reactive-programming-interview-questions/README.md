# Top 100 Reactive Programming Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - Reactive Programming](https://devinterview.io/questions/web-and-mobile-development/reactive-programming-interview-questions)

<br>

## 1. What is _Reactive Programming_, and how does it differ from _Procedural Programming_?

**Reactive Programming** and **Procedural Programming** are fundamentally distinct paradigms, each suited for particular development contexts.

### Core Principles

- **Reactive Programming** focuses on **reactive data composition**, employing data flows and change propagation.
  - Example: UI event handling, streaming, real-time applications.

- **Procedural Programming** emphasizes **sequential task execution** through explicit actions and control flows.
  - Example: User input processing, algorithms.

### Key Components and Concepts

#### Reactive Programming

- **Data Stream**: An ongoing sequence of data that allows concurrent data handling.
- **Observer**: Any entity that subscribes to a stream and reacts to the data.
- **Subscriber**: A reader that is attached to a data stream.

#### Procedural Programming

- **Variables**: These are storage units for data values and have a single, non-concurrent phase.

**Reactive Programming** represents an ongoing data flow with a stream that can be subscribed to by multiple observers. On the other hand, **Procedural Programming** presents data as a single value stored in a variable, which gets executed, taking actions as inputs and producing outputs.

### Code Example: Reactive Programming

Here is the C# code:

```csharp
using System;

public class Program
{
    public static void Main()
    {
        var numbers = new int[] { 1, 2, 3, 4, 5 };

        IObservable<int> numberObservable = numbers.ToObservable();

        using (numberObservable.Subscribe(Console.WriteLine))
        {
            Console.WriteLine("Press any key to exit.");
            Console.ReadKey();
        }
    }
}
```
<br>

## 2. Explain the concept of _data streams_ in _Reactive Programming_.

**Data streams** form the foundational concept of Reactive Programming, facilitating the continuous flow of data and enabling responsive, event-driven app behavior. This paradigm embodies the **Observer Design Pattern**.

### Key Components

1. **Observable**: This represents the data source. When its state changes (or when it produces new data), it pushes the changes to its Observers.

2. **Observer**: Observers subscribe to Observables and receive notifications for any state changes or new data.

3. **Subscription**: This establishes the relationship between the Observable and Observer. Subscription can be one-to-one or one-to-many.

4. **Operators**: Often referred to as transformation functions, these allow data from an Observable to be modified or adapted before reaching the Observer. 

5. **Schedulers**: These tools help manage the time and order of operations in scenarios such as background work and UI updates.

6. **Subjects**: They combine the roles of an Observable and an Observer. These can be both data sources and data consumers. 

### The Data Flow Process

- **Emission**: Data is produced within an Observable and sent to its Observers.

- **Filtering**: Operators can screen incoming data, forwarding only what meets specific criteria.

- **Transformation**: Data is modified—by mapping it, for example—before being relayed to the Observer.

- **Notification**: Upon receiving new data, Observers are informed.

###  Fundamental Characteristics of Streams

- **Continuous**: Data flow persists, allowing for real-time interactions and responses.

- **Asynchronous**: Events aren't guaranteed to occur in a particular sequence, accommodating non-blocking operations.

- **One-directional**: Data moves from the Observable to its subscribers, ensuring a unidirectional flow.

### Stream Categorization

- **Unicast Streams**: These are one-to-one, ensuring that each Observer has an exclusive Observable-source connection.

- **Broadcast Streams**: These are one-to-many, permitting multiple Observers to subscribe to a single Observable. Each Observer receives the full data set, which can be problematic if data privacy is a concern.

### Observable Sequences

- **Hot Observable**: These sequences emit data regardless of Observer presence. If a new Observer subscribes, it starts receiving data from the point of subscription.

- **Cold Observable**: Here, data emission only begins upon subscription. Any new Observer would receive the data from the beginning. 

### Backpressure

**Backpressure** mechanisms regulate data flow to handle potential data overflow or bottlenecks due to disparities in data processing speeds.

For instance, in RxJava, the `Observable` and `Flowable` interfaces differ in that the latter incorporates backpressure support. With `Flowable`, you can employ backpressure strategy configurability to control the emission pace of the Observable relative to the consumption velocity of the Subscriber.

### Practical Application

Whether it's handling asynchronous API calls or managing user inputs, data streams provide a robust and flexible foundation for many everyday programming tasks.

Developers can employ a range of operators, such as `map`, `filter`, `debounce`, and `throttle`, to transform and manipulate data based on specific use-case requirements.

The wide adoption of Reactive Extensions (Rx) libraries, like RxJava for Android or RxJS for Web, underscores data streams' utility in modern software design.
<br>

## 3. What is the _Observer pattern_, and how is it fundamental to _Reactive Programming_?

The **Observer Pattern** forms the backbone of **Reactive Programming**, enabling applications to react and update when data changes. This design pattern facilitates **loose coupling** between observing components and a centralized subject or observable data source.

### Key Components

- **Subject**: This is the source of data or events. Observers "subscribe" to the Subject to receive notifications of changes.
- **Observer**: Receives notifications when the Subject's state changes.

### Observable Data

In a reactive setup, the Subject is responsible for "publishing" changes, and Observers are set up to "subscribe" to those changes. This rules out explicit, direct referencing of data sources and emphasizes a **datastream model**.

### Code Example: Observer Pattern
Here is Java code:

```java
import java.util.ArrayList;
import java.util.List;

interface Observer {
    void update();
}

class Subject {
    private List<Observer> observers = new ArrayList<>();
    private int state;

    public int getState() {
        return state;
    }

    public void setState(int state) {
        this.state = state;
        notifyAllObservers();
    }

    public void attach(Observer observer) {
        observers.add(observer);
    }

    public void notifyAllObservers() {
        for (Observer observer : observers) {
            observer.update();
        }
    }
}

class ConcreteObserver implements Observer {
    private String name;
    private Subject subject;

    public ConcreteObserver(String name, Subject subject) {
        this.name = name;
        this.subject = subject;
    }

    @Override
    public void update() {
        System.out.println("Observer " + name + " updated. New state: " + subject.getState());
    }
}

public class Main {
    public static void main(String[] args) {
        Subject subject = new Subject();
        ConcreteObserver observer1 = new ConcreteObserver("One", subject);
        ConcreteObserver observer2 = new ConcreteObserver("Two", subject);

        subject.attach(observer1);
        subject.attach(observer2);

        subject.setState(5);
    }
}
```

In this example, the `Subject` maintains a list of subscribing `Observers` and notifies them when its state changes.
<br>

## 4. Describe the role of _Observables_ and _Observers_ in _Reactive Programming_.

**Observers** are the consumers of data, while **observables** are the source or producer of data.

### Key Concepts

**Observable**: This is the data source. It emits data or signals, which can be any data types, including custom events. Observers subscribe to observables to "listen" for these emissions.

**Observer**: This is the "listener" or subscriber. It's the one that gets notified when the Observable emits data.

**Subscription**: A link or connection between the Observable and the Observer. When 'subscribe' is called, a subscription is created, and the Observer is "subscribed" to the Observable to receive notifications.

**Operators**: These are the "middlemen" between the Observable and the Observer. They allow you to transform, filter, combine, or handle the data flow emitted by the Observable before it reaches the Observer.

### Communication Flow

The communication flow in a typical observable-observer architecture is **one-way**.

1. The `Observable` emits data or a signal.
2. The `Observer`, which is subscribed to this `Observable`, receives the data or the signal and acts upon it accordingly.

### Code Example: Observable and Observer

Here is a code example in **Python** using the `rx` library.

```python
from rx import Observable, Observer

# Define the Observable
source = Observable.from_(["one", "two", "three"])

# Define the Observer
class MyObserver(Observer):
    def on_next(self, value):
        print(f"Received: {value}")
    def on_completed(self):
        print("Completed!")
    def on_error(self, error):
        print(f"Error Occurred: {error}")

# Subscribe the Observer to the Observable
source.subscribe(MyObserver())
```

In this example:

- `source` is the `Observable` that emits values.
- `MyObserver()` is the `Observer` that receives the emitted values.

By calling `source.subscribe(MyObserver())`, we establish the link, and the **Observer**, in this case, `MyObserver`, is "subscribed" to the **Observable** `source`.
<br>

## 5. How do you create an _Observable stream_?

**Observables** are the core building blocks of reactive programming, representing uni- or multi-directional data flows.

### Basic Structure

**Observable** streams have a definitive structure, always starting with a source followed by **subscribers** who consume the emitted data.

### Source: Factory Methods

Observable streams often begin with single or multiple **data sources**, depending on the chosen factory method. These sources can be anything from a single value to a sequence generated over time or a continuous input feed.

#### Purity and Immutability of Data Streams

In some contexts, observables ensure both purity (functions produce the same output for the same inputs) and immutability.

#### Special Observable Sources

- **Empty**: An Observable that emits no items but terminates normally (`Observable.empty`).
- **Never**: An Observable that never emits any items and never terminates (`Observabe.never`).
- **Error**: An Observable that emits no items and terminates with an error (`Observable.error(exception)`).

### Transformation Operators

Observable streams can be modified and adapted to meet the specific needs of subscribers. Operators offer a variety of transformations, from simple mappings to filtering, combining, and more.

- **Map**: Applies a function to each item emitted by the source observable, then emits the result.
- **Filter**: Applies a predicate to each item emitted by the source observable, only emitting items that satisfy the condition.
- **Take**: Takes a specified number of values from the source.

### Completion

Observables have a start and an end, signaled by the `complete` or `error` method. Once a stream is complete or an error is thrown, nothing more can be emitted or consumed.

**Termination** can happen through either normal or exceptional code paths. The `complete` method signifies a regular termination, while an `error` indicates an exceptional termination caused by an error message or an exception.

### Subscription: Giving Birth to Observers

After the subscriber is connected, an `unsubscribe` method can be called to stop the flow. Unsubscribing is especially relevant when dealing with finite sources or when a subscriber no longer needs data from a specific observable.

The `unsubscribe` method represents an escape hatch; it's a way for subscribers to say, "Okay, I've seen enough. Stop sending data.''

### Code Examples

Here is the Java code:

```java
import io.reactivex.Observable;
import io.reactivex.Observer;
import io.reactivex.disposables.Disposable;

public class BasicRxJavaExample {
    public static void main(String[] args) {
        Observable<Integer> source = Observable.just(1, 2, 3, 4, 5);
        source.subscribe(new Observer<Integer>() {
            @Override
            public void onSubscribe(Disposable d) {
                System.out.println("Subscribed");
            }
            @Override
            public void onNext(Integer integer) {
                System.out.println("Got: " + integer);
            }
            @Override
            public void onError(Throwable e) {
                System.out.println("Oops: " + e.getMessage());
            }
            @Override
            public void onComplete() {
                System.out.println("Received all!");
            }
        });
    }
}
```
<br>

## 6. What is _backpressure_ in the context of _Reactive Programming_?

**Backpressure** acts as a flow control mechanism in systems that process **asynchronous** data streams. Traditionally, systems without backpressure could experience issues like **buffer overflows** or data loss when one component produces data at a faster rate than another can consume.

By contrast, backpressure-aware systems dynamically adjust the rate at which data is emitted. This helps prevent issues associated with data being pushed to consumers faster than they can handle.

### Key Concepts

- **Dataflow Dilemma**: When a data source emits faster than the recipient can handle, backpressure offers mechanisms to adapt the flow. This can involve strategies like data buffering, data dropping, or rate limiting to ensure a balanced throughput.

- **Pull Model**: Backpressure often leverages a "pull" model, allowing consumers to request a certain amount of data when they are ready to process it. This shift from the traditional "push" model gives recipients control over their data intake, avoiding potential data congestion.

### Core Terminology

- **Publisher (or Source)**: The entity producing data.
- **Subscriber (or Consumer)**: The entity consuming data.
- **Subscription**: The contract between a publisher and a subscriber, providing a mechanism for the subscriber to manage its data flow.

### Backpressure Strategies

- **Buffering**: Optical for short-term storage of incoming data.
- **Dropping**: Ignores additional data if the buffer becomes full.
- **Stopping**: Halts the data flow if the recipient cannot keep pace.

### Code Example for Backpressure

Below is the **Kotlin** code.

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.collect
import kotlin.system.measureTimeMillis

suspend fun main() = coroutineScope {
    val startTime = System.currentTimeMillis()
    var count = 0
    val job = launch {
        callbackFlow {
            while(true) {
                count++
                try {
                    // Simulate data source delay
                    delay(100)
                    send(count)  // Potential backpressure here
                } catch (e: Exception) {
                    close(e)
                }
            }
            awaitClose { close() }
        }.collect {
            println("Received $it")
            delay(400)  // Simulate lagging consumer
        }
    }

    // Wait for a while and cancel, observing backpressure effects
    delay(1500)
    job.cancel()
    job.join()
    println("Job completed in ${System.currentTimeMillis() - startTime}ms")
}
```

In Kotlin coroutine, you can invoke `send` for `callbackFlow` in a non-optimized manner, which could lead to **unbounded backpressure**. It's important to use backpressure strategies like `BUFFER`, `DROP`, `LATEST`, alongside the `.onStart` extension function to start the flow with the correct configuration for the producer.

Here is the updated snippet:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.*
import kotlin.system.measureTimeMillis

suspend fun main() = coroutineScope {
    val startTime = System.currentTimeMillis()
    var count = 0
    val job = launch {
        callbackFlow {
            while(true) {
                count++
                try {
                    delay(100)
                    sendBlocking(count)  // Sending data with backpressure
                } catch (e: Exception) {
                    close(e)
                }
            }
            awaitClose { close() }
        }.onStart {  // Configure backpressure strategy
            println("Setting buffer strategy")
            buffer(1)  // Buffer at most 1
        }.collect {
            println("Received $it")
            delay(400)
        }
    }

    delay(1500)
    job.cancel()
    job.join()
    println("Job completed in ${System.currentTimeMillis() - startTime}ms")
}
```
<br>

## 7. Explain the difference between _cold_ and _hot Observables_.

**Observables** are the core building blocks in **Reactive Programming**, representing data streams. They come in two primary categories: **Cold Observables** and **Hot Observables**, each with specific behaviors and use cases.

### Cold Observables: Distinct Streams for Each Subscriber

**Cold Observables** have independent data sequences for each subscriber. When a new observer subscribes, the observable begins its data emission from the start, creating a new, dedicated stream.

Examples of Cold Observables include user input, HTTP requests, and static datasets.

### Hot Observables: Shares a Single Data Stream

In contrast, **Hot Observables** maintain a single data sequence that is common to all subscribers. When a new observer subscribes, it joins an ongoing data flow, potentially missing data emission events that occurred before its subscription.

Examples of Hot Observables include stock ticker prices, sensor inputs, and communicated events.

### Key Distinctions

| Criteria | Hot Observables | Cold Observables |
| --- | --- | --- |
| Data Sharing | Shares a single stream among all subscribers | Each subscriber has its independent stream |
| Subscription Timing | Misses or receives data based on when the subscription occurs | Receives all data, even if it subscribes later |
| Lifecycle | Operates independently of subscriptions | Starts data generation only when it's being observed or subscribed to |
| Synchronous vs. Asynchronous | Can generate and emit data even without subscribers | Begins data emission upon subscription, often resulting in synchronous data transmission |

### Practical Scenarios

- **User Input**: Cold Observable. Every subscription starts a new input stream.
- **Mouse Movements**: Hot Observable. New subscribers might not see prior mouse positions.
- **Data Polling**: 
   -  Cold: Regular updates independent of subscriptions.
   -  Hot: A shared stream ensures all subscribers receive the same updates.
<br>

## 8. What is the role of the _Subscription_ in _Reactive Programming_?

Abstracting away the specifics of how the data are received or generated is one of the lessons of reactive programming. The **Subscription** type is one such abstraction and serves an essential role in this process.

### Core Functions

A `Subscription` typically provides two main methods:

- **Request**: This informs the data source about the number of elements the consumer is ready to receive.
- **Cancel**: This stops the data flow, releasing any resources, such as file handlers or network connections.

### General Concept

The `Subscription` interface acts as an agreement between the data source and the data consumer. It enables data transmission while considering flow control, backpressure, and resource management.

### Backpressure Management

Implementations of the `Publisher` interface in reactive streams assess the subscriber's readiness to handle incoming data, taking into account the current state of the data flow. This mechanism, known as **backpressure**, aims to prevent data overload by instructing the data source to adapt its rate of data transmission accordingly.

The `request` method of the `Subscription` interface is the primary channel by which a subscriber communicates its current capacity to the data source, regulating backpressure.

### Resource Management

Certain data sources, like files, I/O streams, or databases, may require specific resources. The `Subscription` interface provides the means to release these resources when data transmission is no longer necessary.

Upon invoking the `cancel` method, the data source can take appropriate action, such as closing a file or terminating network communication.

### Example: Handling Multiple Subscriptions

Certain types of data sources are capable of furnishing multiple subscriptions simultaneously. Such sources can manage several `Subscription` instances, transmitting data to multiple consumers as per the terms of each subscription.

Let's consider an example of a `ConnectableObservable` in RxJava that, after its `connect` method is invoked, starts relaying data to all the subscriptions it receives.

```java
ConnectableObservable<Integer> connectable = Observable.just(1, 2, 3).publish();

connectable.subscribe(System.out::println);
connectable.subscribe(System.out::println);

connectable.connect();  // Data is emitted to both subscribers.

// ...
// Later, calling these methods would initiate new subscriptions that begin receiving data from the connectable:
// connectable.subscribe(System.out::println);
// connectable.subscribe(System.out::println);
```
<br>

## 9. How do you unsubscribe from a stream to prevent _memory leaks_?

**Unsubscribing** from a stream is key to preventing memory leaks. The process involves cleaning up resources and stopping further emissions.

### Unsubscribe Mechanism

- Core Features: Many reactive libraries, such as RxJava and Angular's RxJS, offer built-in methods for unsubscribing.

- Manual Unsubscription: In some cases, manual cleanup is necessary when automatic unsubscription is not supported or desirable.

### Code Example: Basic Manual Unsubscription with RxJava

Here is the Java code:

```java
public class SubscriptionExample {

    public static void main(String[] args) {
        Observable<Long> observable = Observable.interval(1, TimeUnit.SECONDS);
        Disposable subscription = observable.subscribe(System.out::println);

        // Unsubscribe after 5 seconds
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                subscription.dispose();
            }
        }, 5000L);
    }
}
```

### Code Example: Unsubscribe Mechanism with Angular's RxJS

Here is the TypeScript code:

```typescript
import { Observable, Subscription } from 'rxjs';

export class UnsubscribeExample {

    private subscription: Subscription;

    constructor() {
        this.subscribeToStream();
    }

    private subscribeToStream(): void {
        const observable$: Observable<string> = this.getObservable();

        this.subscription = observable$.subscribe(
            (data: string) => console.log(data),
            (error: any) => console.error(error),
            () => console.log('Stream completed')
        );
    }

    private getObservable(): Observable<string> {
        return new Observable((observer) => {
            observer.next('Hello');
            observer.next('World');
            observer.complete(); // Stream completion
        });
    }

    public stopStream(): void {
        if (this.subscription && !this.subscription.closed) {
            this.subscription.unsubscribe();
        }
    }
}
```

### Best Practice: Lifecycle-Related Unsubscription

- Web Components: It's ideal to synchronize unsubscription of component-bound observables with the component's lifecycle.

- Angular: The `AsyncPipe` and `takeUntil` operator combination can streamline unsubscription based on component lifecycle events.
<br>

## 10. What are _operators_ in _Reactive Programming_, and what are they used for?

**Operators** in **Reactive Programming** serve to create, transform, filter, or combine different data streams, contributing to the framework's flexibility, scalability, and variety of use-cases.

They fall into several categories:

1. **Creating Operators**
    - Emit: `just`, `from`, `range`, `defer`
    - Generate values: `interval`, `timer`

2. **Transforming Operators**
    - Modify: `delay`, `delaySubscription`, `timeInterval`
    - Map: `map`, `flatMap`, `switchMap`, `concatMap`

3. **Filtering Operators**
    - Limit emitted items: `take`, `takeLast`, `takeWhile`
    - Emit distinct items: `distinctUntilChanged`

4. **Combining Operators**
    - Merge streams: `merge`, `zip`
    - Join streams conditionally: `combineLatest`, `withLatestFrom`
      
5. **Utility Operators**
    - Action triggers: `doOnNext`, `doOnComplete`, `doOnError`
    - Side effects: `sideEffect`, `startWith`, `materialize`, `dematerialize`
    - Thread handling: `subscribeOn`, `observeOn`
    - Error Handling: `onErrorReturn`, `onErrorResumeNext`, `retry`

6. **Backpressure Operators**
    - Control flow and rate: `onBackpressureBuffer`, `onBackpressureDrop`, `onBackpressureLatest`
    
7. **Connectable Operators**
   - Control stream emission: `publish`, `replay`, `multicast`

8. **Error Management Operators**
    - Handle errors: `onErrorReturn`, `onErrorResumeNext`, `subscribe`, `retry`

9. **Async/Completing Operator**
    - Control stream termination: `delay`, `timeout`, `timeOut`
  
10.  **Collecting Operators**
    - Collect and emit values as a single item: `toList`
    - Aggregators: `reduce`, `scan`
    - Buffer: `buffer`

11. **Conditional and Boolean Operators**
    - Apply conditions: `all`, `contains`, `isEmpty`
    - Boolean operations: `takeUntil`, `skipUntil`

12. **Lifecycle Management**
    - Control subscription lifecycle: `takeUntil`, `skipUntil`

13. **Testing and Debugging**
    - Probe and debug: `doOnNext`, `doOnError`, `doOnComplete`

14.  **Saturated Feedback Loop**
    - Feed (eagerly): `connect`
<br>

## 11. What is _RxJava_, and how does it implement _Reactive Programming_?

**RxJava** is a popular Java library for designing and executing asynchronous and event-driven code in a **reactive** way, using a rich set of **observable** sequences and operators.

### Fundamental Components

- **Observable**: Represents a data stream of events or data packets.
- **Observer**: Receives and reacts to events.
- **Subscriber**: Similar to an Observer, but built with more functionalities for managing resources and backpressure.
- **Operators**: Allow modifications, transformations, and manipulations of the data stream.
- **Schedulers**: Specify the thread on which an Observable will emit, and on which a Subscriber will be notified.

### Core Concepts

#### 1. Data Streams

RxJava models data as **asynchronous**, time-varying, and continuous flows. These data streams can be of:

- Time-dependent data (like ticker prices).
- Data occurring at irregular intervals (like user interactions).
- Continuous data (like sensor readings).

#### 2. Single Source of Truth

With RxJava, data comes from a single source, making it easier to manage the **state**.

#### 3. Time-Related Events

RxJava handles time, providing features to deal with events that happen within a particular time window. These include:
- **Delay**: Initiates an action after a specified time.
- **Throttling**: Limits events to a fixed rate.


### RxJava in Action: Code Examples

#### Example: Basic Observable and Observer

Here is the Java code:

```java
import io.reactivex.Observable;
import io.reactivex.Observer;
import io.reactivex.disposables.Disposable;

public class SimpleRxExample {
    public static void main(String[] args) {
        // Create an Observable
        Observable<String> simpleObservable = Observable.just("Hello, RxJava!");

        // Create an Observer
        Observer<String> simpleObserver = new Observer<String>() {
            @Override
            public void onSubscribe(Disposable d) {
                // No action needed for demo purposes
            }

            @Override
            public void onNext(String s) {
                System.out.println(s);
            }

            @Override
            public void onError(Throwable e) {
                // No action needed for demo purposes
            }

            @Override
            public void onComplete() {
                // No action needed for demo purposes
            }
        };

        // Connect the Observer with the Observable
        simpleObservable.subscribe(simpleObserver);
    }
}
```

#### Example: Using Operators

RxJava provides various operators for performing common tasks such as filtering, mapping, and more. Example operators include: **map()**, **filter()**, and **observeOn()**.

Here is the Java code:

```java
import io.reactivex.Observable;
import io.reactivex.Observer;
import io.reactivex.disposables.Disposable;
import io.reactivex.schedulers.Schedulers;

public class OperatorExample {
    public static void main(String[] args) {
        // Create an Observable
        Observable<Integer> observable = Observable.range(1, 10)
                .filter(num -> num % 2 == 0)
                .map(num -> num * 2)
                .subscribeOn(Schedulers.computation());  // Execute on a computation thread

        // Create an Observer
        Observer<Integer> observer = new Observer<Integer>() {
            @Override
            public void onSubscribe(Disposable d) {
                // No action needed for demo purposes
            }

            @Override
            public void onNext(Integer integer) {
                System.out.println(integer);
            }

            @Override
            public void onError(Throwable e) {
                // No action needed for demo purposes
            }

            @Override
            public void onComplete() {
                // No action needed for demo purposes
            }
        };

        // Connect the Observer with the Observable and execute
        observable.subscribe(observer);
    }
}
```
<br>

## 12. How does _RxJava_ handle _multithreading_?

**RxJava** provides a flexible, yet potentially complex model for **multithreading**. It manages datastreams using **Schedulers**, ensuring the right tasks are run on the right threads, all while offering off-the-shelf tools for immediate integration.

### Schedulers

- **Immediate Scheduler**: Runs tasks immediately and on the current thread.
- **Trampoline Scheduler**: Schedules tasks after the current one finishes, ensuring sequential execution.
- **New Thread Scheduler**: Starts a new thread for each task, offering parallel execution for independent tasks but resource overhead.
- **Single Scheduler**: Similar to the new thread scheduler, but uses a single thread for all tasks.
- **IO Scheduler**: Designed for IO-bound tasks, uses a thread pool optimized for such operations.

### Android-Specific Schedulers

- **Android Main Thread Scheduler**: For UI tasks, ensuring they run on the main thread.

### Thread Safety and Convenience Methods

RxJava operators and observables are by default **thread-agnostic**, meaning they don't enforce specific threads for tasks. This design grants flexibility but requires developers to ensure their code is **thread-safe**.

RxJava simplifies this process by providing `subscribeOn()` and `observeOn()` methods.

### `subscribeOn()`

Defines the thread in which the Observable will be subscribed to.

### `observeOn()`

Specifies the thread in which the Observer's `onNext()`, `onComplete()`, and `onError()` methods will be invoked.

### Best Practices

- Prefer starting tasks on an **IO scheduler** to prevent congestion on a single thread.
- Always handle **long-running operations** on a background thread to prevent UI lockups.
- Use `observeOn()` to efficiently **transfer data** between threads instead of switching threads within operators.
<br>

## 13. Explain how the `flatMap` operator works in _RxJava_.

**RxJava** lets developers use the `flatMap` operator to map each item from the **Observable source** to a **new Observable**, then **flatten and merge** all of these Observables into a single stream of items.

This is especially useful when dealing with asynchronous operations that might need to be reordered or combined.

### How `flatMap` Works

When an item $X$ is emitted by the Observable source, `flatMap`:

1. **Applies a Function** that maps `X` to a new Observable $let's call this \(O_X$\).
2. **Merges** $O_X$ into the main output Observable (let's call it $M$).

The complete output might merge Observables multiple times, and the order of the final emissions is not necessarily the same as the order they are emitted.

### Code Example: `flatMap`

Here is the Java/Kotlin code:

```java
Observable<Integer> source = Observable.just(1, 2, 3);
source
    .flatMap(x -> Observable.just(x, x + 1))
    .subscribe(System.out::println);
```

In this case, each item $X$ from the source is mapped to a small Observable that immediately emits itself as well as the next incremented number after $X$. The output we observe is the following sequence:

1. 1 (from the first emission of 1 from the source)
2. 2 (from the second emission of 1 from the source)
3. 2 (from the first emission of 2 from the source)
4. 3 (from the second emission of 2 from the source)
5. 3 (from the first emission of 3 from the source)
6. 4 (from the second emission of 3 from the source)

### Key Takeaways

- Unlike `\map`, which transforms each element into a single element, `flatMap` can transform each element into **zero**, **one**, or **multiple** elements, backed by an Observable.
- This operator is versatile for tasks such as:
  - Combining data from multiple sources into a single output.
  - Task parallelism or concurrency by initiating several tasks from a single input.
  - Performing cleanup or any action that needs to associate with the original element when one is removed or replaced in a stream.
<br>

## 14. What is the purpose of the `zip` operator in _RxJava_?

The **zip** operator is a crucial tool in many **reactive programming** libraries like RxJava. It enables developers to combine asynchronous source Observables, where each emission combines with the latest emission from all other sources.

### Key Benefits

- **Synchronization**: Zip is a tracker and only emits when all sources have recent data.
  
- **Resource Efficiency**: It does not accumulate any excess data.

### Architecture-Centric Use-Cases

#### Dynamic UI Composition

- **Scenario**: Say you have two HTTP service requests, one for a user's profile and another for their recent orders. In the UI, you wish to present these two bits of data together.
  
- **Role of Zip**: By zipping the Observables from both requests, you ensure that new data from one source only gets displayed alongside the latest data from the other.

#### Complex Form Validation

- **Scenario**: Imagine a dynamic form where one field's validity depends on the value of another. For example, let's say you have a 'confirm password' field that should match the 'password' field.
  
- **Role of Zip**: Zipping Observables that correspond to these fields' user inputs allows you to synchronize their validation states. This way, you can ensure that both password fields match before form submission.

### Code Example: Dynamic UI Composition

Here is the Java code:

```java
Observable<UserProfile> userProfileObservable = repository.getUserProfile(userId);
Observable<List<Order>> userOrdersObservable = repository.getUserOrders(userId);

Observable<CombinedUserData> combinedUserDataObservable = Observable.zip(
        userProfileObservable, 
        userOrdersObservable, 
        (profile, orders) -> new CombinedUserData(profile, orders)
);
```

### Code Example: Complex Form Validation

Here is the Java code:

```java
Observable<String> passwordObservable = RxTextView.textChanges(passwordEditText)
        .map(CharSequence::toString)
        .skip(1)
        .debounce(300, TimeUnit.MILLISECONDS);
Observable<String> confirmPasswordObservable = RxTextView.textChanges(confirmPasswordEditText)
        .map(CharSequence::toString)
        .skip(1)
        .debounce(300, TimeUnit.MILLISECONDS);

Observable<Boolean> passwordMatchObservable = Observable.zip(passwordObservable, confirmPasswordObservable,
        (password, confirmPassword) -> password.equals(confirmPassword));
```
<br>

## 15. How do you handle _errors_ in an _RxJava_ stream?

**Reactive Extensions** provide multiple mechanisms for handling errors effectively within RxJava streams.

### Error Types in RxJava

- **onError**: Notifies the observer that an error occurred, and the stream won't produce any further events.
- **onErrorReturn**: Instead of terminating the stream, this operator allows you to emit a default value and then end.
- **onErrorResumeNext**: Lets you switch to another observable when an error is encountered.
- **retry**: As the name suggests, it resubscribes to the source observable on error, potentially with a limit on the number of retries.

### Best Practices

- **Choose Your Operator Wisely**: The different error-handling operators have distinct purposes, so make sure to use the one that best fits your specific use case.
- **Keep Operations Light**: It's best to keep any error-handling operations lightweight to maintain code readability and efficiency.
- **Prioritize Readability**: When employing reactive constructs, aim for code that's easy to read, understand, and maintain.
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - Reactive Programming](https://devinterview.io/questions/web-and-mobile-development/reactive-programming-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

