# Top 100 Redux Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - Redux](https://devinterview.io/questions/web-and-mobile-development/redux-interview-questions)

<br>

## 1. What is _Redux_ and how is it used in web development?

**Redux** is a predictable state container principally designed for JavaScript applications. It efficiently manages and updates the app state, serving as a **centralized global store**.

### Core Concepts

#### Store

The **store** is at Redux's nucleus and acts as a single source of truth. Here, you'll find the current state alongside the ability to dispatch actions.

#### Actions

**Actions** are payloads of data - typically objects - that describe what needs to happen. These are dispatched to the store.

#### Reducers

**Reducers** are pure functions that specify how the state changes in response to actions. Each reducer handles a particular part of the state and combines to produce the full state.

#### View

The user interface (UI) or the output your app produces, is ideally a direct derivative of the current state. Both are connected through subscription and rendering.

### Data Flow

Redux follows a straightforward loop, or **unidirectional data flow**, that tallies with its action-reducer-store structure.

1. **Acquiring Action**:
   - The UI elements, like buttons, yield actions.

2. **Dispatcher Role**:
   - Through `store.dispatch(action)`, these actions are transmitted to the store.

3. **Action Execution and State Mutation**:
   - With defined logic, reducers modify the state, originating a new state tree.

4. **State Change Subscription notification**:
   - Subscribed UI segments receive alerts about the state change and update correspondingly.

5. **Render Action**:
   - The updated state triggers view re-renders, aligning the UI with the latest state.

### Key Benefits

- **Increased Predictability**: The sequence of state alterations is preset and controlled.
- **Simplified State Management**: Every part of the state is stored in one location.
- **Streamlined Code Execution**:
  - Reducers define state modifications, making it easier to trace changes.
  - Debugging becomes more straightforward and time-efficient.

### Application Across Tech Stacks

While most associates developed Redux for React, the library is platform-agnostic. Adaptors and connectors permit its use with a spectrum of libraries, chiefly **react-redux**. Other integrations include Angular, Vue.js, and vanilla JavaScript applications.
<br>

## 2. Can you describe the three principles that _Redux_ is based upon?

Let's look at the three key principles forming the backbone of **Redux**: **Single Source of Truth**, **State is Read-Only**, and **Changes with Pure Functions**.

### Single Source of Truth

The **state of your whole application** is stored in a single tree or object within the Redux store. This makes it easier to identify where and how the state is changing, leading to better predictability and manageability.

#### Core Benefit

- **Centralized View of State**: You can get a comprehensive snapshot of your app's state at any point in time, making debugging and tracking data flow easier.

### State is Read-Only

In a Redux setup, the state is never modified directly. Any required changes are implemented by dispatching actions capturing the point-by-point changes expected in the state.

#### Core Benefits

- **Clarity and Control**: By preventing direct state alterations, Redux emphasizes a controlled and defined mechanism for state mutation, which is the action creator.
- **Time-Travel Debugging**: Because actions are recorded, you have the option to revert to specific application states for debugging.

### Changes with Pure Functions

The state changes in Redux are driven by pure functions called **reducers**. These functions take the current state and an action as input, and then **return a new state** without altering the previous state.

#### Core Benefits

- **Predictable Outcomes**: Since reducers are deterministic, their output for any given input and action will always be the same, ensuring consistency.
- **Testability and Maintainability**: Pure functions are easier to test and understand, making your codebase more reliable and maintainable.
<br>

## 3. What is an _action_ in _Redux_?

Let's start with an **overview** of actions in Redux before jumping into the code.

### Actions in Redux

In Redux, an **action** encapsulates a "payload" of data meant to change the state. It also carries a descriptive **type**, defining the nature of the state change.

Actions are dispatched by the front-end components and processed by **reducers**, which are pure functions, ensuring predictability.

#### Anatomy of an Action

An action in Redux is created as an object literal with at least one mandatory property, `type`. Optionally, you can have additional data known as the "payload". Here's a typical structure:

```javascript
const myAction = {
  type: 'ADD_TODO',
  payload: {
    id: 1,
    text: 'Buy groceries'
  }
};
```

### Code Example: Action

Here's the TypeScript code:

```typescript
// Defining the Action Interface
interface AddTodoAction {
  type: 'ADD_TODO';
  payload: { id: number; text: string };
}

// Creating an action
const addTodoAction: AddTodoAction = {
  type: 'ADD_TODO',
  payload: { id: 1, text: 'Buy groceries' }
};
```

### Considerations

- **Encapsulation**: The type and payload ensure actions are descriptive and structured, simplifying code readability and debugging.
- **Type Safety**: Utilizing TypeScript (or a similar language) for actions brings robust type checking to the development process.
  
For added safety and consistency, consider using action creators.
<br>

## 4. How are _actions_ used to change the _state_ in a _Redux_ application?

**Redux** follows a strict unidirectional data flow pattern, which means that **actions** are the only way to modify **state**.

### The Cycle of Action and State

1. **Action Creation**: You use **Action Creators** to create actions.

2. **Dispatch**: Actions are dispatched using `store.dispatch(action)`. Here, the `store` is the single source of truth in a Redux app and is responsible for action delivery to reducers.

3. **Reducer Listening**: Reducers, which are pure functions, listen for dispatched actions. If a matching action is identified, the reducer performs the necessary state modifications and returns the new state.

4. **Store Update**: The updated state is then stored, and any bound UI components are notified for necessary updates.

### Key Points to Remember

- **State is read-only**: Direct changes to state are not allowed. Instead, an action generator must be dispatched.

- **Synchronous Updates**: Reducers are responsible for synchronous state updates. Any asynchronous logic (like API calls) is typically handled by middleware like **Redux Thunk**.

- **Single Source of Truth**: The application's state is maintained in a single **store**, making it easier to manage and debug.

#### Code Example: Dispatching an Action

Here is the JavaScript code:

```javascript
// Action Creator
function increment() {
  return { type: 'INCREMENT' };
}

// Dispatching the action
store.dispatch(increment());
```

In the example, the `increment` function is an **Action Creator** that creates an **action** with type `'INCREMENT'`. The `store.dispatch` method then dispatches this action.

### Handling State with Reducers

Reducers, pure functions, are the cornerstone in the **state** modification process. When they receive an **action** and the current **state**, they decide how to modify that state. Upon returning a new, updated state, the global **store** replaces the current state with this new state.
<br>

## 5. What is a _reducer_ in _Redux_ and what role does it play?

A **reducer** in **Redux** is a pure function responsible for managing specific parts of your application state. It captures state changes and computes the new state based on those changes.

### Key Characteristics

- **Predictability**: Given the same input, a reducer will always produce the same output, making state management more predictable.
  
- **Immutability**: Reducers handle state immutably, ensuring that every state transition results in a new state object. This helps to avoid side effects and enhances the efficiency of UI and state management libraries like React and their change detection algorithms.

- **Single Responsibility**: Each reducer is geared towards managing a particular slice of application state. This limited scope aids in keeping the codebase organized and makes it easier to debug and maintain.

### Methods and Libraries

Redux provides utility functions, such as `combineReducers()`, to help manage multiple reducers, each responsible for a distinct part of the application state. Libraries like **immer** and **Redux Toolkit** further simplify the process of writing reducers, especially for handling immutable state updates.

### Code Example: Reducer

Here is the JavaScript code:

```javascript
const counterReducer = (state = { count: 0 }, action) => {
  switch (action.type) {
    case 'increment':
      return { count: state.count + 1 };
    case 'decrement':
      return { count: state.count - 1 };
    default:
      return state;
  }
};
```

In this example, `counterReducer` ensures **immutability** by creating and returning a new state object with each action dispatched.

### Practical Application

Associating actions with reducers allows for a modular and data-driven approach to state management. For instance, clicking a button might dispatch an **'increment'** action, processed by the **counterReducer** to adjust the count value.
<br>

## 6. How does _Redux_ differ from _local component state_ in _React_?

Let's look at the **key differences** between using **Local Component State** and **Redux** for managing state in **React**.

### Why Use Global State Management

Managing state **globally**:

-  Is especially useful for larger applications, as it streamlines access and updates to state across components.
-  Simplifies **data sharing** between components located at different levels of component hierarchy.

### Advantages and Limitations of Local Component State

#### Advantages

- **Simplicity**: It's quick and easy to set up local state within a component.
- **Scoped Updates**: Changes to local state are contained within the component, making it easier to manage and reason about.

#### Limitations

- **State Duplication**: Keeping similar state in sync across multiple components can lead to inconsistencies and bugs.
- **Passing State and Event Handlers**: Necessitates drilling down state and event handlers to nested and related components.
- **Hierarchical Data Flow**: Forces a one-way, parent-to-child data flow, making certain patterns unwieldy to implement.

### Advantages of Global State Management with Redux

- **Centralized State**: Provides a single source of truth, which safeguards against inconsistencies across the application.
- **Predictable State Updates**: Leveraging reducers ensures controlled and consistent state modifications.
- **Data Sharing**: Facilitates data sharing between components without the need for prop drilling.
- **Time-Travel Debugging**: Through tools like Redux DevTools, it makes it possible to revisit and review the state at any point in time during the application's lifecycle.

### When to Choose Which Approach

- **Local State**: Opt for local state management for simpler, predominantly UI-related state or for small-scale applications with minimal state requirements or data sharing needs.
  
- **Redux (or Another Global State Management)**: Choose a global state management system when working with larger-scale applications, multi-level state dependencies, or when extensive data sharing and consistency across the application is critical.
<br>

## 7. Define “_store_” in the context of _Redux_.

In the context of Redux, **the store** is a key component that acts as a central data hub.

### Store Components

1. **State**: Maintains the current data state, which is read-only. To update it, you dispatch actions.

2. **Reducers**: Action-specific functions are responsible for modifying the state and are combined using `combineReducers()`.

3. **Listeners**: Registered callbacks are notified whenever the state is updated.

### Store Mechanics

- **getState()**: Use this method to access the current state.

- **dispatch(action)**: Enforce state updates by providing an action, typically an object with a `type` property.

- **subscribe(listener)**: Keep listeners updated on state changes by registering them.

- **replaceReducer(nextReducer)**: Use the provided reducer to update state-handling logic.

### Store Core Principles

- **Single Source of Truth**: All application data is managed through a single store, ensuring consistency.

- **State is Read-Only**: Prevent accidental state modifications outside reducers to maintain a deterministic behavior.

### Store Setup

To establish a Redux store in your application, use the `createStore()` function and pass in your app's reducers. Then, you can:

- Access the initialized store using `getState()`, `dispatch()`, and `subscribe()`.
- Update the store's reducers dynamically with `replaceReducer()` when necessary.
<br>

## 8. Can you describe the concept of "_single source of truth_" in _Redux_?

**Redux, a predictable state container**, adheres to the **single source of truth** principle. It stipulates that there is a single immutable state tree shared across the entire application.

### Key Benefits

- **Consistency**: All components reflect the latest state, minimizing data inconsistencies.
- **Easy Debugging**: A single point to monitor state changes improves debugging.
- **Conciseness and Clarity**: The central state tree simplifies data management and visibility.
- **Correctness Guarantee**: No parallel updates are possible, leading to improved application stability.

### Potential Drawbacks and Solutions

- **Performance Implications**: Extract only necessary parts of the state for components, often accomplished through selectors.
- **Possible Bottlenecks**: Employ subscriptions or alternative mechanisms for efficient data flow while recognizing the trade-offs.
<br>

## 9. How do you create a _Redux store_?

Creating a **Redux store** involves several key steps. The store serves as a centralized hub for state management across the application.

You can create a Redux Store in any of the three ways:

- **Directly** using `createStore(reducer, [preloadedState], [enhancer])`
- **Via a function** that combines multiple reducers using `combineReducers(reducers)` 
- With **Middlewares** for executing logic in the data flow using `applyMiddleware(middleware1, middleware2, ...)`

### Steps to Create a Redux Store

#### Define the Reducer

First, you have to define at least one **reducer**, a function that determines the shape and state transitions of the store.

If you are going to use multiple reducers, to manage different parts of the state tree, you will need to combine them.

Here is the example code:

```javascript
// counterReducer.js
const initialState = { count: 0 };

const counterReducer = (state = initialState, action) => {
  switch (action.type) {
    case 'INCREMENT':
      return { count: state.count + 1 };
    case 'DECREMENT':
      return { count: state.count - 1 };
    default:
      return state;
  }
};

export default counterReducer;
```

#### Create a Reducer Function

```javascript
// reducers/index.js
import { combineReducers } from 'redux';
import counterReducer from './counterReducer';

const reducers = combineReducers({
  counter: counterReducer,
  // Add more reducers here
});

export default reducers;
```

#### Import Additional Reducers

If you have more than one reducer, use `combineReducers` to manage them together.

```javascript
// reducers/index.js
import { combineReducers } from 'redux';
import counterReducer from './counterReducer';
import userReducer from './userReducer';

const reducers = combineReducers({
  counter: counterReducer,
  user: userReducer,
});

export default reducers;
```

#### Optional: Add Initial State and Middlewares

A central state can be provided as initial state, for the reducers to pick up their specific sections.

The element `applyMiddleware` should be called with any middlewares you're employing in your application.

```javascript
// store.js
import { createStore, applyMiddleware, compose } from 'redux';
import reducers from './reducers';

const store = createStore(
  reducers,
  {
    counter: { count: 10 },
    // Additional initial states for more reducers
  },
  compose(applyMiddleware(middleware1, middleware2))
);

export default store;
```

### Complete Example

Here is the complete example with all the steps:

```javascript
// counterReducer.js
const initialState = { count: 0 };

const counterReducer = (state = initialState, action) => {
  switch (action.type) {
    case 'INCREMENT':
      return { count: state.count + 1 };
    case 'DECREMENT':
      return { count: state.count - 1 };
    default:
      return state;
  }
};

export default counterReducer;

// reducers/index.js
import { combineReducers } from 'redux';
import counterReducer from './counterReducer';
import userReducer from './userReducer';

const reducers = combineReducers({
  counter: counterReducer,
  user: userReducer,
});

export default reducers;

// store.js
import { createStore, applyMiddleware, compose } from 'redux';
import reducers from './reducers';

const store = createStore(
  reducers,
  {
    counter: { count: 10 },
    // Additional initial states for more reducers
  },
  compose(applyMiddleware(middleware1, middleware2))
);

export default store;
```
<br>

## 10. What is meant by "_immutable state_," and why is it important in _Redux_?

***Immutable State*** is a key concept in **Redux**, which ensures that the state of an application cannot be changed directly. State alterations are achieved through actions and reducers, making the state more predictable and facilitating better debugging.

### Advantages of Immutable State in Redux

- **State History**: By preserving each state change, you can step forward or backward in time, aiding debugging and enabling features like undo/redo.
  
- **Performance Optimizations**: Redux utilizes reference equality checks. If the new and old states are the same, components avoid unnecessary re-renders.

- **Simplified Component Updates**: With unchanged objects, components can avoid updating unless their inputs change.

- **Threading Safety**: In multi-threaded applications, immutable objects guarantee that state changes don't create race conditions.

### Persisting Immutability in Javascript

The `const` declaration in JavaScript, while not offering immutability for object properties, does ensure that the reference itself (the state object in the context of Redux) remains constant, aligning with Redux's immutability requirement.

#### Ways to Ensure Immutability in Redux

- **Spread Operator**: For shallow cloning of state objects
- **Deep Clone**: For nested objects and arrays. However, this approach might be inefficient for large state objects and should be used judiciously.

#### Immutability Libraries

To simplify working with immutable data, you can use libraries such as:

- **Immer**: Offers a minimalistic API for immutable state management
- **Immutable.js**: Provides a complete data structure library, including both primitive and collection data types. Though powerful, it can be challenging to integrate with existing codebases due to its API differences.
<br>

## 11. Explain the significance of the `combineReducers` function.

The `combineReducers` function in **Redux** plays a central role in structuring large web applications.

### Benefits of `combineReducers`

- **Code Modularity**: By organizing reducers into smaller modules, it even becomes easier to share and reuse them across different parts of the application.

- **Convenience and Clarity**: Instead of dealing with a single overwhelmingly large reducer, developers can manage logically distinct parts of the application separately.

- **Selective Data Handling**: Reducers can act in specified functional areas only, keeping the rest of the state untouched.

- **Efficient Data Flow**: Each reducer only 'listens' to that part of the state that is relevant, potentially improving performance.

### Code Example: `combineReducers` in Action

Consider a note-taking app, which might have three distinct state trees: `notes`, `ui`, and `user`.

Here is how `combineReducers` can be employed:

```javascript
import { combineReducers } from 'redux';
import notesReducer from './notesReducer';
import uiReducer from './uiReducer';
import userReducer from './userReducer';

const rootReducer = combineReducers({
  notes: notesReducer,
  ui: uiReducer,
  user: userReducer
});

export default rootReducer;
```

Each of the imported reducers is a slice of the overall state tree. When an action is dispatched, it is then passed to each of these reducers. They process the action, and the relevant parts of the state tree get updated.
<br>

## 12. What are _pure functions_ and _side effects_ in the context of _Redux_?

In the context of **Redux**, actions alter state through **reducers**, and **middlewares** handle side effects. It's essential that reducer functions be **pure** to ensure predictability in state changes.

### Fundamental Concepts

- **Actions**: Objects describing a state change.
- **Reducers**: Pure functions defining the state change.

- **State**: An immutable object accessible throughout the app.

### Pure Functions

**Reducers** must be pure, that is, their behavior should be entirely predictable.

- **Determination of Output**: Output should be a direct result of input. The function should not base its output on any external state or data.
- **No Side Effects**: They should not cause alterations outside their scope, like modifying global variables or invoking functions that perform I/O operations.
- **Idempotence**: Executing the function multiple times with the same input should provide consistent outcomes and not provoke side effects.

Redux utilizes the principle of **immutability**: once in place, actions or state don't undergo alteration. Instead, they generate a 'new' state that replaces the existing one in its entirety.

### Side Effects

Side effects allude to alterations induced by a function that aren't limited to its return value. **Side-effectful functions** can engage in various actions that extend beyond their obvious purpose. While some side effects are necessary, keeping them compartmentalized is crucial to maintaining code stability and ease of debugging. Side-effect management in Redux is facilitated by **middlewares**, like `redux-thunk` or `redux-saga`. 

#### Code Example: Impure Reducer

This is the Python code:

```python
# Non-Pure Function

state = 0

def increment_counter():
    global state  # Modifies global state
    state += 1
    return state  # Returns a value different from the input

# Result: Output is unpredictable and changes external state
```

### Side Effect Management Methods

Various Redux-specific libraries, such as **Thunks** or **Sagas**, provide mechanisms to tackle side effects in a structured way.

#### Thunks

Thunks are functions that can generate other functions, permitting activities that extend past straightforward dispatch. `redux-thunk` is a well-established middleware that empowers thunks in a Redux store.

#### Code Example: Thunk in Action

This is the JavaScript Code:

```javascript
// Action Creator
const incrementAsync = () => {
  // Returning a function (a thunk)
  return (dispatch) => {
    // Asynchronous Operation
    setTimeout(() => {
      dispatch({ type: 'INCREMENT' }); // Dispatch a regular action after a delay
    }, 2000);
  };
};
```

### Sagas

Sagas are designed on the concept of cooperative concurrency to ***streamline actions with side effects***. In a redux-saga powered store, you interact with such sagas using the middleware `applyMiddleware(sagaMiddleware)`. This is immensely useful for simultaneous operations and complex co-dependencies, especially with asynchronous processes.

#### Code Example: Saga for Input Validation

Here is the JavaScript code:

```javascript
import { takeLatest, call, put } from 'redux-saga/effects';
import { START_SUBMIT, SUCCESS_SUBMIT, FAILED_SUBMIT } from './actionTypes';
import { validateInput } from './api';

function* submitForm(action) {
  const {input} = action.payload;
  try {
    yield call(validateInput, input);
    yield put({ type: SUCCESS_SUBMIT });
  } catch (error) {
    yield put({ type: FAILED_SUBMIT, error });
  }
}

export default function* formSaga() {
  yield takeLatest(START_SUBMIT, submitForm);
}
```
<br>

## 13. How do you handle _asynchronous actions_ in _Redux_?

**Redux'** action creators typically produce immediate synchronous actions. To handle asynchronous behavior, **middleware**, especially **redux-thunk**, is employed.

With `redux-thunk`, action creators can return **functions** instead of plain action objects.

### How `Redux-Thunk` Works

1. **Configuration**: When setting up the Redux store, use `applyMiddleware` to incorporate `redux-thunk`.
2. **Action Dispatch**: When an action creator returns a function, `redux-thunk` intercepts it. This function is then given `dispatch` as its first parameter.

3. **Async Control**: Inside the function, you can manually dispatch actions, often used to denote the start and completion of an asynchronous flow.

### Code Example: Using Thunk Middleware in Redux

Here is the JavaScript code:

```javascript
// configureStore.js
import { createStore, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';
import rootReducer from './reducers';

const store = createStore(rootReducer, applyMiddleware(thunk));
export default store;

// actions.js
export const fetchData = () => {
  return async (dispatch) => {
    dispatch({ type: 'FETCH_DATA_REQUEST' });
    try {
      const data = await someAPICall();
      dispatch({ type: 'FETCH_DATA_SUCCESS', payload: data });
    } catch (err) {
      dispatch({ type: 'FETCH_DATA_FAILURE', error: err.message });
    }
  };
};

// reducer.js
const initialState = { data: null, loading: false, error: null };
const reducer = (state = initialState, action) => {
  switch (action.type) {
    case 'FETCH_DATA_REQUEST':
      return { ...state, loading: true, error: null };
    case 'FETCH_DATA_SUCCESS':
      return { ...state, data: action.payload, loading: false };
    case 'FETCH_DATA_FAILURE':
      return { ...state, loading: false, error: action.error };
    default:
      return state;
  }
};
```
<br>

## 14. What is a “_selector_” in _Redux_ and what is its purpose?

A **selector** in **Redux** is a **pure function** that efficiently computes derived data from the **Redux store state**.

### Purpose

The key purpose of selectors is to offer:

- **Data Transformation**: Selectors calculate computed data, such as derived state or complex aggregations.
- **Performance Optimization**: They help prevent expensive recalculations and re-renders in components by caching results.

### Core Segments

- **State Slice**: Selects a relevant part of the application state.
- **Data Transformation**: Computes and processes the state slice to generate derived data.
- **Caching Mechanism**: Efficiently handles and updates the selector's result cache.

### Role in Redux Architecture

1. **Input Source**: Selectors receive the complete state object from the store as their input.
2. **Centralized Data Access**: They serve as the primary data source for Redux-connected components.
3. **Efficiency and Consistency**: By transforming and caching state data, selectors improve overall application performance.

### Code Example: Selectors in Redux

Here is the JavaScript code:

```javascript
// Example State
const appState = {
  user: {
    name: "John Doe",
    email: "john.doe@example.com",
    age: 30,
    lastLogin: "2023-10-15T08:00:00.000Z",
  },
  products: {
    list: [
      { id: 1, name: "Product 1", price: 100, category: "electronics" },
      { id: 2, name: "Product 2", price: 150, category: "clothing" },
      { id: 3, name: "Product 3", price: 200, category: "electronics" },
    ],
  },
};

// Selectors
const getUser = (state) => state.user;
const getUserEmail = (state) => getUser(state).email;
const getElectronicsProducts = (state) => {
  return state.products.list.filter((product) =>
    product.category === "electronics"
  );
};

// Usage with Selector Functions
console.log(getUserEmail(appState)); // Output: john.doe@example.com
console.log(getElectronicsProducts(appState));
// Output: [ { id: 1, name: "Product 1", price: 100, category: "electronics" }, { id: 3, name: "Product 3", price: 200, category: "electronics" } ]

// Caching Mechanism: Example with Memoization
const getAllProducts = (state) => state.products.list;
const memoizedElectronicsSelector = createSelector(
  [getAllProducts],
  (products) => products.filter((product) => product.category === "electronics")
);

// Add a new product to trigger re-calculation
appState.products.list.push({ id: 4, name: "Product 4", price: 300, category: "electronics" });
// Re-run the memoized selector
console.log(memoizedElectronicsSelector(appState));
// Output: [ { id: 1, name: "Product 1", price: 100, category: "electronics" }, { id: 3, name: "Product 3", price: 200, category: "electronics" }, { id: 4, name: "Product 4", price: 300, category: "electronics" } ]
```
<br>

## 15. How does _Redux_ handle the flow of _data_ and _actions_?

In **Redux**, the **unidirectional data flow** concept ensures data consistency by using one-way channels for state and actions. Let's look at a more detailed overview:

### Key Elements

- **Actions**: These are **objects** with a `type` attribute, describing what needs to change in the application. Actions are dispatched from components and received by reducers.

- **Reducers**: These are **pure functions** that specify how the application's state should transform in response to actions.

- **Store**: It's a **single source of truth**. It holds the application's state tree and is responsible for:
  - State modification through reducers.
  - State dispatch as actions are sent.

### One-Way Data Flow

1. **Action Dispatch**: Components dispatch actions (through `store.dispatch(action)`).
2. **Reducer Receipt**: The store sends the action to the appropriate reducers.
3. **State Changes**: The reducer modifies the state based on the dispatched action.
4. **State Subscription**: The modified state is broadcast to subscribed components.
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - Redux](https://devinterview.io/questions/web-and-mobile-development/redux-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

