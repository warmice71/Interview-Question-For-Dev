# 100 Common AngularJS Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - AngularJS](https://devinterview.io/questions/web-and-mobile-development/angular-js-interview-questions)

<br>

## 1. What is _AngularJS_ and what are its core features?

**AngularJS** is an open-source front-end web framework maintained by Google and a vast community. It is specifically designed to overcome challenges in Single-Page Applications (SPAs) development.

### Core Features

#### Modular

AngularJS structures applications into **distinct modules**, facilitating modular development and easy maintenance.

#### Two-Way Data Binding

Data changes in the Model (back-end) or the View (front-end) **automatically update** the other.

#### Dependency Injection

This design pattern makes components independent by **injecting dependencies** from external sources rather than hard-coding them.

#### Directives

They are markers in the DOM that tell AngularJS's **HTML compiler** to attach a specific behavior to the DOM element or to transform the DOM element and its children.

#### Templates & Data Binding

AngularJS leverages **templates** that combine HTML with AngularJS directives and expressions for dynamic content. The framework offers sophisticated **data binding** to keep the UI and app data in sync.

#### RESTful API Interaction

AngularJS simplifies communication with **RESTful APIs**, making it easy to work with server-side data.

#### MVC Architecture

AngularJS embraces the Model-View-Controller (MVC) design pattern, which helps in organizing code into **modular, testable, and maintainable** units.

#### Advanced Routing

The framework supports **advanced routing capabilities**, enabling a multi-view layout by defining the route for each view.

#### Services

AngularJS facilitates the creation of **reusable business logic** units using services. These units encapsulate specific tasks and can be injected where needed.

#### Extensive Testing

Built-in support for **unit testing** allows components to be tested in isolation, aiding in bug detection and rapid iteration.

#### Third-Party Library Integration

AngularJS readily integrates with third-party libraries such as Bootstrap and jQuery.

#### Cross-Platform and Cross-Browser Compatibility

The framework, adhering to web standards, offers **consistent behavior across platforms and browsers**.

#### Data Validation

AngularJS simplifies the process of **form validation** through its built-in directives and services.

#### Continuous Improvements with Versions

As the framework progresses from AngularJS to Angular, later versions offer **enhancements in performance, developer experience, and security**.

### Consistent Data Handling

AngularJS maintains a unidirectional **flow of data** and has **mechanisms in place** to manage state, ensuring data accuracy and minimizing unexpected behaviors.

### Lifecycle Management

Components in AngularJS go through a **consistent lifecycle**: creation, rendering, updates, and eventual destruction. This lifecycle management allows for targeted operations at various stages.

### Pre-Build Optimization

Developers can enhance AngularJS applications using tools like **Ahead-of-Time (AOT) compilation**, which optimizes performance by moving logic and templates to offline, pre-compiled files.
<br>

## 2. Explain the concept of _two-way data binding_ in AngularJS.

**Two-way data binding** in AngularJS ensures that model changes automatically reflect in the view, and vice versa.

When a model or the input field value in UI gets updated, AngularJS instantly reflects the changes both ways. It essentially acts as a bridge, allowing seamless synchronization between the model and view in near real-time.

### Detailed Workflow

1. **Initial Data**: The data originates from the model and populates the view, providing users with initial values or content.

2. **View Modification**: When a user interacts with the UI, such as modifying an input field or selecting an option, AngularJS instantly updates the associated model.

3. **Model Changes**: These model adjustments, occurring in real-time, are then immediately updated in both model and view, aligning the two components.

4. **Validation and Coercion**: Inputs are also validated and optionally coerced.
   - When a user input contradicts data type expectations or fails validation rules, AngularJS can correct or validate it, ensuring data integrity.

5. **View Update**: The now-validated or coerced data returns to the view, empowering users with visual feedback on any adjustments or corrections.

The entire two-way data binding cycle ensures an instantaneous, synchronized relationship between the model and the UI. This feature significantly lowers the need for manual DOM manipulation or event handling.
<br>

## 3. What are _directives_ in AngularJS? Give some examples.

**Directives** in AngularJS are markers in the DOM that AngularJS library can reinterpret to inject extra behavior or allow data-binding. They encompass JS functions, DOM elements, and even comment elements.

### Core Directives

- **ngApp**: Defines the root element of the AngularJS application.

- **ngController**: Associates the controller with a section of the view.

- **ngModel**: Links an HTML element such as an input, select or textarea to a property on the scope.

- **ngBind**: Binds the innerHTML of the element to the expression.

- **ngBindHtml**: For DOM-based XSS protection, binds innerHTML to the expression after sanitizing it.

- **ngBindTemplate**: Used for inline templates.

- **ngClick**: Executes custom behavior on a click event.

- **Validation Directives**: AngularJS offers two types of field validation attributes:

1. **Regular HTML5 Validation Attributes**

   AngularJS sets up the appropriate validation properties on the model using the following HTML5 attributes:

   - required
   - min
   - max
   - minlength
   - maxlength
   - pattern

2. **Custom Validation Directives**

   You can build these custom directives to suit specific requirements. Common examples are:

   - match
   - unique-email

### Event Directives

These directives react to particular DOM events:

- **ngBlur and ngFocus**: Used to detect element focus or blur.

- **ngChange**: Triggers when the associated model is changed.

#### Others

The library supports more specific ones like **ngCut**, **ngCopy**, **ngPaste**.

### Styling and Class Directives

These directives dynamically alter CSS classes and inline styles:

- **ngClass** : switches classes based on object states.
- **ngStyle** : dynamically applies inline styles.

### Integration Directives

AngularJS aligns with other libraries through integration directives:

- **ngIf** : conditionally includes an element.
- **ngSwitch** : operates like a switch statement.
- **ngRepeat** : loops over arrays and objects.

### Forms Integration

AngularJS extends form behavior with:

- **ngForm** : groups form controls.

- **ngSubmit** : binds to the submit method of a form.

- **ngOptions** : dynamically populates select elements.

### Template Handling Directives

- **ngInclude** : fetches, compiles and includes an external HTML fragment into the directive element.

- **ngView** : sets up a consistent mechanism for multiple views in a Single Page Application (SPA).

### Miscellaneous Directives and Interactions

- **ngClassEven** and **ngClassOdd** : Simplify zebra striping.

- **ngCsp** : Essential in Content Security Policy protection.

- **ngPluralize** : Optimizes pluralization for different languages.
<br>

## 4. How does the _AngularJS digest cycle_ work for data binding?

The **AngularJS Digest Cycle** is responsible for managing **two-way** data binding, keeping the **View** and **Model** synchronized. The cycle usually runs automatically but can be triggered manually in certain cases. 

### Key Components

1. **Watchers**: These are functions that observe changes in the model, ensuring the View is updated accordingly.
2. **Dirty-Checking**: Angular performs this by comparing the current state of data (the "new" state) with the previous state (the "old" state). If there's a discrepancy, the system flags the data as "dirty", indicating a change requiring action.
3. **$watch**: This directive is used to track changes in model properties, although it should be used judiciously as excessive `$watcher` creation can negatively impact app performance.

### Manual Trigger Mechanism

Even though Angular aptly manages the Digest Cycle, developers can manually initiate it as well. This can be useful in cases where the Cycle does not trigger automatically.

To do this, you can use:

- `$scope.$apply()`: This tells Angular that a part of the code (a function or code block) was executed outside of its context and changes might have occurred. Consequently, it forces a two-way data binding sync. However, this typically triggers the complete Digest Cycle, which might lead to performance issues if used in high-frequency contexts.
  
- `$scope.$digest()`: Instead of initiating a complete Digest Cycle, you can specify to sync only certain portions. This method runs the Digest Cycle on the current scope and its children, stopping if no changes are detected.

### The Digest Loop Mechanism

Angular initiates the Digest Cycle in response to various events, like user actions or asynchronous activities. The process essentially involves two steps: change detection and view updating.

1. **Change Detection**: Identify alterations in the data. Angular does this by continuously comparing the current state with the previous state, flagging any discrepancies.
   
2. **View Update**: If changes are detected, update the View to reflect the new state. Angular ensures these updates are efficient and minimized to only the affected portions, optimizing application performance.

The Digest Cycle continues iteratively until:

- It encounters a stable state where no further changes are detected.
- It reaches maximum iteration counts, after which it throws an error, indicating a possible problem in the code.

### Example: The Digest Cycle in Action

Below is the AngularJS code that demonstrates the Digest Cycle with `ng-click` and manual control using `$digest`:

### AngularJS Code Example: Digest Cycle

```html
<!DOCTYPE html>
<html>
<head>
	<title>Digest Cycle</title>
	<script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.8.2/angular.min.js"></script>
</head>
<body ng-app="digestApp" ng-controller="digestCtrl">
	<h1>{{ title }}</h1>
	<p>Click count: {{ clickCount }}</p>
	<button ng-click="incrementCount()">Click me</button>
</body>
<script type="text/javascript">
	var app = angular.module('digestApp', []);

	app.controller('digestCtrl', function($scope) {
		$scope.title = 'Digest Cycle Example';
		$scope.clickCount = 0;

		$scope.incrementCount = function() {
			$scope.clickCount++;
			// Manually trigger the Digest Cycle after incrementing the click count
			$scope.$digest();
		};
	});
</script>
</html>
```

In this example, with each button click, we trigger `$scope.incrementCount()` to increment the click count and force a manual Digest Cycle with `$scope.$digest()`. Be aware that the manual initiation might not always be appropriate and can potentially lead to performance issues if not handled carefully.
<br>

## 5. What is _scope_ in AngularJS, and how is it different from the _JavaScript scope_?

**Scope** in **AngularJS** and **JavaScript** serve analogous purposes but differ in several aspects.



### AngularJS Scope and JavaScript Scope

| Distinction               | AngularJS Scope                                  | JavaScript Scope                                 |
|---------------------------|--------------------------------------------------|---------------------------------------------------|
| Scope Creation            | Automatically established for each controller   | Manually created with each function              |
| Initiation Control        | Programmatic declaration in a controller        | Unpredictable scope setup can sometimes lead to issues |
| Hierarchical Structure    | Forms parent-child relationship in views         | Unidirectional from inner to outer functions       |
| Scope Prototyping         | Inherits method `childScope.prototype = parentScope;` | Does not inherit prototype chains                |
| Definition Isolation      | Local to the controller and its views            | Global by default; local when using functions     |
| Scope Destruction Control | Automatic disposal on controller exit           | Garbage collection for variables with no references |

### Code example: AngularJS scope

Here is the JavaScript code:

```javascript
// Controller 1
angular.module('myApp').controller('Ctrl1', function($scope) {
   $scope.message = "Hello from Ctrl1!";
});

// Controller 2
angular.module('myApp').controller('Ctrl2', function($scope) {
   console.log($scope.message);  // This will output "Hello from Ctrl1!"
   $scope.message = "Hello from Ctrl2!";
});
```

Here is the AngularJS code:

```typescript
// Controller 1
angular.module('myApp').controller('Ctrl1', function($scope) {
   $scope.message = "Hello from Ctrl1!";
});

// Controller 2
angular.module('myApp').controller('Ctrl2', function($scope) {
   console.log($scope.message);  // This will output "Hello from Ctrl1!"
   $scope.message = "Hello from Ctrl2!";
});
```
<br>

## 6. Define what a _controller_ is in AngularJS.

In AngularJS, the **controller** is a core component that manages data interaction and mediates between the view and the model.

It's responsible for initializing the state of the $scope, binding model to the view, and handling any user interactions.

The controller's primary role is to initialize the **$scope** object, which acts as the glue between the controller and the view HTML. 

### Key Controller Responsibilities

1. **Managing Scope**: The controller defines what part of the model should be exposed to the view by attaching properties and functions to the `$scope` object.
  
2. **Data Handling**: It mediates data operations, such as fetching initial data, processing input, and updating the model.

3. **Event Handling**: The controller handles both view-specific DOM events and custom events $using `$emit` and `$broadcast` to communicate upwards or downwards through the scope hierarchy, respectively$.

4. **Code Encapsulation**: Encloses data and behavior, limiting the exposure to the global state where other components can also access it.
  
5. **Lifecycle Management**: Conducts cleanup tasks during specific stages of the component's lifecycle, such as removing event listeners or unsubscribing from observables.

6. **Service Integration**: It collaborates with various AngularJS services such as `$http` for making HTTP requests or `$q` for asynchronous operations.

### Entry Point : `ng-controller`

Every controller in AngularJS is associated with an HTML element via the `ng-controller` directive, specifying the controller's name.

For instance, in the code snippet below, the `MyController` is the registered controller, and its functionality is available within the `div` element it's associated with:

```html
<div ng-controller="MyController">
    <p>{{ myProperty }}</p>
</div>
```

### Example and Code Implementation

Here is the AngularJS code:

```javascript
// Define a controller named 'MyController'
app.controller('MyController', function($scope) {
    // Initialize a property on the $scope object
    $scope.myProperty = 'Hello, world!';
});
```

In this example, `app` is the reference to the AngularJS module. By invoking the `controller` method on `app` and providing a name $here, `'MyController'`$ and a **controller function**, a new controller is created. The controller function typically takes `$scope` as an argument, through which it configures the `scope` object for the given view.

Lastly, the `ng-controller` directive is used. When the AngularJS framework encounters this directive, it connects the `div` element and its children with the controller, enabling $scope-based properties and methods within that "scope".
<br>

## 7. Can you explain what a _service_ is in AngularJS?

**Services** in AngularJS are singletons and facilitate sharing of functions, objects, or values among different parts of an application. They cover various aspects such as data management, communication with servers, and more.

### Core Service Types in AngularJS

- `$animate`: Offers methods for animation, helping UI elements transition in a controlled manner.

- `$cacheFactory`: Serves as a key-value store for temporary data, enhancing application performance by reducing excessive data requests.

- `$compile`: Transforms AngularJS directives, templates, and scopes into corresponding HTML, ready for rendering.

- `$controller`: Primarily responsible for constructing the application’s controllers.

- `$document`: Acts as a wrapper around the browser's global `document` object.

- `$exceptionHandler`: Centralizes error handling, easing the process of debugging and monitoring errors in an AngularJS application.

- `$http`: Facilitates synchronous and asynchronous communication with remote servers. It provides support for traditional RESTful APIs as well as AJAX requests.

- `$injector`: Serves as the dependency injection module, managing dependencies across different components of the application.

- `$interval`: Offers recurring task scheduling based on a specific time interval.

- `$location`: Encapsulates and abstracts the URL of the browser.

- `$log`: A centralized logging tool for error and debugging messages.

- `$parse`: Responsible for parsing AngularJS expressions and converting them into functions.

- `$q`: Provides mechanisms for asynchronous task management, such as promises and deferred objects.

- `$rootElement`: Represents the root element of the entire AngularJS application.

- `$rootScope`: Acts as the parent scope of all other scopes within the application. Modifications to `$rootScope` are often discouraged.

- `$templateCache`: Stores AngularJS templates, enabling their retrieval without a server request.

- `$timeout`: Offers a way to schedule specific tasks to execute after a certain time delay.

- `$window`: Acts as a wrapper, encapsulating the global `window` object.

### Custom Service Types

Developers can create their own custom AngularJS services using any of three primary methods: factory, service, and provider.

1. **Factory**: Delivers objects or primitives. A factory function returns whatever type of object you want to provide. This gives you more freedom and can be particularly useful when there are dependencies that need to be injected into the factory function.

2. **Service**: Utilizes **constructor functions**. The service method takes a constructor function, or the name of a constructor function. When AngularJS injects your service into another component, it calls your constructor function with the "new" keyword to create an instance of the service.

3. **Provider**: The provider method is the most flexible and powerful of the three. It allows you to configure your service before your application starts by defining a provider recipe.
<br>

## 8. How do you share data between _controllers_ in AngularJS?

While AngularJS promotes a modular architecture where components are typically self-contained, you may need to share data between controllers. To facilitate this, AngularJS offers a few primary methods.

### Methods for Sharing Data between Controllers in AngularJS

#### 1. Parent-Child Relationship

  Identify that a hierarchal relationship already exists with one controller as the parent of the other. Here it is not mandatory to use the `vm`.

  **When to Use**: Parent-child relationships are useful when one view must contain the other and when you are building composite widgets.

```javascript
// Parent Controller
angular.module('myApp').controller('ParentCtrl', function($scope) {
  $scope.sharedData = 'Hello from parent';
});

// Child Controller
angular.module('myApp').controller('ChildCtrl', function($scope) {
  // Access the shared data from the parent
  $scope.childSharedData = $scope.sharedData;
});
```

#### 2. Using Services

   Create a service that acts as a data mediator between controllers. Use the `this` context to make the service's data accessible.

   **When to Use**: Services are an ideal choice when multiple controllers across the application need access to the same data.

```javascript
// Shared Data Service
angular.module('myApp').service('sharedDataService', function() {
  this.sharedData = 'Hello from shared service';
});

// Controller 1
angular.module('myApp').controller('Ctrl1', function(sharedDataService) {
  this.dataFromService = sharedDataService.sharedData;
});

// Controller 2
angular.module('myApp').controller('Ctrl2', function(sharedDataService) {
  this.dataFromService = sharedDataService.sharedData;
});
```

#### 3. Using AngularJS Events

   Utilize `$emit` and `$broadcast` to trigger and capture events throughout the application. `$rootScope` is essential for broadcasting.

   * `$emit`: Triggers events upwards through the $scope hierarchy.
   * `$broadcast`: Triggers events downwards through the $scope hierarchy.

   **When to Use**: Events are useful when one controller needs to notify other controllers in the application about an update or a change in the data.

```javascript
// Controller 1
angular.module('myApp').controller('Ctrl1', function($rootScope) {
  this.sendMessage = function(data) {
    $rootScope.$emit('customEvent', data);
  };
});

// Controller 2
angular.module('myApp').controller('Ctrl2', function($rootScope) {
  var unregister = $rootScope.$on('customEvent', function(event, data) {
    console.log(data); // Process the received data
    unregister(); // Unsubscribe from the custom event
  });
});
```
<br>

## 9. What is the purpose of the _ng-app directive_?

The **ng-app** directive in AngularJS acts as the application's starting point. It designates the root element of an Angular application and initializes the relevant application module.

### Core Functionality

- **Bootstrapping**: The directive is primarily responsible for kickstarting the Angular application.
- **Module Specification**: By specifying the ng-app directive, you connect it explicitly to an Angular module. Without this connection, the HTML for individual modules could become difficult to manage in larger applications.

### Angular Bootstrapping Process

1. **Load Angular**: At the outset, the browser loads the Angular framework.
2. **Identify Root Element**: Upon document loading, Angular locates the HTML element hosting the ng-app directive. This step is central to the bootstrapping process, signaling the commencement of Angular operations.
3. **Bootstrapping and Initialization**: Angular boots up the application, linking the identified root element to a specific module. The framework thematically organizes application components into modules.
4. **Compilation and Binding**: Angular carries out two major operations: Template Compilation - translating HTML into a set of instructions for the browser, and Data Binding - setting up the data binding context. This process associates the view with the logic managed inside the application modules.

5. **Rendering and Interaction**: The framework renders the prepared, efficient template and establishes interactivity.

### ng-app
The `ng-app` directive points out the root element of an AngularJS application. Once the directive is used on any element, AngularJS assumes control of that part. 

Here is an example:

```html
<!DOCTYPE html>
<html lang="en" ng-app="myApp">
<head>
    <!-- Head content -->
</head>
<body>
    <div ng-controller="myCtrl">
        {{ greeting }} World! <!-- The controller provides the value of 'greeting' -->
    </div>
    <script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.8.2/angular.min.js"></script>

    <!-- The app module and main controller are defined and connected here -->
    <script>
        angular.module('myApp', []).controller('myCtrl', function($scope) {
            $scope.greeting = 'Hello';
        });
    </script>
</body>
</html>
```
<br>

## 10. Explain how _ng-model directive_ works in AngularJS.

AngularJS's two-way data binding feature is powered by directives like `ng-model`. This mechanism synchronizes the Model and View layers, ensuring real-time updates.

### Core Functions

- **Registers User Input**: `ng-model` binds form elements to model data, using their value/input as a source.
- **Keeps Model Updated**: When the form element changes, the associated model updates automatically. This action is observable through `$watch` functions.
- **Syncs with the Backend**: When the model changes, `ng-model` triggers necessary operations, like form validity checks and web service updates, ensuring the backend remains in sync with the UI.

### Code Example: ng-model in Action

Here is the HTML:

```html
<div ng-app="myApp" ng-controller="myCtrl">
  <p>Name: <input type="text" ng-model="name"></p>
  <p>Your name is: {{name}}</p>
</div>
```

The JavaScript:

```javascript
var app = angular.module('myApp', []);
app.controller('myCtrl', function($scope) {
  $scope.name = "John Doe";
});
```
<br>

## 11. What is the role of _$scope_ in AngularJS?

The `$scope` in AngularJS represents the context within which **model**, **view**, and **controller** interact. This two-way bridge enables real-time data synchronization and is fundamental to AngularJS 1.x applications.

### Key Responsibilities

- **Data Share**: `$scope` stores both model data and references to functions or objects, making them available across controllers, directives, and views.

- **Watchers**: These special agents, maintained by the `$digest` cycle, monitor `$scope` properties for changes. When a change is detected, the associated actions are executed, ensuring dynamic view updates.

### Code Example: `$scope` and Data Binding in AngularJS

Here is the JavaScript code:

```javascript
// Define a controller
app.controller('MyController', function($scope) {
  // Initialize a property on scope
  $scope.username = 'John Doe';

  // Define a function to change the username
  $scope.changeUsername = function() {
    $scope.username = 'Jane Doe';
  };
});
```

And here's the HTML:

```html
<div ng-controller="MyController">
  <input type="text" ng-model="username">
  <button ng-click="changeUsername()">Change Name</button>
  <h2>Welcome, {{ username }}!</h2>
</div>
```

In this example, the `ng-model` directive establishes a data-binding between the text input and the `$scope.username`. As a result, any changes to the `username` property in the `MyController` reflect in real-time in the associated view.
<br>

## 12. How would you use _$rootScope_ in AngularJS?

While using `AngularJS`, you may benefit from the `$rootScope` in specific scenarios. However, **its use should generally be avoided** as it could lead to poorer code readability and harder debugging.

- Directives
- Cross-Component Communication
- Bootstrap Scope (Rare)
  - Scope Level switches in Angular: $rootScope  
  **Medium**: This answer can be delivered in 60-90 seconds.

### Code Example: Use of `$rootScope` in AngularJS

  ```html
  <div ng-app="myApp" ng-controller="myCtrl">
    <button ng-click="incrementCounter()">Increment</button>
    <my-custom-component></my-custom-component>
  </div>
  ```

  ```javascript
  var app = angular.module('myApp', []);

  app.controller('myCtrl', function($scope, $rootScope) {
    $scope.counter = 0;

    $scope.incrementCounter = function () {
      $scope.counter++;
      $rootScope.$broadcast('counterUpdated', $scope.counter);
    };
  });

  app.directive('myCustomComponent', function() {
    return {
      restrict: 'E',
      template: '<p>Counter from RootScope: {{rootCounter}}</p>',
      link: function(scope) {
        var counterListener = $rootScope.$on('counterUpdated', function(event, counter) {
            scope.rootCounter = counter;
        });

        scope.$on('$destroy', counterListener);  // cleanup to prevent memory leaks
      }
    };
  });
  ```
<br>

## 13. Can you explain the concept of _scope hierarchy_ in AngularJS?

**Scope hierarchy** refers to the **nesting of scopes** in an AngularJS application, resembling a tree structure. Each scope is responsible for a section of the DOM, and its **lifecycle** is closely linked with this DOM segment.

### Key Components of Scope Hierarchy in AngularJS

#### $rootScope

AngularJS has a **global scope** represented by `$rootScope`. It's primarily used to share data or trigger events across the application.

#### $scope

Each AngularJS controller **instantiates its own scope**. This local scope is a child of the `$rootScope` and serves as a **foundation for batching** related to its assigned view.

### Management of Scope Hierarchy

- **Segregation**: Controllers define specific boundaries within the DOM by creating new child scopes.
- **Inheritance**: Scopes inherit elements from their parent scopes, establishing a flow of data from the **top to the bottom** of the tree.

#### Data Flows

- **(Parent -> Child):** A change in a parent scope can affect all its child scopes, but the reverse is not true. This **unidirectional data flow** solidifies encapsulation and enhances control.

#### Using Prototypical Inheritance

- Parent elements assign objects or function references to their scopes. Child scopes that **don't redefine these elements** retain references to the parent's objects. However, when a child scope modifies a prototypically-linked object, it does create a new reference, leading to potential side effects.
<br>

## 14. What is the role of a _controller_ in AngularJS?

In AngularJS, a **controller** binds the view, typically an HTML page, with the model data. It plays a pivotal role in defining and initializing the data and business logic of a section of your application.

### Key Responsibilities

1. **Data Modeling**: The controller depicts defined data models using Scope objects. AngularJS uses **two-way data binding**, automatically keeping the model and view in sync.

2. **Business Logic**: The controller implements business logic, often in the form of functions, that governs the behavior of the model and its interaction with the view.

3. **Event Handling**: Controllers can respond to user-initiated events like clicks or form submissions. They also emit and handle custom events within the application for inter-component communication.

4. **Isolation**: AngularJS controllers offer varying degrees of scope isolation, ensuring modularity and preventing data or action overlaps in nested or sibling components.

5. **Initialization**: Controllers set up an initial state, execute start-up tasks, and prepare the contextual environment for the rest of the application.

### Controller Definition

A controller in AngularJS is defined using the `app.controller()` method, where `app` is your module:

#### Controller Definition Code

- JavaScript:

  ```javascript
  // Define the module
  var app = angular.module('myApp', []);

  // Create the controller
  app.controller('MyController', function($scope) {
    $scope.greeting = 'Hello, World!';
  });
  ```

  The controller is attached to a module using the `app.controller()` method, which takes the controller name and a function, called the controller's constructor.

- HTML:

  ```html
  <div ng-app="myApp" ng-controller="MyController">
    {{ greeting }}
  </div>
  ```

  In this example, `ng-controller="MyController"` attaches the defined controller to the section of HTML enclosed within the element.

### Data Binding

AngularJS uses **two-way data binding**. Any changes in the model, like variables defined in the `$scope` object, automatically reflect in the view, and vice versa.

#### Two-Way Binding Code

- JavaScript: 

  ```javascript
  app.controller('MyController', function($scope) {
    $scope.greeting = 'Hello, World!';
    $scope.updateGreeting = function(newGreeting) {
      $scope.greeting = newGreeting;
    };
  });
  ```

- HTML:

  ```html
  <div ng-app="myApp" ng-controller="MyController">
    <input type="text" ng-model="greeting">
    <button ng-click="updateGreeting('Goodbye!')">Update Greeting</button>
  </div>
  ```

  As seen in both the HTML and JavaScript snippets, any changes in the text input or via the button immediately update the `greeting` displayed.

### Scope & Dependency Injection

AngularJS controllers are constructed using the **Dependency Injection** design pattern. The `$scope` object is the nexus for data-binding and a medium through which the controller interacts with the view.

The definition of the controller function receives dependencies like `$scope`, and AngularJS maintains a registry of these dependencies, resolving them whenever the controller is initialized.
<br>

## 15. How do you define a _controller's method_ in AngularJS?

**AngularJS** uses the **MVC** architectural pattern. In this context, a controller contains methods that manipulate the **model**. In AngularJS, you can define and handle **Controller methods** through several techniques, including `$scope`.

### Define Methods on `$scope`

1. Define Method on `$scope` in the Controller:

```javascript
  app.controller("myController", function($scope) {
      $scope.greet = function(name) {
          alert("Hello, " + name + "!");
      };
  });
```

Here, `$scope.greet` is the method.

2. **Invoke in the View** with a suitable directive such as `ng-click`. You can then pass arguments, such as an `ng-model` directly.

```html
<!DOCTYPE html>
<html ng-app="myApp">

<head>
	<title>AngularJS Application</title>
	<script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.8.2/angular.min.js"></script>
</head>

<body>
	<div ng-controller="myController">
		<input type="text" ng-model="user.name" placeholder="Enter your name" />
		<button ng-click="greet(user.name)">Greet</button>
	</div>

	<script>
		var app = angular.module('myApp', []);
		app.controller('myController', function($scope) {
			$scope.greet = function(name) {
				alert("Hello, " + name + "!");
			};
		});
	</script>
</body>

</html>
```
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - AngularJS](https://devinterview.io/questions/web-and-mobile-development/angular-js-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

