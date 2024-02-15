# Top 100 Vue.js Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - Vue.js](https://devinterview.io/questions/web-and-mobile-development/vue-interview-questions)

<br>

## 1. What is _Vue.js_ and why would you use it?

**Vue.js** is a progressive JavaScript framework primarily used for building **user interfaces** and single-page applications. It is known for its adaptability, small file size, and the progressive learning curve.

### Key Features

#### Virtual DOM

Vue uses a **virtual DOM**, which is a lightweight copy of the actual DOM. This approach significantly speeds up UI updates by only rendering what has changed.

#### JS and Template Integration

Vue combines JavaScript and HTML-like templates for component structure. It then uses a **virtual DOM renderer** to update the actual DOM when data changes.

#### Two-Way Data Binding

Vue offers **two-way data binding** using the `v-model` directive, where changes in the UI instantly reflect in the data model and vice versa.

### Directives

Vue leverages **HTML directives** for actions, loops, and more. Examples include `v-if`, `v-for`, and `v-on`.

### Integration with Major Front-End Tools

Vue pairs remarkably well with a broad array of front-end tools including:

- **Webpack**: For asset bundling.
- **Babel**: To transpile ES6.
- **TypeScript**: For type-safety.
- **ESLint**: For code linting.

#### Let's look at the versioning and the stance on backward compatibility:

- **Versioning Scheme**: Vue adheres to Semantic Versioning, making it easier for users to understand when there are breaking changes.

- **Backward Compatibility**: While Vue places a strong emphasis on maintaining backward compatibility, it might introduce breaking changes in major releases. However, the Vue team assists users in the migration process, providing detailed documentation and migration paths.

### Code Example: Bidirectional Data Binding

Here is the Vue.js code:

```vue
<template>
  <div>
    <input v-model="message" placeholder="Edit me">
    <p>Message is: {{ message }}</p>
  </div>
</template>

<script>
export default {
  data() {
    return {
      message: 'Hello Vue!'
    }
    }
}
</script>
```
<br>

## 2. How do you set up a project with _Vue.js_?

To **initialize** a Vue.js project, utilize a **package manager** (npm or yarn) and the Vue CLI. This empowers you with tools like webpack and Babel, facilitates **incremental adoption**, and ensures best practices from the beginning.

### Steps for Project Setup

1. **Install Vue CLI**:  
   Run the specified command in a terminal.

   - **npm**: `npm install -g @vue/cli`
   - **Yarn**: `yarn global add @vue/cli`

2. **Create a Vue Project**:  
   Generate a new project using Vue CLI.

   ```bash
   vue create my-vue-app
   ```

   Select a default preset or **customize** the setup based on the project's requirements. After this, navigate into your project folder.

3. **Project Structure**:  
   Vue CLI sets up the project with a recommended structure including `public`, `src`, and various configuration files.

4. **Development Server**:  
   Boot up a local server to dynamically view project changes.

   ```bash
   npm run serve
   ```

5. **Production Build**:  
   Generate an optimized, minified, and production-ready build.

   ```bash
   npm run build
   ```

6. **Project Configuration**:  
   Manage project settings via `package.json` and `vue.config.js`. The latter aids in customizing build and development setups.

7. **Code Editor Integration**:  
   Use Vetur plugin for advanced Vue tooling support in VS Code. Other code editors might benefit from relevant extensions.

8. **Ready for Development**:  
   You are all set! Start building your Vue app.

### Configurable Features

The Vue CLI allows selection from various **application features** during the initial setup. This ensures a starting point that is tailored to the project’s needs.

For instance, you can span from the basic setup like:

- **Babel**: Employ the cutting-edge JavaScript features that are not completely standardized yet.
- **Linter**: Keep code quality in check with ESLint.

To the advanced setup, which might include features such as:

- **Vue Router**: For handling routing within the application.
- **Vuex**: For state management, specifically in larger applications.
- **CSS Pre-processors**: Integrate with Sass, Less, or Stylus for enhanced stylesheet capabilities.
- **Unit Testing**: Incorporate unit tests often alongside tools like Jest or Mocha.

### Vue CLI 5 Upgrades

Depending on the **Vue CLI version** you're working with, the provided commands and available features might differ. Always leverage the most recent Vue CLI version.

To ensure **smooth transitions** between versions and harness the newest Vue capabilities, precise guidance on version-specific features and upgrades can be found in the official Vue CLI documentation.
<br>

## 3. Can you explain the _Vue.js lifecycle hooks_?

Vue.js organizes **lifecycle events** into distinct stages, each linked to a specific lifecycle hook. These hooks enable developers to execute custom logic at crucial points during a component's life cycle.

### Vue.js Lifecycle Stages

1. **Initialization**: The component is being set up.
2. **Mounting**: The component is being added to the DOM.
3. **Update**: Data in the component undergoes changes.
4. **Destruction**: The component is being removed from the DOM.

### Lifecycle Hooks

1. **beforeCreate**: Occurs at the earliest step in initialization, before data observations and initializations are in place.
2. **created**: After the component's data and events are set up, this hook permits you to work with the component synchronously.
3. **beforeMount**: Just before the component is added to the DOM, execute logic.
4. **mounted**: The component is now in the DOM and is accessible for UI-related interactions.
5. **beforeUpdate**: Before the component re-renders, you can perform certain tasks here.
6. **updated**: The component has re-rendered and the DOM now reflects the most recent data.
7. **beforeDestroy**: Just before the component is destroyed, carry out cleanup tasks, such as uncoupling event listeners.
8. **destroyed**: Once the component is destroyed, perform any final teardown work like stopping intervals or closing connections.


### Code Example: Lifecycle Hooks

Here is the **JavaScript** code:

```javascript
new Vue({
  el: '#app',
  data: {
    message: 'Hello Vue!'
  },
  beforeCreate: function() {
    console.log('Before creation - The message is: ' + this.message);
  },
  created: function() {
    console.log('Component created');
  },
  beforeMount: function() {
    console.log('Before mounting');
  },
  mounted: function() {
    console.log('Component mounted');
  },
  beforeUpdate: function() {
    console.log('Before update - The message is: ' + this.message);
  },
  updated: function() {
    console.log('Component updated');
  },
  beforeDestroy: function() {
    console.log('Before component destruction');
  },
  destroyed: function() {
    console.log('Component destroyed');
  },
  methods: {
    updateMessage: function() {
      this.message = 'Updated message';
    },
    destroyComponent: function() {
      this.$destroy();
    }
  }
});
```
<br>

## 4. What are _Vue.js_ _components_ and how do you use them?

**Vue.js components** are reusable, self-contained elements that consist of three main parts: a **template**, **script**, and **style**. These encapsulated building blocks enable a clear separation of concerns within your web application. Efficiently constructed components can significantly contribute to the maintainability and scalability of your Vue app.

In component-based **development**, a larger application is broken down into smaller, interconnected units—the components. Each component focuses on one specific task, such as presenting the user interface for an email, a customer, or a form. This modular structure streamlines the development process and enhances code reusability.

### Anatomy of a Vue Component

Here's the step by step:

#### 1. Template

The template is the visual representation of a component, typically constructed using HTML. You can embed Vue directives, attributes, and expressions, allowing for dynamic rendering based on component data and logic.

##### Example: Vue Component Template

```html
<!-- vue-component.vue -->
<template>
  <div>
    <h1>{{ title }}</h1>
    <p v-show="isContentVisible">{{ content }}</p>
    <button @click="toggleContentVisibility">Toggle Content</button>
  </div>
</template>
```

#### 2. Script

The script provides the component's behavior and data logic. It defines the Vue object options such as data, methods, computed properties, and watchers.

##### Example: Vue Component Script

```html
<!-- vue-component.vue -->
<script>
export default {
  data() {
    return {
      title: 'Component Title',
      content: 'Dynamic content here!',
      isContentVisible: true,
    };
  },
  methods: {
    toggleContentVisibility() {
      this.isContentVisible = !this.isContentVisible;
    },
  },
};
</script>
```

#### 3. Style

The style section allows for component-specific CSS (or other pre-processed styles like SASS or LESS) encapsulation, preventing style bleed and conflicts across different parts of your application.

##### Example: Vue Component Style

```html
<!-- vue-component.vue -->
<style scoped>
h1 {
  font-size: 1.5em;
  color: #333;
}
p {
  color: darkslategray;
  font-size: 1em;
}
</style>
```

### Registering Components

There are multiple ways to register Vue components:

1. **Globally Registered Components**: Perfect for app-wide and frequent usage. Quickly added using Vue's `Vue.component` method.

   Example:

    ```javascript
    Vue.component('my-global-component', {
      // Component options
    });
    ```

2. **Locally Registered Components**: Best for encapsulated, single-file components. Import and declare in a parent component.

   Example:

   ```javascript
   import MyBaseComponent from './components/MyBaseComponent.vue';

   export default {
     components: {
       'my-local-base-component': MyBaseComponent,
     },
   };
   ```

3. **Automatically Imported Components**: Offers on-the-fly component import. Ideal for tree-shaking and smaller bundle sizes.

   Example:

   ```javascript
   async function loadGloballyRegisteredComponents() {
     const { MyGlobalComponent } = await import('./components/MyGlobalComponent.vue');
   }
   ```

4. **Component Registration via CLI or Build Tools**: Benefit from Vue's build tools and more streamlined workflows.

   Example:

    ```javascript
    import MyLocalComponent from './components/MyLocalComponent.vue';

    // In the build setup
    Vue.component('my-local-component', MyLocalComponent);
    ```

### Practical Component-Driven Development

1. **Code Reusability**: A well-designed and atomic component library ensures code efficiency and reusability.

2. **Collaborative Development**: Parallel development and team collaboration are facilitated as each team member can work on different components independently.

3. **Clear Code Structure**: Component-based development promotes a clear, self-documenting code structure, making onboarding and maintenance more manageable.

4. **Improved Code Quality**: Smaller, isolated components are easier to test, reducing the potential for bugs or unexpected behavior.

5. **Enhanced Project Scalability**: As your app grows, you can integrate new features and scale more seamlessly with the help of components.
<br>

## 5. How do you _bind data_ to the _view_ in _Vue.js_?

In **Vue.js**, **Data Binding** establishes a connection between the model and the view, ensuring that changes in one are reflected in the other. Vue offers three primary binding types:

- **One-time**: Ideal for scenarios where data doesn't change, and you want to initialize the view with it. Using this option can increase performance.
- **One-way**: Data changes are propagated from the model to the view. It's beneficial in ensuring that the model is the single source of truth, simplifying the understanding of your application flow.
- **Two-way**: This type of binding relationship is more dynamic, enabling changes made in either the view or the model to be reflected in the opposite entity.

### Data Binding Syntax

- **Text**: `{{ }}`
- **HTML Attributes**: `:attrName="dataProp"` or `v-bind:attrName="dataProp"`
- **Element Visibility**: `v-show="condition"` or `v-if="condition"`
- **List Rendering**: `v-for`
- **Event Listening**: `@event="handlerMethod"` or `v-on:event="handlerMethod"`

### Key Directives

- **v-bind**: Binds data to an element.
- **v-model**: Establishes two-way data binding for form inputs.
- **v-on**: Listens to events on elements and triggers event handler methods defined in the component.

### Code Example: Vue.js Binding Types

Here is the Vue data

```javascript
data() {
  return {
    message: "Hello, World!",
    isButtonDisabled: false,
    textColor: "red",
    users: [
      { id: 1, name: "Alice" },
      { id: 2, name: "Bob" }
    ],
    userProfile: {
      id: 1,
      name: "Alice",
      email: "alice@example.com"
    }
  };
},
methods: {
  toggleButton: function() {
    this.isButtonDisabled = !this.isButtonDisabled;
  }
}
```

Here is the Vue HTML

```html
<p>{{ message }}</p>
<button :disabled="isButtonDisabled" @click="toggleButton">Toggle</button>
<p :style="{ color: textColor }">Text in dynamic color</p>
<ul>
  <li v-for="user in users" :key="user.id">{{ user.name }}</li>
</ul>
<input v-model="userProfile.name" type="text">
```
<br>

## 6. What is a _Vue instance_ and its purpose?

The **Vue Instance** serves as the entry point for Vue applications, acting as the core engine that controls and coordinates the entire application.

### Core Functions

The Vue Instance offers key functions, including:

#### Data Management

- **Data**: Houses the application's state, organized as key-value pairs. Changes to data are automatically reflected across the application.
- **Computed Properties**: Derive new data from existing app state, ensuring reactivity and performance gains.
- **Watchers**: Track specific data points (or computed properties) for changes, enabling additional actions or logic.

#### Lifecycle Management

- **Lifecycle Hooks**: Provides customized behaviors at key application stages, such as when a component is created, mounted, updated, or destroyed.

#### DOM Interaction

- **Directives**: Special attributes that modify the DOM when linked by Vue through its templating engine.
- **Methods**: Functions that can alter app state or execute actions in response to user or system events.

#### Component Composition

- **Components**: Encapsulate sections of the UI and logic, enabling modularity and reusability within applications. The Vue Instance serves as the orchestrator that brings these components together.
<br>

## 7. Explain the _virtual DOM_ in _Vue.js_.

The **Virtual DOM** in Vue.js acts as an intermediary layer between the actual DOM and the Vue.js component's in-memory representation. Vue.js uses a **differencing algorithm** known as Dynamic Update to match the Virtual DOM to the actual DOM, leading to minimal, efficient, and real-time updates.

### Key Concepts

1. **Declarative UI**: Vue.js employs declarative UI, allowing developers to state "what" should be displayed, and abstracts the "how" regarding DOM updates, using the Virtual DOM as a tool for efficiency.

2. **Retention of State**: The Virtual DOM serves as a memory bank, preserving component states and rendering them to the actual DOM when necessary.

3. **Batching**: Vue.js strategically groups multiple DOM updates from a single "tick" in a queue for optimum performance, regularly updating the real DOM only as needed based on queue contents.

4. **Performance Optimization**: Through the efficient use of the Virtual DOM for DOM updates, Vue.js minimizes redundant re-renders and ensures that the cost of updates is proportional to the actual changes.

5. **Cross-Platform Support**: As both the real DOM and the Virtual DOM have unified interfaces, components remain platform-independent, and developers can transition between different platforms seamlessly.

### Advantages

- **Performance**: Vue.js leverages the Virtual DOM to optimize and streamline DOM updates, resulting in faster user interfaces.
   
- **Ease of Use**: By abstracting the complexities associated with direct DOM manipulation, Vue.js empowers developers to focus on the application's logic and presentation layer.

### Code Example: Virtual DOM

Here is the JavaScript code:

```javascript
// Initial Array
const numbers = [1, 2, 3, 4, 5];

// Changing Array
numbers[2] = 7;

// Ensured updated Array
console.log(numbers);  // [1, 2, 7, 4, 5]
```

In this example, even though we update an element in the `numbers` array directly without using any Vue.js reactivity feature, the console log still prints out the modified array. This behavior stems from Vue.js reactivity and its reliance on the Virtual DOM for enhanced efficiency.
<br>

## 8. What are _directives_ in _Vue.js_? Can you give examples?

**Vue.js** directives are special attributes that manipulate the Document Object Model (DOM) based on the bound values.

Despite closely resembling HTML attributes, **directives** are distinct and recognized by the **v-** prefix, e.g., v-bind. They are fundamental for data-driven applications and ensure the View remains synced with the ViewModel.

### Core Directives

#### v-bind: Dynamically Setting Attributes

The `v-bind` directive syncs an HTML attribute with a Vue.js expression. This is especially useful for setting dynamic classes, styles, and attributes, including **inputs**.

**Reactive Class Example:**

```vue
<template>
  <div v-bind:class="{ active: isActive }"></div>
</template>
```

#### v-if and v-show: Conditionally Rendering Elements

Both of these directives control an element's visibility based on a Boolean.

- `v-if` completely removes the element from the DOM when its attached value is `false`.
- `v-show` toggles the element's CSS `display` property.

**Use Case Distinction**: Employ `v-if` for conditional rendering where visibility alterations are infrequent, while `v-show` is suitable for toggling visibility in response to user actions.

```vue
<template>
  <div v-if="isDisplayed">Displayed on True</div>
  <div v-show="isOpen">Always Shown</div>
</template>
```

#### v-for: Generating Repetitive Elements

The `v-for` directive iterates over a dataset and replicates the current DOM element for every item in the collection or array.

**Dynamic List Example:**

```vue
<template>
  <ul>
    <li v-for="item in items" :key="item.id">{{ item.name }}</li>
  </ul>
</template>

<script>
export default {
  data() {
    return {
      items: [
        { id: 1, name: 'First Item' },
        { id: 2, name: 'Second Item' }
      ]
    };
  }
};
</script>
```

#### v-on: Event Handling

The `v-on` directive reacts to DOM events and triggers methods or updates data accordingly.

**Event Handling Example:**

```vue
<template>
  <button v-on:click="toggleState">Toggle State</button>
</template>

<script>
export default {
  data() {
    return {
      isActive: true
    };
  },
  methods: {
    toggleState() {
      this.isActive = !this.isActive;
    }
  }
};
</script>
```

#### v-model: Two-Way Data Binding

This directive links form input elements to Vue.js data, ensuring that any changes in the View update the underlying data and vice versa.

**Two-Way Binding Example:**

```vue
<template>
  <input type="text" v-model="inputValue" />
  <p>{{ inputValue }}</p>
</template>

<script>
export default {
  data() {
    return {
      inputValue: ''
    };
  }
};
</script>
```
<br>

## 9. How do you handle user _inputs_ and _form submissions_ in _Vue.js_?

**Vue.js** simplifies user interactions and form handling through its reactive two-way data binding, utility directives, and validation libraries. Let's explore these features further.

### Two-Way Data Binding

Vue.js's **two-way data binding** ensures that changes in input fields are reflected in corresponding data properties and vice versa.

For two-way binding, use the `v-model` directive:

```html
<input v-model="name" />
<p>{{ name }}</p>
```

In this example, `name` is the corresponding data property, and its value is updated as the user types in the `input` field.

### Handling Event

Vue.js provides the **v-on** directive to handle user-triggered events, like `click`, `input`, or `submit`. 

For example:
```html
<button v-on:click="handleClick">Click Me</button>
```

Here, the `handleClick` method is called when the button is clicked.

### Customize Event Handling

You can combine `v-on` with the `.stop`, `.prevent`, and `.self` modifiers to fine-tune event behavior. For instance:

- `.stop`: Prevent event propagation.
- `.prevent`: Call `event.preventDefault()`.
- `.self`: Trigger only if the event initiates from the element, not its children.

**Example**:
```html
<button v-on:click.stop="log">Do not propagate</button>
```

### Form Submission

To handle form submissions and their associated actions, such as making an API call, use the **v-on:submit** directive.

```html
<form v-on:submit.prevent="submitForm">
  <input type="text" v-model="name" />
  <button type="submit">Submit</button>
</form>
```

In this example, `submitForm` is triggered when the form is submitted, preventing the default form action.

### Async Form Submission

If your form submission involves asynchronous operations, like calling an API, you can use `async` methods with `await` within `v-on:submit` handlers:

```html
<form v-on:submit.prevent="submitForm">
  <!-- ... form inputs ... -->
  <button type="submit">Submit</button>
</form>

<script>
  methods: {
    async submitForm() {
        // perform async operations here
    }
  }
</script>
```

### Input Validation with Watchers

You can ensure user inputs meet specific criteria using **watchers**:

```javascript
data() {
  return {
    username: ''
  };
},
watch: {
  username(newVal) {
    if (newVal.length < 3) {
      console.log('Username must be at least 3 characters long');
    }
  }
}
```

In this scenario, `submitForm` will only be called if the condition in the `watch` block is met. If the condition fails, the submission is prevented.

### Extended Input Components

For form-specific needs, Vue provides dedicated components like `v-radio`, `v-checkbox`, and `v-select`:

```html
<v-form>
    <v-radio v-model="gender" :options="['male', 'female']">Gender:</v-radio>
    <v-checkbox v-model="agree">I accept the terms and conditions</v-checkbox>
    <v-checkbox v-model="newsletter">Subscribe to our newsletter</v-checkbox>
    <v-button @click="submit">Submit</v-button>
</v-form>
```

Each component performs both form **input submission** and **validation**.
<br>

## 10. Discuss the _methods_ property in a _Vue component_.

**The `methods` Property** in Vue.js lets you define and use custom methods within a component. This separates component logic into manageable pieces for improved reusability and maintainability.

### Key Features

1. **Method Definition**: Declare named methods within the `methods` object.

2. **Access to Data and Lifecycle**: Methods can directly interact with local data and are aware of the component's lifecycle.

3. **Event Handling**: Commonly used for event callbacks, serving as an alternative to inline event-handling functions.

4. **Code Reusability**: Methods support modularity and code reusability within the component.

5. **Performance Considerations**: If you need to rerender only portions of the UI, you can use **Vue's reactivity system**. The methods referred to within the template are reactive.

6. **Scope**: Methods are accessible within the component's context, keeping them separate from global or parent component methods.

### Code Example: Using `methods`

Here is the Vue JS code:

```vue
<template>
  <div>
    <button @click="incrementCounter">Increment</button>
    <p>Counter value: {{ counter }}</p>
  </div>
</template>

<script>
export default {
  data() {
    return {
      counter: 0
    };
  },
  methods: {
    incrementCounter() {
      this.counter++;
    }
  }
};
</script>
```
<br>

## 11. Explain the difference between `v-bind` and `v-model`.

**Vue.js**, while it has concise syntax, uses varied directives in its **templating** engines for distinct functions.

### The Essentials

- **v-bind**: Links variables from the model to views.
- **v-model**: Directs two-way binding, enabling real-time updates between inputs and the application's data model.

### Supporting Aspects

- **Scope**: V-bind is ideally used for one-way data flow, from the model to the view. V-Model incorporates both one-way and two-way data flow, **updating both model and view** as data changes.

- **Compatible Elements:** V-Model primarily targets form elements, allowing for user input tracking.

  ```html
  <!-- Text Input -->
  <input type="text" v-model="message">
  ```

  In contrast, V-Bind is versatile and is compatible with a wide array of HTML attributes.

  ```html
  <!-- Linking Attribute -->
  <a v-bind:href="url">Vue.js Website</a>
  ```

### Best Use-Cases

- **V-Bind**:  Ideal for unidirectional data flow and non-form elements.

  ```html
  <!-- One-Way List Binding -->
  <ul>
    <li v-for="item in items" v-bind:key="item.id">
      {{ item.name }}
    </li>
  </ul>
  ```

- **V-Model**: Suitable for effortlessly managing data in forms.

  ```html
  <!-- Checkboxes with Data Binding -->
  <input type="checkbox" id="checkbox" v-model="checked">
  ```

### Code Example: V-Bind and V-Model

Here is the HTML:

```html
<div id="app">
  <p>{{ message }}</p>
  <input type="text" v-model="message">
  <a v-bind:href="url">{{ linkText }}</a>
</div>
```

Here is the Vue.js code:

```javascript
new Vue({
  el: '#app',
  data: {
    message: 'Hello, Vue!',
    url: 'https://vuejs.org/',
    linkText: 'Vue.js Official Website'
  }
});
```
<br>

## 12. What are _computed properties_ and how are they used?

**Computed properties** in Vue.js are derived data elements that automatically update whenever their "watched" dependencies change.

They eliminate the need for manual tracking and updating of data, offering a simpler and more efficient data management tool.

### Core Benefits

- **Caching**: Computed values are cached based on their dependencies. They recalculate only when a dependency changes, enhancing performance.

- **Simplicity**: Create and use computed properties with the same seamless syntax as other data properties, like `data` and `methods`.

### When to Use Computed Properties

- **Derivative Data**: It's best used for data that is derived from other data, rather than being independent.

- **Data Aggregation**: For processes like filtering, sorting, and data aggregations like sums or averages.

### Practical Applications

- **Display Formatting**: For dynamic formatting, such as concealing sensitive data or changing number representations.

- **Dependency Management**: To handle complex data relationships without manual tracking.

- **User Interactions**: For managing user inputs and dynamic interface updates, such as marking all items in a list as "done."

### Performance Considerations

While computed properties can aid performance through caching, be cautious with extremely complex and time-consuming computations. These can still introduce bottlenecks. In such cases, consider using techniques like debouncing or minimizing the use of relatively expensive computed properties.

### Code Example: A Basic Computed Property

Here is the Vue.js code:

```vue
<template>
  <div>
    <p>{{ reversedMessage }}</p>
    <input v-model="message" />
  </div>
</template>

<script>
export default {
  data() {
    return {
      message: 'Hello, Vue!',
    };
  },
  computed: {
    reversedMessage() {
      return this.message.split('').reverse().join('');
    },
  },
};
</script>
```
<br>

## 13. Can you discuss the difference between _computed properties_ and _methods_?

- **Computed Properties**: These are like data fields that are derived from other data **based on specific logic**. You define computed properties using the `computed` option in a Vue component.
- **Methods**: They are like functions, meaning they are **called on demand**. You define methods using the `methods` option in a Vue component.

#### Efficiency

- **Computed Properties**: The natural caching mechanism in Vue ensures that if the data properties on which the computed property depends haven't changed, the computed property doesn't recompute.
- **Methods**: They execute every time they are needed, irrespective of whether the underlying data has changed or not.

#### Use in Templates

- **Computed Properties**: They can be used in Vue templates very much like data properties. Vue knows when a computed property needs to be recomputed and takes care of it automatically.
- **Methods**: Their use in templates is often not ideal because they can lead to unnecessary re-rendering, especially if the method involves some heavy computation.

#### Data Reusability & Data Source

- **Computed Properties**: They are perfect when you want to use a processed or derived version of the data in multiple places within the template or component. They abstract and cache the data transformation for you.
- **Methods**: Useful when you need to perform an operation just once or need to trigger an action on an event, for example a button press.

#### Common Examples

- **Computed Properties**: Calculating the total price of items in a shopping cart, applying a filter to a list, or sorting an array.
- **Methods**: Handling form submissions, initiating data fetch or AJAX requests, or any kind of action that needs to be explicitly triggered.
<br>

## 14. What are _watchers_ in _Vue.js_?

In Vue.js, **watchers are key features** that track data changes and enable developers to execute specific logic or tasks in response to those changes.

These watchers are defined inside Vue components and serve as an indispensable tool for building reactive behavior. They work in tandem with the Vue instance to monitor specific data properties, methods, or even component options.

### How Are Watchers Defined?

Developers can set up watchers in two primary ways:

1. **Declared in the Watch Option**  
    This option is preferred for more complex watcher functions that respond to multiple data sources or need additional parameters.

    ```javascript
    export default {
      data() {
        return {
          items: [],
          totalCount: 0
        };
      },
      watch: {
        items: {
          deep: true,  // Detect changes at nested levels, if items are objects or arrays.
          immediate: true, // Call the handler immediately once the component is created.
          handler(newItems, oldItems) {
            // Update the total count on item changes.
            this.totalCount = newItems.length;
          }
        }
      }
    };
    ```

2. **Using a Method to Bypass Limitations**  
    By using a more conventional method, Vue developers can watch computed properties, for example. They can also handle multiple related data sources if required.

    ```javascript
    export default {
      data() {
        return {
          firstName: '',
          lastName: ''.
          fullName: ''
        }
      },
      watch: {
        firstName(val) {
          this.fullName = val + ' ' + this.lastName;
        },
        lastName(val) {
          this.fullName = this.firstName + ' ' + val;
        }
      }
    };
    ```

### Performance Considerations

While watchers are highly effective mechanisms for managing reactivity, unnecessary or excessive watchers can lead to performance bottlenecks in bigger Vue.js projects.

Developers can use several techniques to optimize watchers and minimize their impact:

-   **Leverage Computed Properties**: Instead of watching the same data value for changes, use computed properties to derive values and update automatically when dependent data changes.
-   **Batch Updates When Possible**: If multiple changes to a data source would cause many watcher updates, Vue.js can batch these updates. But be sure to verify if the app architecture supports this behavior.
-   **Reduce Unnecessary Updates**: Tools like `Immediate: true` in watchers ensure the handler is called once when the component loads, but use this option only when necessary.
<br>

## 15. How do you _bind inline styles_ in a _Vue template_?

In Vue.js, you can **bind inline styles to an element** through several methods like templates with `v-bind`, dynamic class bindings, and JS object-style declarations.

### Inline Style Binding Approaches

1. **Using the `:style` Directive**: You can define an object with CSS styles and then bind it to the element. Each key-value pair represents a style property and its value.

    ```vue
    <template>
      <div :style="myStyles">Styled Text</div>
    </template>
    
    <script>
    export default {
      data() {
        return {
          myStyles: {
            color: 'blue',
            'font-weight': 'bold',
            fontFamily: 'Arial, sans-serif'
          }
        };
      }
    };
    </script>
    ```

2. **Binding Inline Styles Directly**: This approach is useful when the style values need to be dynamic. However, it's less modular than using a style object.

    ```vue
    <template>
        <div :style="{ color: textColor, fontSize: textSize + 'px'}">Dynamic Text</div>
    </template>
    
    <script>
    export default {
      data() {
        return {
          textColor: 'green',
          textSize: 24
        };
      }
    };
    </script>
    ```

3. **CSS Modules**: Places **styles in a separate file, processes them with Vue Loader, and requires modular scoped styles in components**. This approach provides isolation and reusability.

    1. Create a `styles.module.css` file

        ```css
        .styledText {
           color: blue;
           font-weight: bold;
           font-family: 'Arial, sans-serif';
        }
        ```

    2. Use the CSS module in the component

        ```vue
        <template>
           <div :class="$style.styledText">Styled Text</div>
        </template>
        
        <script>
        export default {
           // ...rest of the component
        };
        </script>
        ```

### Benefits of Different Approaches

- **Code Reusability**: The object approach with `:style` and the modular styles using CSS modules are great for style management and reusability.
- **Dynamic Styling**: Binding styles directly with `:style` and using JS objects is suitable for dynamic style changes.
- **Readability and Maintainability**: Each approach has its own visual language. Use the one that's most readable and maintainable for your team.

### Additional Recommendations

- **Multiple Styles**: Avoid grouping multiple styles under one key if you expect them to change independently.

    ```vue
    <!-- Where feasible, avoid -->
    <div :style="{color: dynamicColor, fontSize: dynamicSize + 'px'}">
    ```

- **Modular Styles**: If you're using the object approach with `:style`, consider managing the styles in a separate object for better readability and maintainability.

    ```vue
    <template>
      <div :style="mySeparateStylesObject">Styled Text</div>
    </template>

    <script>
    export default {
      data() {
        return {
          mySeparateStylesObject: {
            color: 'blue',
            fontSize: '22px'
          }
        };
      }
    };
    </script>
    ```
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - Vue.js](https://devinterview.io/questions/web-and-mobile-development/vue-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

