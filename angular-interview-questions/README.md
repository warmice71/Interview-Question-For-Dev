# 100 Core Angular Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - Angular](https://devinterview.io/questions/web-and-mobile-development/angular-interview-questions)

<br>

## 1. What is _Angular_ and what are its key features?

**Angular** is a robust, structural, **TypeScript-based** open-source front-end web application platform. It is especially well-suited for creating **Single Page Applications** (SPAs) and maintains a rich ecosystem of libraries, extensions, and tools. 

### Core Features

- **Modularity**: Through **NG Modules**, different parts of an Angular application can be structured and managed as distinct and cohesive units.

- **Component-based Architecture**: Angular is **built around components**, fostering a modular, reusable, and testable design.

- **Directives**: These markers on a DOM element instruct Angular to attach a particular kind of behavior to that element or even transform the element and its children.

- **Data Binding**: Angular offers several types of data binding, enabling live management of data across the model, view, and components.

- **Dependency Injection (DI)**: Angular has its own DI framework, which makes it possible to get services and share data across components.

- **Templates**: Enhanced HTML templates in Angular lead to seamless incorporation of specialized constructs, like directives and data binding.

- **Model-driven Forms**: Angular approaches forms with modularity through custom NG modules, while also employing two-way data binding.

- **Template-driven Forms**: Here, the emphasis is on minimizing the need for explicit model management on the component through directives that can observe and manage forms.

- **Inter-component Communications**: Angular supports several methods for components to interact and share data, including Input, Output, ViewChild, and services based mechanisms.

- **Asynchronous Operations**: Built on top of Promises, Observables offer a more flexible and powerful way to deal with sequences of events, HTTP responses, and more.

- **Directives**: Angular comes with several built-in directives for management of the DOM, such as `*ngIf`, `*ngFor`, and `*ngSwitch`.
- **Advanced Routing**: Angular's powerful Router employs configurable routes, location services, and guards to navigate between views seamlessly.

- **Provisioning**: The DI system in Angular centralizes the management of instances of services, ensuring singletons where necessary and other strategies based on the provider settings.
<br>

## 2. Explain _data-binding_ in Angular. What are the different types?

**Data Binding** in Angular represents the communication between a **component** and the **DOM**. It ensures that the model and view are synchronized. Angular offers different types of data binding to cater to varied application requirements.

### Key Types of Data Binding

1. **One-Way Data Binding**
    - Data flows in a single direction from the **component** to the **DOM** or **vice versa**.
    - **Example**: Interpolation, Property Binding, Event Binding.
  
2. **Two-Way Data Binding**
    - Enables  bi-directional data flow, offering real-time synchronization  between the **component** and the **DOM**.
    - **Syntax**: Utilize `[(ngModel)]` or `[( )]` for **attribute** binding. `[(ngModel)]` is specifically designed for forms, necessitating the `FormsModule` for integration.
  
3. **One-Way from Source**
    **One-way** binding ensures that changes in the source will dictate whether the destination in the DOM is updated or not.

    - **Example**: Style or Attribute Binding.

4. **One-Time Binding**

    One-time binding involves a single transfer of data from source to target without ongoing synchronization. This is useful when the data doesn't change and you don't want the overhead of continuous checks.

    - **For Efficiency**: Use in scenarios with data that's static or changes infrequently.


### Best Practices for Data Binding

- **Simplicity Breeds Clarity**: Limit two-way and one-time bindings to clear and justified contexts.
  
- **Temporal Precision**: Use one-time bindings when data is static.
  
- **Systematic Updates**: Employ strategies that maintain data integrity, such as `ChangeDetectionStrategy.OnPush`, and manually triggering `ChangeDetectorRef`.

- **Performance Considerations**: Understand the potential performance implications of each data binding type and use them judiciously.

### Code Example: Types of Data Binding

Here is the TypeScript code:

```typescript
import { Component, ChangeDetectionStrategy, ChangeDetectorRef } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AppComponent {
  public message = "Initial message";
  public btnContent = "Disable";

  constructor(private cdr: ChangeDetectorRef) {}

  updateMessage() {
    this.message = new Date().toTimeString();
    // Manually trigger Change Detection
    this.cdr.detectChanges();
  }

  toggleBtn() {
    this.btnContent = this.btnContent === "Disable" ? "Enable" : "Disable";
  }
}
```
<br>

## 3. Describe the _Angular application architecture_.

The **Angular application architecture** adheres to the principles of modularity, components, and a unidirectional data flow. It includes **four foundational elements**: modules, components, services, and the routing module.

### Key Concepts

- **Modules**: Serve as containers for a cohesive set of functionalities within an app. Angular uses dependency injection to manage the modules and their components.

- **Components**: Represent the building blocks of the app. Each component is a small, self-contained unit, responsible for both UI and logic.

- **Services**: Provide specialized functionality throughout the app. They are singletons and can be injected into any component or another service.

- **Routing Module**: Manages navigation between application views.

### Data Flow Mechanism: One-way Binding

- **@Input()**: Data flows into a component from its parent using this decorator.
- **@Output()**: Components emit events to notify the parent through this decorator.

### App Structure

- **Root Module**: Starting point of an Angular app. Coordinates and configures other modules, and defines the root component.

- **Feature Modules**: Unique to Angular, they group functionality and components based on the specific feature they provide. Feature modules can be eagerly or lazily loaded.

### Code Example: Root Module

Here is the Angular Code:

```typescript
// app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { HomeComponent } from './home.component';
import { ContactComponent } from './contact.component';
import { AppRoutingModule } from './app-routing.module';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    ContactComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
```
<br>

## 4. What is a _component_ in Angular and how is it used?

In Angular, a **component** represents a logical UI element that defines a part of the user interface. It consists of a TypeScript class that holds the component's data and logic, and an HTML file that defines the view, along with a CSS file for styling. Components can nest inside each other to form a component tree, and they often communicate with each other using **inputs**, **outputs**, **services**, and **observables**.

### Key Component Parts

- **Class**: Represents the component's behavior and data using TypeScript. It may include properties, methods, lifecycles, and decorators.
- **Template**: Specifies the UI structure using HTML, often integrated with Angular directives and data binding.
- **Styles**: Uses CSS to define the component's visual appearance. Can be scope-limited to the component.

### Core Concepts

- **Component Tree**: Refers to the hierarchical relationship among components where a top-level component can have child components, and these children can further have their own children, creating a tree structure.
- **Data Binding**: Establishes a connection between the component's data (the model) and the template, enabling synchronization.

### Unique Features

#### Component-Scoped Styles

Angular lets you define styles specific to a component, ensuring they don't affect other parts of the application. This scoping is achieved using CSS Encapsulation techniques such as emulation of Shadow DOM or generated, unique attribute selectors.

#### Modular Design for UI Elements

Components offer a modular way to design and develop user interface elements. Each component is self-contained, focused on a single responsibility, and can encapsulate its HTML, styles, and related logic.

#### Reusability: 

Via elements like `@Input()` and `@Output()`, a component's functionality and data can be exposed, making its task more adaptable, reusable, and modular within the application.

#### Clear Separation of Concerns:

The segregation of a component's class (handling of logic and data) from its template (dealing with the presentation) ensures a divide between the application's UI and its underlying functional structure.

### Code Example: Basic Angular Component

Here is the Angular component:

```typescript
import { Component } from '@angular/core';

@Component({  // The @Component Decorator
  selector: 'app-hello',  // CSS Selector - This component can be used as <app-hello></app-hello> in HTML
  template: '<h2>Hello, {{name}}!</h2>',  // The component's template
  styles: ['h2 { color: green; }']  // The component's styles - using a simple inline array
})
export class HelloComponent {  // The component's class, named HelloComponent
  name = 'User';  // A public property, accessible in the template

  // A method that can be called from the template
  setName(newName: string): void {
    this.name = newName;
  }
  
  constructor() {
    // Constructor logic, executed when an instance of the component is created.
  }
}
```
<br>

## 5. What are _directives_ in Angular and can you name a few commonly used ones?

**Directives** in Angular are powerful tools that allow you to extend HTML vocabulary. They attach special behaviors to elements or **transform** DOM structure and View elements in several unique ways.

### Types of Directives

1. **Component Directives**: These are the most common directives. They define components responsible for handling views and logic.

2. **Attribute Directives**: These modify the behavior and appearance of DOM elements. They are essentially markers on a DOM element that invoke some JavaScript logic.

3. **Structural Directives**: These are a special type of directives that modify the DOM layout by adding, removing, or manipulating elements.

### Commonly Used Directives

1. **ngIf**: This Angular structural directive conditionally adds or **removes** elements from the DOM tree.
  
2. **ngFor**: Useful for iterating through `arrays` and iterating over `object` properties. It dynamically renders elements based on the **collection** it's given.

3. **ngStyle**: This attribute directive allows for inline CSS styling based on template expressions.

4. **ngClass**: This attribute directive dynamically adds and **removes** classes from elements based on template expressions.

5. **ngModel**: This directive establishes **two-way data binding** between input elements and component data. It's commonly used in forms.

6. **ngSwitch**: This set of **structural** directives is like an enhanced version of `ngIf` by providing `else` and `default` matching functionalities.

### Code Example: ngFor

Here is the Angular code:

```typescript
@Component({
  selector: 'app-item-list',
  template: `
    <ul>
      <li *ngFor="let item of items">{{ item.name }}</li>
    </ul>
  `
})
export class ItemListComponent {
  items: any[] = [{ name: 'Item 1' }, { name: 'Item 2' }];
}
```

In the HTML template, the `ngFor` directive iterates over the `items` array and renders an `li` element for each item.

### Why Use Directives?

Directives provide a **declarative approach** to organizing your code, making it more intuitive and easier to maintain, with a clear separation of UI and application logic.
<br>

## 6. How do you create a _service_ in Angular and why would you use one?

**Services** are instrumental in Angular for finer architectural design and sharing common functionality across components.

Angular automatically **injects** a service when a component or another service needs it. This mechanism fosters the "Don't Repeat Yourself" (DRY) principle, leading to more modular, maintainable, and testable code.

### Service Creation

You can  create a service in Angular using either of these methods:

1. **CLI**: Use the Angular CLI to generate a service.
    ```bash
    ng generate service my-service
    ```

2. **Manual**: Create a `.ts` file for the service and define the class.

### Using the Service

1. **Service Registration**:

    - **Module**: Link the service to a specific module by adding it to the `providers` array in `@NgModule`.

    ```typescript
    @NgModule({
      declarations: [
        MyComponent
      ],
      providers: [MyService],
      imports: [CommonModule]
    })
    ```

    - **Dependency Injection Tree**: Use a tree level below the root or at a component level.

    ```typescript
    @Injectable({
      providedIn: 'root'
    })
    ```

2. **Dependency Injection**:

Annotate the constructor in the component or service to be injected.

```typescript
constructor(private myService: MyService) {}
```

3. **Lifecycle Management**: Handle service lifecycle based on the specific requirements, such as persistent state management.

### Code Example: Service

Here is the TypeScript code:

```typescript
// service.ts
@Injectable({
  providedIn: 'root'
})
export class MyService {
  private data: any;

  setData(data: any): void {
    this.data = data;
  }

  getData(): any {
    return this.data;
  }
}

// component.ts
export class MyComponent {
  constructor(private myService: MyService) {}

  saveDataLocally(data: any): void {
    this.myService.setData(data);
  }

  fetchStoredData(): any {
    return this.myService.getData();
  }
}
```

In this case, the `MyService` will persist its `data` property throughout its lifetime, and any component or service can access or modify it using the defined methods.
<br>

## 7. Can you explain what _dependency injection_ is in Angular?

**Dependency Injection (DI)** is a core concept in Angular, where components (or services) depend on other components. Angular handles the creation and management of these dependencies.

### Simplified Explanation

DI takes three steps:

1. **Registration**: Identify the components to be injected.
2. **Resolution**: Find the appropriate dependencies.
3. **Injection**: Insert the resolved dependencies.

### Key Angular Features Linked to DI

- **Modules**: Angular applications are made up of modules, each with its dependency injector.
- **Providers**: Within modules, providers offer a mechanism for registering dependencies.

### Code Example: DI in Angular

Here is the Angular code:

```typescript
// Service definition
@Injectable()
export class DataService {
  getData() {
    return "Some data";
  }
}

// Register in a module
@NgModule({
  providers: [DataService],
  // ...
})
export class MyModule {}

// Constructor injection in a component
@Component({
  // ...
})
export class MyComponent {
  constructor(private dataService: DataService) {}

  ngOnInit() {
    console.log(this.dataService.getData());
  }
}
```
<br>

## 8. What is a _module_ in Angular and what is its purpose?

In Angular, a **module** is a way to group components, services, directives, and pipes. It helps in both organizing and dividing your application into smaller, more manageable and efficient **pieces**.

### Key Module Elements

- **Components**: The visual and behavioral building blocks of your application.
- **Directives**: Tools for modifying the DOM or containing certain behaviors.
- **Services**: Reusable units of code, often central to your application's functionality.
- **Pipes**: Data transformation agents, primarily used for UI purposes.

### Types of Modules

- **Root Module**: The core module that serves as the entry point for your application. It's often called `AppModule`.
- **Feature Module**: An optional module that's usually smaller in scope and can be **lazily loaded**. It usually targets a specific feature or a set of related features, allowing for better code organization and loading only when needed.

### Advantages of Using Modules

1. **Organization and Reusability**: Components, services, directives, and more are logically grouped, making their intent clear and their code easily accessible. They can also be shared across modules as needed.
2. **Performance and Efficiency**: Modules can be **eager-loaded** (automatically loaded with the application) or **lazily-loaded** (loaded on-demand), optimizing initial bundle size and reducing start-up time.
3. **Collaborative Development**: By defining clear boundaries between components, directives, and services, modules facilitate team collaboration and help in preventing naming conflicts or unintentional dependencies.

### Code Example: Module Structure

Here is the Angular code:

```typescript
// File: app.module.ts
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { AppComponent } from './app.component';

@NgModule({
  declarations: [AppComponent], // Components declared in this module
  imports: [BrowserModule], // Other modules this module requires
  providers: [], // Services provided by this module
  bootstrap: [AppComponent] // The root component of this module
})
export class AppModule {}  // The root module

// File: user.module.ts (Feature Module Example)
import { NgModule } from '@angular/core';
import { UserComponent } from './user.component';

@NgModule({
  declarations: [UserComponent], // Components declared in this module
  imports: [], // Other modules this module requires
  providers: [], // Services provided by this module
})
export class UserModule {}
```
<br>

## 9. How do you handle _events_ in Angular?

**Handling events** in Angular involves capturing and responding to user or system actions. Angular provides **declarative** and **imperative** methods to accomplish this.

### Declarative Approach

Declarative methods involve writing event handlers directly in the Angular template using **event binding**.

#### Syntax

For event binding, Angular uses `(event)` syntax to listen for DOM events and execute an associated method in the component.

**(e.g.) Click Event:**

- Template:
  ```html
  <button (click)="onClick($event)">Click Me</button>
  ```

- Component:
  ```typescript
  onClick(event: MouseEvent): void {
      console.log('Button was clicked', event);
  }
  ```

**(e.g.) Input Event:**

- Template:
  ```html
  <input (input)="onInput($event)">
  ```

- Component:
  ```typescript
  onInput(event: Event): void {
      const inputText = (event.target as HTMLInputElement).value;
      console.log('Input value changed:', inputText);
  }
  ```

**Event Objects** are optionally passed to event handling functions. These objects contain specific properties based on the event type, such as `MouseEvent` for click events and `KeyboardEvent` for keyboard-related ones.

### Imperative Approach

While Angular promotes a **declarative style**, it also supports an **imperative** one, where event listeners are manually added and removed through the `@ViewChild` decorator of TypeScript.

#### Syntax

- **HTML Template:**
  ```html
  <div #targetDiv>Target DIV</div>
  ```

- **Component Class:**
  ```typescript
  @ViewChild('targetDiv') targetDiv: ElementRef;

  ngAfterViewInit(): void {
    this.targetDiv.nativeElement.addEventListener('click', this.onClick);
  }

  ngOnDestroy(): void {
    this.targetDiv.nativeElement.removeEventListener('click', this.onClick);
  }

  onClick(event: MouseEvent): void {
    console.log('Div clicked:', event);
  }
  ```

In this method, the `ngAfterViewInit` method sets up the event listener, and the `ngOnDestroy` method removes it to prevent memory leaks or unexpected behavior.
<br>

## 10. What is _two-way binding_ and how do you implement it in Angular?

**Two-way binding** in Angular synchronizes data between the data model and the view in **both directions**. Any changes in the data model automatically reflect in the view and vice versa.

### Implementation in Angular

Angular primarily uses two-way binding through the `[(ngModel)]` directive, leveraging the `FormsModule` or `ReactiveFormsModule`.

### Using `[(ngModel)]` and `FormsModule`

1. **Module Setup**: Import the `FormsModule` in your Angular module. 
```typescript
    import { FormsModule } from '@angular/forms';

    @NgModule({
        imports: [FormsModule],
        // ...
    })
    export class AppModule { }
```

2. **Input Binding**: Use `[(ngModel)]` in the view to enable two-way binding with an input element.

```html
    <input [(ngModel)]="name" name="name" />
```

3. **Data Model**: Define the associated property in the component.
```typescript
    @Component({...})
    export class TwoWayBindingComponent {
        public name: string;
    }
```

This setup ensures that any changes to the input element are reflected in the `name` property and vice versa.

### Using `[(ngModel)]` with `ReactiveFormsModule`

If you choose to integrate `[(ngModel)]` with `ReactiveFormsModule`, follow these steps:

1. **Module Setup**: Import the `ReactiveFormsModule` in your Angular module.
    - Code Example: `app.module.ts`
    ```typescript
    import { ReactiveFormsModule } from '@angular/forms';

    @NgModule({
        imports: [ReactiveFormsModule],
        // ...
    })
    export class AppModule { }
    ```

2. **FormGroup Creation**: Create an Angular `FormGroup` and associate it with the template and component.

3. **Model Binding**: Use the `formControlName` directive in the view to bind an input to a specific form control.

```html
    <form [formGroup]="myForm">
        <input formControlName="name" />
    </form>
```

```typescript
    import { FormBuilder, FormGroup } from '@angular/forms';

    @Component({...})
    export class TwoWayBindingReactiveComponent {
        public myForm: FormGroup;

        constructor(private fb: FormBuilder) {
            this.myForm = this.fb.group({
                name: ['']
            });
        }
    }
```

This approach ensures synchronized data between the form input and the `FormControl` associated with it.

### Best Practices

- **Consistent Tracking**: Whether through the `FormsModule` or `ReactiveFormsModule`, ensure consistent data tracking to avoid unexpected behavior.
- **Input Element Type**: Not all elements support two-way binding. Use two-way bindings like `[(ngModel)]` with compatible input elements such as `<input>` and `<textarea>`.

### When to Use Two-Way Binding

While two-way binding can simplify form handling and updates in smaller applications, its use in larger, complex applications might introduce maintenance challenges and make it harder to understand data flow. In such scenarios, one-way data flow using reactive patterns or unidirectional data flow might be more suitable.
<br>

## 11. Explain the difference between an Angular _component_ and a _directive_.

While both **components** and **directives** are fundamental to Angular, their roles and functionalities differ.

### Key Distinctions

#### Purpose
   Components are the building blocks of the UI, consisting of HTML templates and design logic. Directives alter the behavior or appearance of elements - `Structural Directives` can also manipulate the DOM.

#### Nature
   Components are more comprehensive and self-contained, representing entire parts of the UI. In contrast, directives can be attribute-based or reusable, handling specific behaviors or responsibilities.

#### Templating
   Components always have their template, providing a view for users. On the other hand, directives can have their template, operate within an existing one, or not have a template at all.

#### Code Reusability
   While both components and directives ensure code modularization, directives, especially attribute-based ones, are more about sharing specific functionalities across different components.

#### Interaction with Angular Material
   Angular Material has been used for the demo. Let me know if you want to skip it.

   While **BetterChoiceComponent** is a reusable component that can be used anywhere in your app, **app-better-choice-directive** functions more like an HTML attribute with its style functionalities.

#### Technical Overview
    Components are **@Component**-decorated classes that encapsulate a template, CSS styles, and application-specific logic. Directives, on the other hand, come in the flavor of **@Directive**, **@Component**, **@ViewChild**, or **@ViewChildren**, allowing you to create **attribute directives** leveraging **@HostListener** and **@HostBinding**, or **structural directives** like **ngIf** and **ngFor** for DOM manipulation.

#### Example Use-Cases

   - Directive based: SuperviseElementDirective - to monitor if a specific element is in view.
   - Component-based: WeatherWidgetComponent to display weather data in a more intricate setup.
<br>

## 12. What are _Pipes_ in Angular and where would you use them?

**Pipes** in Angular allow you to **transform** displayed values in templates. Use them to format strings, dates, decimals; or to sort and filter arrays.

### Core Pipe Types

#### String

Use the `string` parameter to perform text transformations. In the template, use the pipe as follows:

```angular2html
<p>{{ name | uppercase }}</p>
```

#### Numeric

The built-in number pipes allow decimal and currency formatting, and the percent pipe displays a number as a percentage.

```angular2html
<p>{{ pi | number: '3.1-5' }}</p>
<p>{{ price | currency: 'EUR' }}</p>
<p>{{ rate | percent }}</p>
```

#### Date

The `date` pipe can format dates as per your requirements:

```angular2html
<p>{{ today | date: 'dd/MM/yyyy' }}</p>
```

#### Array

The `array` pipe allows you to sort or filter an array in the view. For example, to sort a list of names:

```angular2html
<label for="sortOrder">Ascending</label>
<input type="checkbox" id="sortOrder" [(ngModel)]="ascending" />
<ul>
  <li *ngFor="let name of namesList | sortName: ascOrder">{{ name }}</li>
</ul>
```

#### Custom Pipes

You can create **custom pipes** when the built-in ones don't meet your requirements. For example:

```typescript
@Pipe({ name: 'sortName' })
export class SortNamePipe implements PipeTransform {
  transform(value: string[], ascOrder: boolean = true): string[] {
    if (ascOrder) {
      return value.sort();
    }
    return value.sort().reverse();
  }
}
```
<br>

## 13. How do you handle _form submissions_ in Angular?

In Angular, **creating, validating, and submitting forms** can happen either using **Template-Driven Forms** or **Reactive Forms**.

### Template-Driven Forms

**Template-Driven Forms** are easier to set up but offer limited functionality compared to Reactive Forms.

#### Steps to Use Template-Driven Forms

1. **Import the FormsModule**: In the app module, you need to import `FormsModule`.

2. **Add a `form` tag**: Within your component's markup, place a `<form>` tag that binds to `NgForm`.

3. **Bind to Form Controls**: Use directives like `ngModel` to handle data bindings and validation.

4. **Customize Validation**: Make use of built-in directives like `required`, or customize them like `NgModel with ngControl`.

5. **Work with Submission**: Define a method for the `ngSubmit` event of the form.

#### TypeScript Example: Implementing Template-Driven Forms

Here is the Angular typescript code:

```typescript
import { Component } from '@angular/core';

@Component({
  selector: 'template-driven-form',
  template: `
    <form (ngSubmit)="onSubmit()" #userForm="ngForm">
      <input type="text" class="form-control" name="name" required ngModel>
      <button type="submit" [disabled]="userForm.invalid">Submit</button>
    </form>
  `
})
export class TemplateDrivenFormComponent {
  onSubmit() {
    // form submit logic
  }
}
```

### Reactive Forms

**Reactive Forms** offer more flexibility, allow you to define form controls in the component class, can be easier to test and provide a clearer code structure.

#### Steps to Use Reactive Forms

1. **Import the ReactiveFormsModule**: In the app module, you need to import `ReactiveFormsModule`.

2. **Create the Form Controls Programmatically**: In the component class, use the `FormControl`, `FormGroup`, and `FormBuilder` classes to create form controls.
   
3. **Bind to Form Controls**: Use directives such as `formControlName`, or `formGroup` to link form controls to HTML elements.

4. **Customize Validation**: Use Validators from `@angular/forms`, and create custom validators as needed.

5. **Work with Submission**: Subscribe to the form's `value` or `status` changes, and perform actions accordingly, instead of using a method for `ngSubmit` as in Template-Driven forms.

#### TypeScript Example: Implementing Reactive Forms

Here is the Angular typescript code:

```typescript
import { Component } from '@angular/core';
import { FormBuilder, Validators } from '@angular/forms';

@Component({
  selector: 'reactive-form',
  template: `
    <form [formGroup]="userForm" (ngSubmit)="onSubmit()">
      <input type="text" formControlName="name" class="form-control">
      <button type="submit" [disabled]="userForm.invalid">Submit</button>
    </form>
  `
})
export class ReactiveFormComponent {
  userForm = this.formBuilder.group({
    name: ['', Validators.required]
  });

  constructor(private formBuilder: FormBuilder) {}

  onSubmit() {
    // form submit logic
  }
}
```

### Code for App Module

Here is the Angular typescript code for App Module:

```typescript
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { ReactiveFormsModule } from '@angular/forms';

import { TemplateDrivenFormComponent } from './template-driven-form.component';
import { ReactiveFormComponent } from './reactive-form.component';

@NgModule({
  imports: [BrowserModule, ReactiveFormsModule],
  declarations: [TemplateDrivenFormComponent, ReactiveFormComponent],
  bootstrap: [TemplateDrivenFormComponent, ReactiveFormComponent]
})
export class AppModule {}
```
<br>

## 14. What is _Angular CLI_ and what can it be used for?

**Angular CLI** (Command Line Interface) is a powerful tool that accelerates **Angular** development. It automates various tasks from initializing projects to deployment. 

### Key Features and Uses

- **Project Initialization**: Angular CLI streamlines the creation of new projects, sparing developers from having to set up configurations manually.

- **Scaffold**: It generates files and folders for **components**, **services** and other Angular elements with structured code.

- **Integrated Testing**: Developers can run both unit tests and end-to-end tests seamlessly using built-in tools like Karma and Protractor.

- **Web Server**: For local development, Angular CLI has a built-in web server.

- **Live Code Changes**: Utilizing LiveReload, the development server immediately reflects code changes in the browser.

- **Code Optimization and Bundling**: Angular CLI ensures production-ready applications through mechanisms such as **minification** and **tree shaking**.

- **Deployment**: The CLI's optimized builds are deployable across platforms.

- **Custom Schematics**: Developers can create custom project blueprints to standardize processes within an organization.

- **Global Consistency across Teams**: Using CLI commands ensures a uniform code structure and development workflow across teams.

### The Angular CLI Workflow: Start to Finish

1. **Install Angular CLI**: Use npm to install CLI globally: 
    ```bash
    npm install -g @angular/cli
    ```

2. **Create New Project**: Initiate a new Angular project:
    ```bash
    ng new my-angular-app
    ```

3. **Serve the Application**: Test the app locally with the integrated web server:
    ```bash
    ng serve
    ```

4. **Develop and Iterate**: Use scaffold tools and test the application as you build it.

5. **Build for Production**: Create an optimized build for deployment:
    ```bash
    ng build --prod
    ```

6. **Deploy**: Deploy the app using host-specific instructions.

7. **Stay Updated**: Keep Angular CLI up-to-date:
    ```bash
    npm install -g @angular/cli
    ```
<br>

## 15. Describe how to make _HTTP requests_ in Angular using _HttpClient_.

In Angular, you can make **HTTP requests** using Angular's built-in `HttpClient` service. This is a more modern approach than the now deprecated `HttpModule`.

### Setting Up `HttpClientModule`

In your Angular module, import `HttpClientModule` and add it to the `imports` array.

```typescript
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { AppComponent } from './app.component';

@NgModule({
  declarations: [AppComponent],  
  imports: [BrowserModule, HttpClientModule],  
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
```

### Making Simple HTTP GET Request

Here's an example of making a simple GET request to a REST API:

```typescript
import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable()
export class DataService {
  private apiURL = 'https://api.example.com/data';

  constructor(private http: HttpClient) {}

  fetchData(): Observable<any> {
    return this.http.get<any>(this.apiURL);
  }
}
```

### Handling Response

You can also handle the response using various `HttpClient` methods. Here are a few examples:

#### `get` Method with JSON Response

```typescript
getData(): Observable<DataModel> {
  return this.http.get<DataModel>(this.apiURL);
}
```

#### `get` Method for Non-JSON Data

If you expect response data in a format other than JSON, `HttpClient` allows you to specify the response type:

```typescript
getTextData(): Observable<string> {
  return this.http.get(this.apiURL, { responseType: 'text' });
}
```

#### Error Handling

You can also handle errors using RxJS `catchError` operator. The `HttpClient` methods such as `get`, `post`, etc., return an `Observable` that can be further manipulated using RxJS operators.

Here is an example:

```typescript
import { catchError, map } from 'rxjs/operators';
import { of } from 'rxjs';

getDataWithCatchError(): Observable<DataModel> {
  return this.http.get<DataModel>(this.apiURL).pipe(
    catchError((error) => {
      console.error('Error:', error);
      return of(null); // Return a default value or re-throw the error
    })
  );
}
```

### Making Other HTTP Requests

- **POST Request**: Use `post` method. It allows you to send a request body.

  ```typescript
  postData(data: any): Observable<any> {
    return this.http.post<any>(this.apiURL, data);
  }
  ```

- **PUT Request**: Use `put` method for updating resources.

  ```typescript
  updateData(data: any): Observable<any> {
    return this.http.put<any>(this.apiURL, data);
  }
  ```

- **DELETE Request**: Use `delete` method for deleting resources.

  ```typescript
  deleteData(id: string): Observable<any> {
    return this.http.delete<any>(`${this.apiURL}/${id}`);
  }
  ```

- **Custom Headers**: You can pass an `HttpHeaders` object for custom headers.

  ```typescript
  import { HttpHeaders } from '@angular/common/http';
  
  // Set up headers
  const headers = new HttpHeaders().set('Authorization', 'Bearer my-jwt-token');
  
  // Pass headers in the request
  return this.http.get<any>(this.apiURL, { headers });
  ```

### Security Considerations

For production-grade applications, it's important to secure your HTTP requests over insecure networks (like the internet) using **SSL/TLS**.
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - Angular](https://devinterview.io/questions/web-and-mobile-development/angular-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

