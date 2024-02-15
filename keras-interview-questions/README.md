# 70 Core Keras Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/machine-learning-and-data-science/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fmachine-learning-and-data-science-github-img.jpg?alt=media&token=c511359d-cb91-4157-9465-a8e75a0242fe" alt="machine-learning-and-data-science" width="100%">
</a>
</p>

#### You can also find all 70 answers here 👉 [Devinterview.io - Keras](https://devinterview.io/questions/machine-learning-and-data-science/keras-interview-questions)

<br>

## 1. What is _Keras_ and how does it relate to _TensorFlow_?

**Keras** serves as TensorFlow's high-level API, pioneering a user-friendly, modular deep learning framework. Here are the key features:

### Key Features and Advantages

- **High-Level Abstraction**: Keras simplifies model construction, making it accessible even to novices in deep learning.
- **Fast Prototyping**: The API allows rapid building and testing of neural network architectures.
- **Compatibility**: It's compatible with TensorFlow, Theano, and Microsoft Cognitive Toolkit (CNTK).
- **Support for Multiple Backends**: Enables users to move between different computation engines seamlessly.
- **Modularity and Flexibility**: Layers, models, and optimizers in Keras are modular and customizable.
- **Integrated Utilities**: Offers a built-in toolset for tasks like data preprocessing, model evaluation, and visualization.
- **Wide Range of Applications**: Keras is customizable and adaptable, catering to various machine learning tasks.
- **Huge Community**: Due to its user-friendly yet powerful nature, Keras has a large and active community of users.

  Keras serves as an abstraction layer, offering a high-level, user-friendly interface for building and training neural network models in TensorFlow.

### Keras vs Pure TensorFlow

When should you use **Keras** over pure **TensorFlow**, and vice versa?

#### Keras

- **Advantages**:

  - User-Friendly: Keras is easier to learn and use for beginners.
  - Faster Prototyping: Its simplicity and mode of use aid in rapid model creation and testing.
  - Clear and Concise Code: Keras's high-level abstraction means code is often more easy to read and follow.

- **When to Use It**:

  - For quick, small to medium-sized projects
  - When you prize simplicity and "rapid proof of concept"
  - If you're starting out in deep learning

#### TensorFlow with `tf.keras`

- **Advantages**:

  - Seamless Integration: As TensorFlow's high-level API, it integrates smoothly with lower-level TensorFlow operations and workflows.
  - Greater Control: Offers more flexibility and control over the model and training process.
  - Real-World Applications: Suits larger, more complex projects and customized models.

- **When to Use It**:

  - For projects requiring advanced or highly specialized models
  - If you prioritize control and scalability
  - In production-grade applications

### Code Example: CIFAR-10 Classification with Keras

Here is the Python code:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and prepare the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Model definition
model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  Flatten(),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
```
<br>

## 2. Can you explain the concept of a _deep learning framework_?

A **Deep Learning Framework** acts as a vital foundation for designing and training neural networks. It facilitates numerous tasks such as data preprocessing, model specification, optimization settings, and model evaluation.

### Purpose of a Deep Learning Framework

- **Abstraction**: Hides intricate math behind user-friendly interfaces.
- **Performance Optimizations**: Often uses GPU acceleration to expedite computations.
- **Versatility**: Allows for both high-level and low-level model architecture design, catering to novice and experienced users.
- **Scalability**: Streamlines deployment, enabling the transition from small-scale experiments to large-scale industrial applications.

### Core Components

1. **Model Library**: Offers a variety of network architectures, such as CNNs, RNNs, and GANs, along with pre-trained models for transfer learning.
2. **Layers & Modules**: Building blocks for constructing models, ranging from simple ones like dense layers to more intricate ones like LSTM cells.
3. **Loss Functions**: Metrics quantifying model performance during training.
4. **Optimizers**: Algorithms guiding the model's learning process.
5. **Data Utilities**: Accommodates data pre-processing, augmentation, and pipelines for seamless feeding of data to models.

### Common Deep Learning Frameworks

- **TensorFlow (Google)**: Its distinctive feature is its data flow graph mechanism.
  
- **Keras (Community-Maintained Lens on TensorFlow)**: An API designed for user-friendliness, acting as the go-to choice for beginners and rapid prototypers.

- **PyTorch (Facebook)**: Based on a dynamic computation graph paradigm, it has gained traction for its intuitive nature and has become the tool of choice for numerous machine learning researchers.

- **Caffe (UC Berkeley)**: This framework shines in its capacity to efficiently handle computer vision tasks and deep learning networks.

- **Theano (University of Montreal)**: One of the earlier frameworks, it offered functionalities for defining, optimizing, and evaluating mathematical expressions.

- **MXNet (Apache)**: Known for its speed and efficiency, MXNet supports various front-ends (including Keras) and has embedded support for a range of high-level languages.

- **Chainer (Preferred by RIKEN and Preferred Networks Co., Ltd.)**: Standing out with its ease-of-use and straightforwardness, it is appreciated for its flexibility in network design and its dynamic graph mechanism.
<br>

## 3. What are the _core components_ of a _Keras model_?

A Keras model is built around its core components, the **Input and Output Layers**, and is structured through its **Architecture and Training Configurations**.

### Key Components of a Keras Model

1. **Input Layer**: Passes data directly to the first hidden layer. It includes shape information for the data.

2. **Output Layer**: The final layer that produces predictions. The type of Model (Sequential or Functional) determines its shape.

3. **Architecture**: The arrangement of layers. Keras offers two main types: the straightforward **Sequential** model and the more flexible **Functional** model.

4. **Training Configurations**: Define how the model learns from data using optimizers, loss functions, and evaluation metrics.

#### Code Example: Keras Model Building (Sequential and Functional)

Here is the Python code:

```python
# Sequential model
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=1, activation='sigmoid'))

# Functional model
inputs = keras.Input(shape=(100,))
x = Dense(units=64, activation='relu')(inputs)
outputs = Dense(units=1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

### Flexibility of Keras Functional Model

The **Functional API** can effectively handle complex architectures like multi-input and multi-output models, shared layers, and more intricate network topologies, making it an ideal choice for **sophisticated model designs**. This aspect provides more versatility compared to the straightforward **Sequential** model.

For instance, in the case of an image model which processes images from two taps, we can split the input into two branches and then combine those branches' output. Here is the code, in Python, for it:

```python
from keras.layers import Input, Embedding, LSTM, concatenate, Dense
from keras.models import Model

input_1 = Input(shape=(28, 28))
input_2 = Input(shape=(28, 28))

shared_embedding = Embedding(1000, 64)
x1 = shared_embedding(input_1)
x2 = shared_embedding(input_2)

lstm_out = LSTM(32)

out_1 = lstm_out(x1)
out_2 = lstm_out(x2)

concatenated = concatenate([out_1, out_2])

output = Dense(1, activation='sigmoid')(concatenated)

model = Model(inputs=[input_1, input_2], outputs=output)
```
<br>

## 4. How do you _configure a neural network_ in _Keras_?

In Keras, you can **configure** a neural network through two approaches:

### 01. Sequential Model Configuration

This method is suitable for **simple network architectures** characterized by a linear, one-input-one-output stacking of layers.

#### Code Example: Sequential Model

Here is the Python code:

```python
import keras
from keras import layers

# Initialize a sequential model
model = keras.Sequential()

# Add layers one by one
model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))
```

### 02. Functional API Model Configuration

This method is the recommended choice for more **complex architecture designs** such as multi-input and multi-output models, layer sharing, and non-linear topologies.

#### Code Example: Functional API Model

Here is the Python code:

```python
import keras
from keras import layers

# Input tensor placeholder
inputs = keras.Input(shape=(784,))

# Hidden layers
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)

# Output layer for 10-class classification
outputs = layers.Dense(10, activation='softmax')(x)

# Define the model
model = keras.Model(inputs=inputs, outputs=outputs)
```
<br>

## 5. Explain the difference between _sequential_ and _functional APIs_ in _Keras_.

While **Sequential** and **Functional** APIs in Keras both facilitate the creation of neural networks, they differ in **flexibility, complexity** of models that can be built, and the way each API is used.

### Key Distinctions

- **Control Flow**:  
   - **Sequential**: Fixed.
   - **Functional**: Arbitrary, using Keras layers.
 
- **Model Complexity**:  
   - **Sequential**: Suited for straight, linear stacks of layers where each layer has one input and one output.
   - **Functional**: For complex, multi-input, multi-output architectures, including shared layers and branches.

- **Use Case**:  
   - **Sequential**: Ideal for beginners and linear networks.
   - **Functional**: Better for advanced architectures and research.

- **Evaluation and Inference**: 
   - **Sequential**: Simple. 
      - For evaluation, use `model.evaluate()`.
      - For prediction, use `model.predict()`.
   
   - **Functional**: Potentially complex, with custom feedback mechanisms.  
      - For evaluation, use `model.evaluate()` and possibly custom measures.
      - For prediction, typically `model.predict()`, but can involve multi-input or multi-output structures.

### Benefits of Using the Functional API

- **Multi-Modal Inputs and Multi-Output Structures**:  
   - Models can process different types of data through the use of multiple input layers.
   - Useful for tasks like joint sentiment analysis and emotion recognition from text and images in social media data.

- **Model Branching**:  
   - Encourages the creation of 'branching' or 'dual pathway' models where different inputs may follow separate paths before being combined.

- **Model Sharing with Multiple Inputs and Outputs**:  
   - Supports models where multiple output layers could be predicted simultaneously using the same internal state from the shared layers.

- **Non-Sequential Connectivity**:  
   - Non-linear connections can exist between layers.

- **Custom Loss Functions and Model Metrics**:  
   - Allows for the computation of more complicated loss functions to handle the multiple outputs or inputs.

- **Layer Sharing and Resue**:  
   - Layers can be reused across different parts of the model, making it easier to work with complex architectures.

### Code Example: Multi-Input/Output Model using Functional API

Here is the Python code:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define input for numerical data
numerical_input = keras.Input(shape=(2,), name="numerical_input")

# Define input for categorical data
categorical_input = keras.Input(shape=(3,), name="categorical_input")

# Multi-modal input concatenation
concatenation = layers.Concatenate()([numerical_input, categorical_input])

# Hidden layer
hidden1 = layers.Dense(3, activation='relu')(concatenation)

# Define two output branches from the hidden layer
output1 = layers.Dense(1, name="output1")(hidden1)
output2 = layers.Dense(1, name="output2")(hidden1)

# Create the model
model = keras.Model(inputs=[numerical_input, categorical_input], outputs=[output1, output2])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), 
              loss={"output1": "mse", "output2": "mse"},
              metrics={"output1": "mae", "output2": "mae"})
```
<br>

## 6. Describe how you would _install and set up Keras_ in a _Python_ environment.

**Keras** can be installed using `pip` or `conda`. Before installation, make sure you have Python $>= 3.5$, as Keras does not support Python 2.

### Installing Keras

Using `pip`:

```bash
pip install tensorflow   # Install TensorFlow backend as Keras may not work without TensorFlow.
pip install keras
```

Using `conda` (recommended with Anaconda distribution):

```bash
conda install -c conda-forge keras
conda install -c conda-forge tensorflow
```

If using TensorFlow as the Keras backend, it is recommended to install Keras with TensorFlow as it is incorporated better.

### Setting Keras Backend

After the installation, you may need to configure the Keras backend. If using TensorFlow and installed it separately, you typically don't need to set it up; otherwise, to link Keras to TensorFlow, run the following Python code:

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
```

### Verifying the Installation

To check if Keras is installed and set up correctly, import Keras and verify the configuration using the following Python code:

```python
import keras
from keras import backend as K

print(keras.__version__)  # Display Keras version.
print(K.backend())        # Display the current backend (e.g., 'tensorflow').
```
<br>

## 7. What are some _advantages_ of using _Keras_ over other _deep learning frameworks_?

**Keras** offers several distinct advantages as a library for building neural networks over other frameworks like TensorFlow or PyTorch.

### Key Advantages of Keras

1. **Intuitive Interface**: Keras is designed with simplicity in mind – making it ideal for both beginners and experts. Its API is clear and concise, providing a smooth learning curve.

2. **Modularity and Flexibility**: Keras models are created by stacking layers making them highly intuitive. It allows easy model reusability, modular architecture, models' customization, and experiment management.

3. **Back-end Agnosticism**: Although originally built on top of TensorFlow, Keras is now integrated with TensorFlow, Theano, and CNTK. It's especially useful when one framework has an edge for specific tasks or hardware.

4. **Multiple Compute Platforms**: Keras supports CPU and GPU computation, and its compatibility with Google's TPU (Tensor Processing Unit) through TensorFlow makes it highly versatile.

5. **Clean and Lightweight**: With relatively fewer lines of code, Keras is less verbose than other deep learning frameworks, enhancing readability and speed of model prototyping.

6. **Increased Performance in Data Parallel Jobs**:  Data-parallel jobs (like training on multisocket CPUs and multiple GPUs) see performance gains as tensors get split at computation boundaries.

7. **Package Delivery and Resource-efficiency with Tensors**: It facilitates the sharing, command sequences, and eliminates redundant memory copying.

### Contextual Comparison

- **TensorFlow alone offers in-depth control.** However, the trade-off is added complexity and a steeper learning curve.
  
- **PyTorch, especially favored by researchers**, boasts dynamic computation graphs and imperative programming, but the trade-off is increased code complexity.

- **Keras** balances a focus on user-friendliness, efficient experimentation, and deployment-ready models. 

Incorporating higher-level libraries, such as `tf.keras` and `keras-tuner`, adds streamlined model optimization, auto-ML capability, and improved compatibility with TensorFlow features like TF Records, Estimators, and TPUs.
<br>

## 8. How do you _save and load models_ in _Keras_?

Keras provides straightforward methods to **save and load trained models**.

### Save and Load Model with Keras

You can save or load two types of Keras models:

1. **Architecture-Only Models**: Saved as JSON or YAML files. These files contain the model's architecture but not its weights or training configuration.
2. **Stateful Models**: Saved as Hierarchical Data Format (HDF5) files. They store the model's architecture, weights, and training configuration. This is the preferred way for saving trained models.

Here is the Keras code that demonstrates how to do that:

#### Code: Save and Load Models in Keras

Here is the Python code:

```python
from keras.models import model_from_json

# Save model in JSON format (architecture-only)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Load model from JSON
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Save model in HDF5 format (stateful model)
model.save("model.h5")

# Load model from HDF5
loaded_model_h5 = load_model('model.h5')
```

### Additional Considerations

- **Historical Data**: The HDF5 format does not save the training metrics. This means that the loaded model will not include the training and validation histories. If you need those, you'll have to save them separately. 

- **Custom Objects**: If your model uses custom layers or other custom objects, you need to pass them to the loading mechanism using the `custom_objects` parameter. This is particularly crucial when loading pre-trained models.
<br>

## 9. What is the purpose of the _Dense layer_ in _Keras_?

The **Dense layer** in Keras is a standard, fully-connected neural network layer. It is essential in most network architectures, serving as the primary mechanism for training model parameters. This layer ensures a structured, efficient data flow in both forward and backward propagation.

### Key Components

- **Dense Layer**: Each neuron in this layer is connected to every neuron in the preceding and succeeding layers. Mathematically, it multiplies the input data by a weight matrix and adds a bias vector.
- **Activation Function**: This non-linear function introduces complexity and adaptability to the network.

### Mathematical Representation

The output $y$ from a single neuron in a dense layer with $n$ input features is calculated as:

$$
y = \sigma \left( \sum_{i=1}^{n} (w_i \cdot x_i) + b \right)
$$

where:

- $\sigma$ is the activation function.
- $w_i$ are the weights associated with each input feature $x_i$.
- $b$ is the bias term.

### Code Example: Single Neuron in a Dense Layer

Here is the Python code:

```python
# Import relevant libraries
from keras.models import Sequential
from keras.layers import Dense

# Initialize neural network model
model = Sequential()

# Add a dense layer with one neuron and sigmoid activation
model.add(Dense(units=1, input_dim=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
```
<br>

## 10. How would you implement a _Convolutional Neural Network_ in _Keras_?

In Keras, you can create a **Convolutional Neural Network** (CNN) using the `Sequential` model. This model requires you to add layers in a sequential pattern. 

To build a CNN, you would use `Conv2D`, `MaxPooling2D`, and other layers as well as **flatten** and **dense** layers.

### Key Keras Modules

- **Layers**: Define the layers, such as `Dense` for fully connected layers and `Conv2D` for convolutional layers.
- **Model**: Combines layers to build the neural network.
- **Optimizers**: Configures the model for training, e.g., `Adam`.
- **Loss Functions**: Indicate the goal of the learning task, e.g., `categorical_crossentropy` for classification.
- **Metrics**: Measure performance, e.g., `accuracy` for classification tasks.

### Step-by-Step with Code Example

Here is the Python code:

  ```python
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    # Create a Sequential model
    model = Sequential()

    # Add Convolutional layers
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output from preceding layers
    model.add(Flatten())

    # Add a Dense layer for final classification
    model.add(Dense(units=128, activation='relu'))  # Optional intermediate layer
    model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model before fitting it to data
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```
<br>

## 11. Can you describe how _Recurrent Neural Networks_ are different and how to implement one in _Keras_?

A **Recurrent Neural Network** (RNN) processes **sequential data** by maintaining a hidden state that encapsulates the network's knowledge up to a certain point in the sequence.

For instance, if our input sequence is $(x_1, x_2, x_3, \ldots, x_T)$, the RNN updates its hidden state at each time step $t$ using the input at that time step $x_t$ and the previous hidden state $h_{t-1}$. Mathematically, this is represented as:

$$
h_t = \text{RNN}(x_t, h_{t-1})
$$

### Key Distinctive Features of RNNs

#### Concept of Hidden State

RNNs leverage a notion called the **hidden state**, $h_t$, to maintain a contextual understanding of all the preceding elements in the sequence. The hidden state is recursively updated at each time step using both the current input element and the previous hidden state.

#### Shared Parameters

In an RNN, the transformation that maps the input and the previous hidden state to the next hidden state uses the **same set of parameters** for every time step. This feature enables the RNN to generalize across the sequence.

Mathematically:

$$
h_t = \text{tanh}(W_{hh}h_{t-1} + W_{xh}x_t + b_h)
$$

Here, $W_{hh}$ and $W_{xh}$ are the shared weight matrices, and $b_h$ is the shared bias vector.

### How to Implement an RNN in Keras

Keras provides a high-level API for building RNNs. Specifically, you can use the **SimpleRNN** layer to construct basic RNN models.

Here is the Python code:

```python
from keras.models import Sequential
from keras.layers import SimpleRNN, Embedding
from keras.layers import Dense

model = Sequential()

# SimpleRNN with 32 hidden neurons
model.add(SimpleRNN(32, input_shape=(None, 100))) # Input shape: (num_timesteps, input_dim)

# A final Dense layer for prediction
model.add(Dense(1, activation='sigmoid'))

model.summary()
```

In the example above:

- We define a `Sequential` model.
- We add a `SimpleRNN` layer with 32 hidden neurons. The layer infers the input shape, `(num_timesteps, input_dim)`, from the shape of its input.
- We add a final `Dense` layer with a sigmoid activation for binary classification tasks.
<br>

## 12. Explain the purpose of _dropout layers_ and how to use them in _Keras_.

**Dropout layers** are a powerful tool designed to **prevent overfitting** in deep learning models.

### Purpose of Dropout Layers

- **Regularization**: Dropout discourages overfitting, enhancing a model's ability to generalize.
- **Ensemble Learning**: It simulates training and testing on multiple different thinned-out (or "dropped-out") networks, which can boost robustness.

### Dropout Mechanism

During training, each neuron in the dropout layer has a probability $p$ of being "dropped out" or temporarily removed. The remaining neurons are then "scaled up" by a factor of $1/(1-p)$ to maintain model expectation.

At inference, all neurons are active, but their outputs are divided by $1-p$ to ensure consistency with the training phase.

### Code Example: Using Dropout in Keras

Here is the Python code:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, input_shape=(10,), activation='relu'),
    Dropout(0.2),  # 20% dropout rate
    Dense(64, activation='relu'),
    Dropout(0.3),  # 30% dropout rate
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
```
<br>

## 13. How do you use _Batch Normalization_ in a _Keras model_?

**Batch Normalization** is key to faster and more stable training in neural networks.

In Keras, you can incorporate **Batch Normalization** via the `BatchNormalization` layer and also via the `use_bias` argument with layers like `Dense` and `Conv2D`.

### Code Example: Keras Model with Batch Normalization

Here is the Python code:

```python
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization

model = Sequential()

# First hidden layer with Batch Normalization
model.add(Dense(64, input_dim=100))  # Assuming 100 input dimensions
model.add(BatchNormalization())
model.add(Activation('relu'))

# Second hidden layer with Batch Normalization(Python Module)
model.add(Dense(64))
model.add(BatchNormalization())

# Third hidden layer with Batch Normalization AND no bias term
model.add(Dense(64, use_bias=False))
model.add(BatchNormalization())

# Final output layer without Batch Normalization
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
<br>

## 14. What is a _custom layer_ in _Keras_ and how would you implement one?

**Custom layers** in Keras enable you to create **proprietary operations** or architectures **not pre-defined** in the standard libraries.

Implementing a custom layer typically involves handling input and output shapes, as well as the layer's core logic.

### Code Example: Custom Layer in Keras

Here is the Python code:

```python
import tensorflow as tf

class MyCustomLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None):
        super(MyCustomLayer, self).__init__()
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", 
                                      shape=[int(input_shape[-1]), self.output_dim])

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.kernel))

# Instantiate the layer
my_layer = MyCustomLayer(32, activation='relu')

# Test the layer
output = my_layer(tf.zeros([10, 5]))
assert output.shape == (10, 32)
```

In this code:

- `MyCustomLayer` inherits from `tf.keras.layers.Layer`.
- The constructor `__init__` sets layer attributes. Here, `output_dim` and `activation`. The activation parameter is optional. If you do not need or want to specify an activation function, set the default value to `None` and use the tf.keras.activations.get method as shown in the code.
- The method `build` is where layer weights are defined. Here, a 2D `kernel` weight shared by all inputs is created.
- The `call` method describes the layer's computation - in this case, a matrix multiplication followed by an activation function.

### Sequential vs. Functional API

You can instantiate this custom layer using both the **Sequential** and **Functional API**. The code below illustrates both:

#### Sequential API

```python
model = tf.keras.Sequential([MyCustomLayer(32, activation='relu')])
```

####  Functional API

```python
inputs = tf.keras.Input(shape=(5,))
x = MyCustomLayer(32, activation='relu')(inputs)
model = tf.keras.Model(inputs=inputs, outputs=x)
```
<br>

## 15. Discuss how you would construct a _residual network (ResNet)_ in _Keras_.

In **Keras**, the implementation of **Residual Networks (ResNets)** is straightforward, primarily due to the framework's modularity.

ResNets introduced a novel concept with the **identity shortcut connection**, or skip-connection, to ease training and enable the building of deeper networks. Keras simplifies the implementation of these connections using, for example, `Add` layers.

### Key Components of a ResNet Model

1. **Convolutional Blocks**: A sequence of convolutional and activation layers.
2. **Identity Blocks**: These contain at least three layers with a skip-connection to maintain the same feature map size.

### Code Example: ResNet Identity Block in Keras

Here is the Keras compatible Python code:

```python
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from keras.models import Model

def identity_block(X, f, filters):
    """
    Implementation of the identity block as defined by the ResNet architecture in Keras.
    
    Arguments:
    X -- input to the block
    f -- integer, specifying the shape of the CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers
    
    Returns:
    X -- output of the identity block
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value:
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

# To test the implementation
input_shape = (3, 256, 256)  # for RGB images
X_input = Input(input_shape)
X = identity_block(X_input, f=3, filters=[64, 64, 256])
model = Model(inputs = X_input, outputs = X)
model.summary()
```
<br>



#### Explore all 70 answers here 👉 [Devinterview.io - Keras](https://devinterview.io/questions/machine-learning-and-data-science/keras-interview-questions)

<br>

<a href="https://devinterview.io/questions/machine-learning-and-data-science/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fmachine-learning-and-data-science-github-img.jpg?alt=media&token=c511359d-cb91-4157-9465-a8e75a0242fe" alt="machine-learning-and-data-science" width="100%">
</a>
</p>

