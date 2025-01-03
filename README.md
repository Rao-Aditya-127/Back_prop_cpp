# Back_prop_cpp
Back_prop_cpp is a C++ implementation of an automatic differentiation engine. It computes forward passes to evaluate expressions and performs reverse-mode differentiation (backpropagation) to compute gradients, which is essential in machine learning.

## Features

- **Automatic Differentiation**: Implements reverse-mode differentiation for scalar computations.
- **Customizable Computational Graph**: Allows constructing and visualizing computational graphs.
- **Gradient Propagation**: Computes gradients of expressions involving multiple operations.
- **Supported Operations**:
  - Addition, subtraction, multiplication, division
  - Exponential and power functions
  - Hyperbolic tangent (`tanh`)

## File Structure

- **`value.h`**: Header file defining the `value` class and its methods.
- **`value.cpp`**: Implementation of the `value` class methods, including operations, gradient computation, and graph traversal.
- **`main.cpp`**: Demonstrates the functionality of the library by constructing a computational graph and performing backpropagation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Rao-Aditya-127/Back_prop_cpp.git
   cd Back_prop_cpp
   ```

2. Ensure your compiler supports C++17 or later.

## Usage

1. Compile the code:

   ```bash
    g++ main.cpp value.cpp -o backprop
   ```

2. Run the executable:

   ```bash
   ./backprop
   ```


### Sample Code

Here is a snippet demonstrating the library's usage:

```cpp
value x1("x1", 2.0), x2("x2", 0.0), b("b", 6.8813735), w1("w1", -3.0), w2("w2", 1.0);
    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = *x1w1 + *x2w2;
    auto n = *x1w1x2w2 + b;

    auto x = *n * 2;
    auto e = (*x).exp();
    auto f = *e - 1;
    auto g = *e + 1;

    auto inverse = g->val_pow(-1);
    auto out = *f * *inverse;

    out->grad = 1.0;
    out->backward();

    unordered_set<const value*> visited;
    value::printGraph(*out, "", true, visited);
```

## Example Output

The program constructs a computational graph and prints its structure, including data, operations, gradients, and memory addresses of nodes.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1ebd7001-c90f-41a8-94b9-ec669fe786bf" alt="image">
</p>


## Contributing

Contributions are welcome! Please open an issue or submit a pull request to contribute to this project.

## Acknowledgments

This project is inspired by Andrej Karpathy's YouTube video. You can find it [here](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6577s).
