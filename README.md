# RustVsPython in Machine Learning: A Performance Comparison
![Alt text](https://github.com/refedico/RustVsPython/blob/main/rustvspython.png)

This repository aims to compare the performance of Python and Rust in the field of Machine Learning (ML). As Python has been the go-to language for ML due to its extensive libraries and ease of use, Rust is emerging as a powerful alternative, offering potential advantages in performance and memory safety.

### Contents
    - Benchmark Tests: A series of benchmarks to measure the performance of Python and Rust in various ML tasks.
    - Sample Projects: Implementations of common ML algorithms in both Python and Rust, including data preprocessing, model training, and evaluation.
    - Performance Analysis: Detailed analysis and comparison of execution times, memory usage, and scalability between the two languages.

### ML algorithms implemented
    - Linear Regression
    - Clustering: K-means
    - DecisionTree
    - MLP Neural Networks

## Contributing

We welcome contributions from the community! If you'd like to help improve this project, here are some ways you can contribute:

### Implementing New Algorithms

You can implement additional machine learning algorithms in both Rust and Python. The goal is to expand our benchmark suite and provide a comprehensive comparison of performance between the two languages. Here’s how you can get started:

1. **Choose an Algorithm**: Select a machine learning algorithm that is not yet included in the repository. Some suggestions include:
   - Random Forests
   - Support Vector Machines
   - LLMs
   - ecc.

2. **Implement the Algorithm in Python**:
   - Create a new Python script or module in the `python` directory.
   - Ensure your implementation is efficient and makes use of most-used libraries (e.g., NumPy, scikit-learn, TensorFlow).

3. **Implement the Algorithm in Rust**:
   - Create a new Rust module in the `rust` directory.
   - Ensure your implementation is optimized for performance and makes use of appropriate crates (e.g., ndarray, tch-rs, candle).

4. **Benchmark the Implementations**:
   - Add your implementations to the benchmark suite.
   - Ensure that the benchmarking scripts can measure and compare the performance of both the Python and Rust implementations of your algorithm.

5. **Submit a Pull Request**:
   - Fork the repository and create a new branch for your contribution.
   - Commit your changes and push them to your fork.
   - Open a pull request to the main repository with a clear description of your changes and any relevant performance results.

### Guidelines for Contribution

- Ensure that your code is well-documented.
- Update the documentation as necessary to include your new algorithms and any changes to the setup or benchmarking process.

By contributing new algorithms and performance tests, you help make this repository a valuable resource for the community to understand the strengths and trade-offs of using Python and Rust in machine learning.

We look forward to your contributions!



## Getting Started

### Clone the Repository
First, clone the repository to your local machine:
```sh
git clone https://github.com/refedico/RustVsPython.git
cd RustVsPython

