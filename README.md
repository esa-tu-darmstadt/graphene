# Graphene Linear Algebra Framework

Graphene is an open-source linear algebra framework designed for high-performance sparse linear algebra computations on GraphCore IPUs. By leveraging two complementary domain-specific languages (DSLs), Graphene allows users to express complex algebraic algorithms in a form that is close to their mathematical notation. The framework automatically transforms these high-level expressions into optimized Poplar dataflow graphs and execution schedules for execution on the IPU. **Graphene is released under the AGPL v3 license.**

For further details and technical insights, please refer to our associated scientific publication: T. Noack, L. Krüger, A. Koch (2025) *Accelerating Sparse Linear Solvers on Intelligence Processing Units*. In: 39th IEEE International Parallel and Distributed Processing Symposium (IPDPS 2025)

## Overview

- **High-Level DSLs:**  
  - **CodeDSL:** A tile-centric language for expressing low-level operations and generating codelets.
  - **TensorDSL:** A global language for tensor operations, including elementwise operations, reductions, and broadcasting.
  
- **Automatic Graph Generation:**  
  The DSLs are symbolically executed to create optimized Poplar dataflow graphs, which are then compiled and executed on the IPU.

- **Optimized for IPU Architecture:**  
  Leverages the IPU's thousands of independent cores, high-bandwidth on-chip memory, and all-to-all communication fabric.

- **Advanced Solvers:**  
  Includes implementations of state-of-the-art solvers such as PBiCGStab, Gauss-Seidel, ILU0, and support for mixed-precision iterative refinement.

- **Extended Precision Techniques:**  
  Provides both software-emulated double precision (`Type::float64`) and double-word arithmetic (`Type::twofloat32`) to overcome the lack of native double-precision support on the IPU.

- **Efficient Matrix Management:**  
  Implements a custom CRS format with a dedicated dense vector for diagonal elements and employs a novel matrix reordering strategy to facilitate efficient halo exchanges between tiles.


## Domain-Specific Languages (DSLs)

Graphene provides two interlinked domain-specific languages that form the foundation of its programming model. The first, **CodeDSL**, is designed to represent computational kernels that execute directly on the IPU tiles. Each kernel is confined to the tile on which it runs, meaning it can only access data that has been explicitly mapped to that tile. This aligns with the IPU's architecture, in which each tile can only access its own local memory.

In contrast, **TensorDSL** operates on entire tensors, regardless of whether their data is distributed across multiple tiles or even multiple IPUs. TensorDSL supports elementwise operations and reductions over whole tensors, abstracting away the complexities of data distribution. TensorDSL expressions are automatically translated into CodeDSL kernels, allowing users to write high-level tensor operations without worrying about the underlying tile-level implementation.


## Tensor Types and Data Movement

Graphene distinguishes between three types of tensor storage, each corresponding to a different memory location. Computations can only be performed on Tensors stored in the IPU’s near-processor memory (tile memory). Data must be explicitly transferred between these storage types:

- **Tensor:**  
  Data stored directly in the IPU’s near-processor SRAM memory (tile memory). Represented by the `graphene::Tensor` class.

- **RemoteTensor:**  
  Data stored in the *remote memory*, which is DDR4 memory attached to the IPU. Represented by the `graphene::RemoteTensor` class.

- **HostTensor:**  
  Data stored in the host machine’s memory. Represented by the `graphene::HostTensor` class.

Unlike many frameworks that automatically manage data movement, Poplar and thus Graphene requires explicit transfers between these different storage locations:

- **From Host to DDR4:**  
  Use `HostTensor::copyToRemote` to transfer data from host memory to DDR4 memory.
- **From DDR4 to Tile (IPU):**  
  Use `RemoteTensor::copyToTile` to move data from DDR4 memory into the IPU’s SRAM.
- **From Tile to DDR4:**  
  Use `Tensor::copyToRemote` to copy data from the IPU to DDR4 memory.
- **From Tile to Host:**  
  Use `Tensor::copyToHost` to transfer data from the IPU back to host memory.


## Extended Precision

Due to the IPU's lack of native double-precision support, Graphene offers two extended precision types:
- **Software-Emulated Double Precision:**  
  Provides higher precision through software, though with a higher computational cost. Available as `Type::float64`.
- **Double-Word Arithmetic:**  
  Represents numbers as the sum of two single-precision values, achieving increased precision with lower overhead. Uses the  double-word arithmetic  proposed by Joldes et al. 2017 ([Tight and rigorous error bounds for basic building blocks of double-word arithmetic](https://doi.org/10.1145/3121432)). Implemented in our `twofloat` library. Available as `Type::twofloat32`.

These types are particularly important in the Mixed-Precision Iterative Refinement (MPIR) method, ensuring that high-precision solutions can be obtained even on hardware limited to single precision.


## Usage and Example

Below is a simple example demonstrating how to write and execute a DSL-based algorithm with Graphene:

```cpp
#include "libgraphene/dsl/tensor/ControlFlow.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/util/Runtime.hpp"

using namespace graphene;
using namespace graphene::cf;

int main() {
  // Initialize the runtime with 1 IPU
  Runtime runtime(1);

  // Define a tensor 'a' with initial values
  Tensor a = Tensor::withInitialValue({1.0f, 2.0f, 3.0f, 4.0f});

  // Use a conditional construct to modify 'a'
  If(
      a.reduce() == 0,
      [&] {
        // True branch: Increment each element by 2
        a = a + 2;
      },
      [&] {
        // False branch: Multiply 'a' by 1, repeated 10 times
        Repeat(10, [&]() { a = a * 1; });
      });

  // Compile the generated graph and run it on the IPU
  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);

  return 0;
}
```

Feel free to look at the [benchmark](applications/benchmark/benchmark.cpp) application to find out how matrices are loaded and solvers are configured in Graphene.

## Compilation & Execution Pipeline

Graphene’s build and execution process involves several stages:

1. **Compilation for the CPU:**  
   The DSL code is compiled into a CPU executable using a C++20 compiler.

2. **Symbolic Execution:**  
   The executable symbolically runs the DSL code to generate a Poplar dataflow graph, an execution schedule, and codelets.

3. **Graph Compilation:**  
   The Poplar compiler optimizes the dataflow graph and translates it into machine code for the IPU.

4. **Concrete Execution:**  
   The generated graph program is executed on the IPU or in a simulator, with support for CPU callbacks to handle data transfers and mixed execution.

## Solvers and Preconditioners

Graphene includes a suite of solvers optimized for sparse linear systems:

- **Preconditioned Bi-Conjugate Gradient Stabilized (PBiCGStab):**  
  A robust Krylov subspace solver for both symmetric and nonsymmetric systems.

- **Mixed-Precision Iterative Refinement (MPIR):**  
  Enhances solver precision by combining single-precision operations with extended precision techniques.

- **Gauss-Seidel:**  
  A simple iterative solver useful both as a standalone method and as a smoother in multigrid algorithms.

- **Incomplete LU Factorization (ILU0) & Diagonal-Based ILU (DILU):**  
  Preconditioners that approximate LU decomposition while preserving the matrix sparsity.

These solvers can be configured via JSON files, allowing for nested solver strategies and flexible preconditioning schemes.

---

## License

Graphene is distributed under the **AGPL v3 license**. Please refer to the [LICENSE](LICENSE) file for complete details.

