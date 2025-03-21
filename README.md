# Graphene Linear Algebra Framework

Graphene is an open-source linear algebra framework designed for high-performance sparse linear algebra computations on GraphCore IPUs.

Existing machine learning frameworks for the IPU, such as JAX or PyTorch, automatically distribute tensors and computations across available processors. Such automatic parallelization is effective for algorithms whose results remain consistent regardless of how computations are divided across multiple processors. However, in iterative matrix solvers, parallelizing the computations usually leads to differences in the numerical results of individual iterations, even though (hopefully) converging to the same solution. Graphene addresses this by giving users explicit control over how computations and tensors are distributed across the IPU tiles if needed. 

However, Graphene automatically partitions matrices row-wise across IPU tiles. Using this partitioning, the framework automatically determines the correct tensor distributions — known as `DistributedShape` and `TileMapping` — for matrix-related tensors. As a result, developers writing iterative solvers generally do not need to manage how matrices and vectors are distributed across IPU tiles manually.

For further details and technical insights, please refer to our associated scientific publication: T. Noack, L. Krüger, A. Koch (2025) *Accelerating Sparse Linear Solvers on Intelligence Processing Units*. In: 39th IEEE International Parallel and Distributed Processing Symposium (IPDPS 2025)


## Usage and Example

To build Graphene, you need to have the Poplar SDK installed and sourced in your environment. For a controlled environment, you can use the provided [Dockerfile](.devcontainer/Dockerfile) or open the project in a [Visual Studio Code Dev Container](https://code.visualstudio.com/docs/remote/containers).

 The following steps will guide you through the process:
1. Clone the repository recursively to include the submodules:
   ```bash
   git clone --recursive https://github.com/esa-tu-darmstadt/graphene.git
    ```
2. Install the required dependencies:
   ```bash
   sudo apt-get install libmetis-dev
   ```
3. Build the library using CMake:
   ```bash
    cmake -B build -DPOPLIBS_ENABLED_IPU_ARCH_NAMES=ipu2,ipu21
    cmake --build build
    ```
This will build the Graphene library and the associated applications for the IPU2 and IPU21 architectures. 

Below is a simple example demonstrating how to write and execute a DSL-based algorithm with Graphene:

```cpp
#include "libgraphene/common/Shape.hpp"
#include "libgraphene/dsl/tensor/ControlFlow.hpp"
#include "libgraphene/dsl/tensor/HostTensor.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/dsl/tensor/RemoteTensor.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/util/Runtime.hpp"

using namespace graphene;
using namespace graphene::cf;

int main() {
  // Initialize the runtime with 1 IPU
  Runtime runtime(1);

  // Define a tensor 'a' with initial values mapped to a tile 42
  DistributedShape shape = DistributedShape::onSingleTile({4}, 42);
  Tensor a = Tensor::withInitialValue({1.0f, -1.0f, 2.0f, -2.0f}, shape);

  // Copy a to remote memory and back
  RemoteTensor remoteA = a.copyToRemote();
  Tensor b = remoteA.copyToTile();

  // Copy b back to host memory
  b.copyToHost([](const HostTensor &hostB) {
    std::cout << hostB.numElements() << " elements available on host"
              << std::endl;
  });

  // Print b to stdout:
  b.print("b");

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

Feel free to look at the [benchmark](applications/benchmark/benchmark.cpp) application to find out how matrices are loaded and solvers are configured in Graphene. For a real-world example of TensorDSL, check out the source of the [PBiCGStab solver](libgraphene/libgraphene/matrix/solver/pbicgstab/Solver.cpp). For a CodeDSL example, check out the implementation of the [SpMV kernel](libgraphene/libgraphene/matrix/details/crs/CRSMatrix.cpp).

## Compilation & Execution Pipeline
When working with Graphene, it is essential to understand how the framework compiles and executes DSL code on the IPU:

1. **Compilation for the CPU:**  
   The DSL code (your C++ code) is compiled into a CPU executable using a CMake and standard C++20 compiler.

2. **Symbolic Execution:**  
   The CPU symbolically executes the DSL code to generate a Poplar dataflow graph, an execution schedule, and codelets. This involves all of your code before the `runtime.compileGraph()` call. It is important to understand that CodeDSL and TensorDSL operations are not actually executed at this stage, but rather generate a symbolic representation of the computation (the dataflow graph). Thus, a function like `Tensor::copyToHost` does not actually copy data to the host at this point, but registers a callback to be executed when the graph is run.

3. **Graph Compilation:**  
   The Poplar compiler optimizes the dataflow graph and translates it into machine code for the IPU (`runtime.compileGraph()`).

4. **Concrete Execution:**  
   The generated graph program is executed on the IPU or in a simulator, with support for CPU callbacks to handle data transfers and mixed execution (`runtime.loadAndRunEngine(engine)`).

## Domain-Specific Languages (DSLs)
Graphene employs two interconnected DSLs that bridge the gap between mathematical notation and IPU execution:

* **CodeDSL** is tile-centric and focuses on computational kernels running directly on individual IPU tiles. It provides precise control over memory and computations local to each tile, aligning with the IPU’s tile-centric architecture.
* **TensorDSL** offers a global perspective, enabling tensor-level operations such as elementwise computations, reductions, and broadcasting across multiple tiles or IPUs. It is similiar to the tensor operations in popular frameworks like TensorFlow or PyTorch, altough still allowing for explicit control over data distribution if needed.

## Tensor Types and Data Movement

Graphene distinguishes between three types of tensor storage, each corresponding to a different memory location. Computations can only be performed on Tensors stored in the IPU’s near-processor memory (tile memory). Data must be explicitly transferred between these storage types:

- **Tensor:**  
  Data stored directly in the IPU’s near-processor SRAM memory (tile memory). Represented by the `graphene::Tensor` class.

- **RemoteTensor:**  
  Data stored in the *remote memory*, which is DDR4 memory attached to the IPU. Represented by the `graphene::RemoteTensor` class.

- **HostTensor:**  
  Data stored in the host machine’s memory. Represented by the `graphene::HostTensor` class.

Unlike many frameworks or hardware that automatically manage data movement, Poplar and thus Graphene requires explicit transfers between these different storage locations:

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

