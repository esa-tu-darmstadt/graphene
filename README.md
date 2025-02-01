# Graphene Linear Algebra Framework
Graphene is a linear algebra framework for GraphCore IPUs. 

The framework provides a domain-specific language (DSL) for expressing algebraic algorithms. The DSL consists of C++ classes that can be used to express algorithms in a high-level, mathematical way. 

The users describes its algorithm in the DSL and compiles it to a host program using a standard C++ compiler. When the resulting program is run on the host, the algorithm is symbolically executed and translated into a Poplar dataflow graph. The graph is then compiled and executed on the IPU.

Altough symbolically executed, the DSL is designed to be used as if it were directly executed. This presents a particular challenge when dealing with memory transfers between the host and the IPU. Data can only be transferred to the IPU after the full graph has been compiled and is loaded onto the IPU. Thus, when memory is copied from host to the IPU (i.e. `HostValue::copyToTile`), the data is stored in the `Runtime` and copied to the IPU after the graph is compiled and loaded.

## Usage of the DSL
The basic building block of the DSL is the `Value` class. It represents an assignable tensor:
```C++
Runtime runtime(1 /* Number of IPUs */);
Value<float> a({1, 2, 3, 4});
If(a.reduce() == 0, [&]{
  // Condition is true
  a = a + 2;
}, [&]{
  // Condition is false
  Repeat(10, [&](){
    // Repeat 10 times
    a = a * 1;
  });
});
poplar::Engine engine = runtime.compileGraph();
runtime.loadAndRunEngine(engine);
```

## Matrices
Matrices in tile memory or host memory are represented by the `Matrix` or `HostMatrix` classes respectively. A `HostMatrix` can be constructed from a COO matrix triplet, or by loading a matrix from a matrix market file. The `HostMatrix` can be copied to the tile memory using the `copyToTile` method.
 
Internally, only a custom CRS format is currently supported. The custom CRS format uses a dense vector for the diagonal values.

## Solvers
Graphene provides a number of solvers for linear systems of equations:
- (Preconditioned) Bi-Conjugate Gradient Stabilized (BiCGStab)
- (Mixed Precision) Iterative Refinement
- Gauss-Seidel
- Incomplete LU Factorization without fill-in (ILU0)

The mixed precision iterative refinement solver is key to solving large systems of equations. Due to the lack of native double-precision support on the IPU, solvers must use single precision. The mixed precision iterative refinement solver uses double-word arithmetic for the solution vector and to compute its current residual. It iteratively minimizes the residual by calculating a correction vector using any of the other solvers in single precision.

## Codegen performance discussion
Issues:
- Reduction vertices produce suboptimal code because they reload the accumulator from memory for every iteration. This could be fixed by reordering the loops so that the reduction is the innermost loop, and then using a variable to store the accumulator.
- Use __builtin_assume to enable hardware loops where possible: TensorDSL lowering + solvers
Positives:
- Memset is utilized by the compiler if possible.