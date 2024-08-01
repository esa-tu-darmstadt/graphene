#include "libgraphene/matrix/Matrix.hpp"

#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/solver/Solver.hpp"

namespace graphene::matrix {
template <DataType Type>
void Matrix<Type>::solve(Value<Type> &x, Value<Type> &b,
                         std::shared_ptr<solver::Configuration> &config) {
  auto solver = solver::Solver<Type>::createSolver(*this, config);
  solver->solve(x, b);
}

// Explicit instantiation
template class Matrix<float>;
}  // namespace graphene::matrix