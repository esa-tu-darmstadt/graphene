#include "libgraphene/matrix/Matrix.hpp"

#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"

namespace graphene::matrix {
void Matrix::solve(Tensor &x, Tensor &b,
                   std::shared_ptr<solver::Configuration> &config) {
  auto solver = solver::Solver::createSolver(*this, config);
  solver->solve(x, b);
}

}  // namespace graphene::matrix