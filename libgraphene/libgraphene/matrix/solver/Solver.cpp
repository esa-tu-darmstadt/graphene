#include "libgraphene/matrix/solver/Solver.hpp"

#include <spdlog/spdlog.h>

#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/solver/gauss-seidel/Solver.hpp"
#include "libgraphene/matrix/solver/ilu/Solver.hpp"
#include "libgraphene/matrix/solver/iterative-refinement/Solver.hpp"
#include "libgraphene/matrix/solver/pbicgstab/Solver.hpp"

namespace graphene::matrix::solver {
template <DataType Type>
std::unique_ptr<Solver<Type>> Solver<Type>::createSolver(
    const Matrix<Type> &matrix, std::shared_ptr<Configuration> config) {
  std::string solverName = config->solverName();
  spdlog::debug("Creating solver {}", solverName);

  if (auto gaussSeidelConfig =
          std::dynamic_pointer_cast<gaussseidel::Configuration>(config)) {
    return std::make_unique<gaussseidel::Solver<Type>>(matrix,
                                                       gaussSeidelConfig);
  } else if (auto irConfig =
                 std::dynamic_pointer_cast<iterativerefinement::Configuration>(
                     config)) {
    return std::make_unique<iterativerefinement::Solver<Type>>(matrix,
                                                               irConfig);
  } else if (auto iluConfig =
                 std::dynamic_pointer_cast<ilu::Configuration>(config)) {
    return std::make_unique<ilu::Solver<Type>>(matrix, iluConfig);
  } else if (auto pbicgstabConfig =
                 std::dynamic_pointer_cast<pbicgstab::Configuration>(config)) {
    return std::make_unique<pbicgstab::Solver<Type>>(matrix, pbicgstabConfig);
  } else {
    throw std::runtime_error("Unknown solver: " + solverName);
  }
}

template <DataType Type>
bool Solver<Type>::shouldUseMulticolor(MultiColorMode mode) const {
  switch (mode) {
    case MultiColorMode::On:
      return true;
    case MultiColorMode::Off:
      return false;
    case MultiColorMode::Auto:
      return matrix().hostMatrix().multicolorRecommended();
  }
  return false;
}

template class Solver<float>;

}  // namespace graphene::matrix::solver