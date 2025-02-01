#include "libgraphene/matrix/solver/Solver.hpp"

#include <spdlog/spdlog.h>

#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/solver/gauss-seidel/Solver.hpp"
#include "libgraphene/matrix/solver/ilu/Solver.hpp"
#include "libgraphene/matrix/solver/iterative-refinement/Solver.hpp"
#include "libgraphene/matrix/solver/pbicgstab/Solver.hpp"
#include "libgraphene/matrix/solver/restarter/Solver.hpp"

namespace graphene::matrix::solver {
std::unique_ptr<Solver> Solver::createSolver(
    const Matrix &matrix, std::shared_ptr<Configuration> config) {
  std::string solverName = config->solverName();
  spdlog::debug("Creating solver {}", solverName);

  if (auto gaussSeidelConfig =
          std::dynamic_pointer_cast<gaussseidel::Configuration>(config)) {
    return std::make_unique<gaussseidel::Solver>(matrix, gaussSeidelConfig);
  } else if (auto irConfig =
                 std::dynamic_pointer_cast<iterativerefinement::Configuration>(
                     config)) {
    return std::make_unique<iterativerefinement::Solver>(matrix, irConfig);
  } else if (auto iluConfig =
                 std::dynamic_pointer_cast<ilu::Configuration>(config)) {
    return std::make_unique<ilu::Solver>(matrix, iluConfig);
  } else if (auto pbicgstabConfig =
                 std::dynamic_pointer_cast<pbicgstab::Configuration>(config)) {
    return std::make_unique<pbicgstab::Solver>(matrix, pbicgstabConfig);
  } else if (auto restarterConfig =
                 std::dynamic_pointer_cast<restarter::Configuration>(config)) {
    return std::make_unique<restarter::Solver>(matrix, restarterConfig);
  } else {
    throw std::runtime_error("Unknown solver: " + solverName);
  }
}

bool Solver::shouldUseMulticolor(MultiColorMode mode) const {
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

}  // namespace graphene::matrix::solver