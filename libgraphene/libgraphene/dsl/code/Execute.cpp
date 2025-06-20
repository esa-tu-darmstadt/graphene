/*
 * Graphene Linear Algebra Framework for Intelligence Processing Units.
 * Copyright (C) 2025 Embedded Systems and Applications, TU Darmstadt.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "libgraphene/dsl/code/Execute.hpp"

#include <cstring>
#include <poplar/Target.hpp>
#include <poplar/TargetType.hpp>

#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/dsl/code/ControlFlow.hpp"
#include "libgraphene/dsl/code/Function.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"
#include "libgraphene/util/Runtime.hpp"
#include "libgraphene/util/VertexCache.hpp"

using namespace graphene;

namespace {
std::string getCompileTargetName() {
  TargetType targetType = Runtime::instance().getTargetType();
  switch (targetType) {
    case TargetType::CPU:
      return "cpu";
    case TargetType::IPU1:
      return "ipu1";
    case TargetType::IPU2:
      return "ipu2";
    case TargetType::IPU21:
      return "ipu21";
    default:
      throw std::runtime_error("Unknown target type");
  }
}
}  // namespace

void graphene::codedsl::ExecuteAsMapped(
    std::vector<Vertex::MemberVarInfo> vars, VertexKind kind,
    std::function<void(std::vector<Value>)> code, bool broadcastTensors,
    std::optional<size_t> targetTile, bool ipuOnly) {
  if (ipuOnly && Runtime::instance().getTargetType() == TargetType::CPU) {
    throw std::runtime_error("Trying to execute IPU-only code on CPU");
  }

  auto transformedCode = [&](std::vector<Value> memberVars) -> Function {
    if (kind == VertexKind::MultiVertex)
      return Function("compute", Type::BOOL, {Type::UINT32}, ThreadKind::Worker,
                      [&](Parameter workerID) -> void {
                        // Call the user-provided code with the worker ID and
                        // member variables
                        std::vector<Value> args;
                        args.push_back(workerID);
                        args.insert(args.end(), memberVars.begin(),
                                    memberVars.end());
                        code(args);

                        // Return true if the user code does not do so
                        Return(true);
                      });
    else
      return Function(
          "compute", Type::BOOL,
          kind == VertexKind::SupervisorVertex ? ThreadKind::Supervisor
                                               : ThreadKind::Worker,
          [&]() -> void {
            // Call the user-provided code with the member variables
            std::vector<Value> args;
            args.insert(args.end(), memberVars.begin(), memberVars.end());
            code(args);
            // Return true if the user code does not do so
            Return(true);
          });
  };

  // Emit the necessary includes
  CodeGen::emitInclude("ipu_intrinsics", true, true);
  CodeGen::emitInclude("poplar/Vertex.hpp", true);
  CodeGen::emitInclude("array", true);
  CodeGen::emitInclude("libtwofloat/arithmetics/double-word-arithmetic.hpp",
                       false);
  CodeGen::emitInclude("libtwofloat/operators.hpp", false);
  CodeGen::emitInclude("ipu-thread-sync/ipu-thread-sync.hpp", false, true);
  CodeGen::emitInclude("print.h", true);

  // Emit definitions for worker and supervisor function attributes. They are
  // only supported on IPU targets!
  CodeGen::emitCode("#ifdef __IPU__\n");
  CodeGen::emitCode("#define WORKERFUNC __attribute__((target(\"worker\")))\n");
  CodeGen::emitCode(
      "#define SUPERVISORFUNC __attribute__((target(\"supervisor\")))\n");
  CodeGen::emitCode("#else\n");
  CodeGen::emitCode("#define WORKERFUNC\n");
  CodeGen::emitCode("#define SUPERVISORFUNC\n");
  CodeGen::emitCode("#endif // __IPU__\n");

  // Generate the vertex with a placeholder name. This enables us to generate a
  // hash of the code, independent of the vertex name. The actual vertex name
  // will be set after hashing.
  std::string vertexName = VertexCache::getVertexNamePlaceholder();
  Vertex vertex(vertexName, vars, kind, transformedCode);

  std::stringstream ss = CodeGen::reset();
  std::string codeStr = ss.str();

  size_t hash = std::hash<std::string>{}(codeStr);

  if (std::optional<VertexCache::Entry> cached = VertexCache::lookup(hash)) {
    // cache hit
    vertexName = cached->vertexName;
    spdlog::trace("Restored vertex {} from cache {}", vertexName,
                  cached->srcPath.string());
  } else {
    // cache miss, compile it and insert it into the cache
    vertexName = VertexCache::getUniqueVertexName(hash);
    CodeGen::replaceVertexNamePlaceholder(
        codeStr, VertexCache::getVertexNamePlaceholder(), vertexName);

    // Write the vertex to a file
    std::filesystem::path srcPath = VertexCache::getVertexSourcePath(hash);

    std::ofstream srcFile(srcPath);
    srcFile << codeStr;
    srcFile.close();

    // Compile the vertex
    std::filesystem::path objectFilePath =
        VertexCache::getVertexObjectPath(hash);
    std::string baseCmd = "popc -O3 " + srcPath.string();
    baseCmd +=
        " -I" + Runtime::instance()
                    .getRuntimeLibIncludeDir(Runtime::RuntimeLib::TwoFloat)
                    .string();
    baseCmd +=
        " -I" + Runtime::instance()
                    .getRuntimeLibIncludeDir(Runtime::RuntimeLib::IpuThreadSync)
                    .string();
    // When we are REALLY naughty, we can enable this. This will likely break
    // double-word arithmetics completely..
    // baseCmd += " -X -ffast-math";

    std::string targetString = "ipu2,ipu21";
    if (!ipuOnly) targetString += ",cpu";

    std::string elfCmd = baseCmd + " --target=" + targetString + " -o " +
                         objectFilePath.string();
    std::string asmCmd = baseCmd + " --target=ipu21 -S -o " +
                         srcPath.replace_extension(".S").string();
    spdlog::trace("Compiling vertex with command: {}", elfCmd);
    std::string irCmd = baseCmd + " --emit-llvm --target=ipu21 -o " +
                        srcPath.replace_extension(".ll").string();
    if (std::system(elfCmd.c_str()) != 0) {
      throw std::runtime_error("Failed to compile vertex");
    }
    spdlog::trace("Compiled vertex to: {}", objectFilePath.string());
    if (Runtime::instance().dumpExpressionAsm()) {
      if (std::system(asmCmd.c_str()) != 0) {
        throw std::runtime_error("Failed to compile vertex to assembly");
      }
      spdlog::trace("Stored vertex asm in: {}",
                    srcPath.replace_extension(".S").string());
    }
    if (Runtime::instance().dumpExpressionIR()) {
      if (std::system(irCmd.c_str()) != 0) {
        throw std::runtime_error("Failed to compile vertex to IR");
      }
      spdlog::trace("Stored vertex IR in: {}",
                    srcPath.replace_extension(".ll").string());
    }

    // Insert the vertex into the cache
    VertexCache::insert(hash, vertexName);

    spdlog::trace("Vertex {} compiled and cached", vertexName);
  }

  // Add the codelet to the graph
  auto& graph = Context::graph();
  graph.addCodelets(VertexCache::getVertexObjectPath(hash).string(),
                    poplar::CodeletFileType::Object);

  // Get the tensor mappings
  std::vector<TileMapping> tensorMappings;
  std::vector<poplar::Tensor> tensors;
  for (const auto& var : vars) {
    if (var.isTensorMemberVar())
      tensors.push_back(var.tensorMemberVar().tensor);
  }
  for (const auto& tensor : tensors) {
    tensorMappings.push_back(
        TileMapping::fromPoplar(graph.getTileMapping(tensor)));
  }

  // Add an instance of the vertex to each tile
  Context::Execute exec({"CodeDSL"});
  for (size_t tile = 0; tile < graph.getTarget().getNumTiles(); tile++) {
    // Common, you can do better than this
    if (targetTile.has_value() && tile != targetTile.value()) continue;

    // Check if the vertex has any data mapped to this tile
    bool isEmpty = true;
    for (size_t i = 0; i < tensors.size(); i++) {
      if (!tensorMappings[i][tile].empty()) {
        isEmpty = false;
        break;
      }
    }

    if (isEmpty) continue;

    poplar::VertexRef v = graph.addVertex(exec.computeSet(), vertexName);
    graph.setTileMapping(v, tile);
    graph.setPerfEstimate(v, 10000);  // FIXME: Set a reasonable estimate

    size_t longestRank =
        std::max_element(tensors.begin(), tensors.end(),
                         [](const poplar::Tensor& a, const poplar::Tensor& b) {
                           return a.rank() < b.rank();
                         })
            ->rank();

    // Connect the vertex to the tensors
    for (size_t i = 0; i < tensors.size(); i++) {
      // Either broadcast or slice the tensor to the tile
      bool broadcastThisTensor =
          broadcastTensors &&
          (tensors[i].dim(0) == 1 || tensors[i].rank() < longestRank);
      poplar::Tensor localTensor =
          broadcastThisTensor
              ? tensors[i]
              : sliceTensorToTile(tensors[i], tile, tensorMappings[i]);
      graph.connect(v[vertex.fields()[i].expr()], localTensor.flatten());
    }
  }
}