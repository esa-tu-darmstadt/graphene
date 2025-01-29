#include "libgraphene/dsl/code/Execute.hpp"

using namespace graphene;

void graphene::codedsl::ExecuteAsMapped(
    std::vector<poplar::Tensor> tensors, std::vector<TypeRef> tensorTypes,
    std::vector<VertexInOutType::Direction> directions, bool multiVertex,
    std::function<void(std::vector<Value>)> code, bool broadcastTensors) {
  if (tensorTypes.size() != tensors.size() ||
      directions.size() != tensors.size()) {
    throw std::runtime_error(
        "Number of tensors, types, and directions must "
        "match");
  }

  auto transformedCode = [&](std::vector<Value> memberVars) -> Function {
    if (multiVertex)
      return Function("compute", Type::BOOL, {Type::UINT32},
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
      return Function("compute", Type::BOOL, [&]() -> void {
        // Call the user-provided code with the member variables
        std::vector<Value> args;
        args.insert(args.end(), memberVars.begin(), memberVars.end());
        code(args);
        // Return true if the user code does not do so
        Return(true);
      });
  };

  // Emit the necessary includes
  CodeGen::emitInclude("poplar/Vertex.hpp", true);
  CodeGen::emitInclude("libtwofloat/arithmetics/double-word-arithmetic.hpp",
                       false);
  CodeGen::emitInclude("libtwofloat/operators.hpp", false);
  CodeGen::emitInclude("print.h", true);

  // Generate the vertex
  std::string vertexName = CodeGen::generateVertexName();
  Vertex vertex(vertexName, tensorTypes, directions, transformedCode);

  std::stringstream ss = CodeGen::reset();
  auto& graph = Context::graph();

  // Compile the vertex
  {
    // Write the vertex to a file
    std::filesystem::path srcPath =
        Runtime::instance().getExpressionStorageDir() / (vertexName + ".cpp");
    std::ofstream srcFile(srcPath);
    srcFile << ss.str();
    srcFile.close();

    std::string baseCmd = "popc -O3 -I" +
                          Runtime::instance().getTwoFloatSourceDir().string() +
                          " " + srcPath.string();
    std::string elfCmd =
        baseCmd + " -o " + srcPath.replace_extension(".elf").string();
    std::string asmCmd = baseCmd + " --target=ipu21 -S -o " +
                         srcPath.replace_extension(".S").string();
    spdlog::trace("Compiling vertex with command: {}", elfCmd);
    std::string irCmd = baseCmd + " --target=ipu21 --emit-llvm -o " +
                        srcPath.replace_extension(".ll").string();
    if (std::system(elfCmd.c_str()) != 0) {
      throw std::runtime_error("Failed to compile vertex");
    }
    spdlog::trace("Compiled vertex to: {}",
                  srcPath.replace_extension(".elf").string());
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

    graph.addCodelets(srcPath.replace_extension(".elf").string(),
                      poplar::CodeletFileType::Object);
  }

  // Get the tensor mappings
  std::vector<TileMapping> tensorMappings;
  for (const auto& tensor : tensors) {
    tensorMappings.push_back(
        TileMapping::fromPoplar(graph.getTileMapping(tensor)));
  }

  // Add an instance of the vertex to each tile
  DebugInfo di("CodeDSL");
  poplar::ComputeSet cs = graph.addComputeSet(di);
  for (size_t tile = 0; tile < graph.getTarget().getNumTiles(); tile++) {
    // Check if the vertex has any data mapped to this tile
    bool isEmpty = true;
    for (size_t i = 0; i < tensors.size(); i++) {
      if (!tensorMappings[i][tile].empty()) {
        isEmpty = false;
        break;
      }
    }

    if (isEmpty) continue;

    poplar::VertexRef v = graph.addVertex(cs, vertexName);
    graph.setTileMapping(v, tile);

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

  // Add the compute set to the program
  Context::program().add(poplar::program::Execute(cs, di));
}