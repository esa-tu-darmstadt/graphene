#pragma once
#include <poplar/Tensor.hpp>
#include <vector>

#include "libgraphene/common/Helpers.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/code/Function.hpp"
#include "libgraphene/dsl/code/Value.hpp"
#include "libgraphene/dsl/code/VertexTypes.hpp"
namespace graphene::codedsl {

enum class VertexKind { Vertex, MultiVertex, SupervisorVertex };

/**
 * @brief Represents a vertex in the CodeDSL language.
 */
class Vertex {
 public:
  /// Describes a (to be generated) member variable of a vertex. The variable
  /// can be either generated for a poplar tensor that is to be connected to it
  /// or an additional member variable that is not connected to a tensor.
  class MemberVarInfo {
    struct TensorMemberVar {
      TypeRef elementType;
      poplar::Tensor tensor;
      VertexInOutType::Direction direction;
    };
    struct UnconnectedMemberVar {
      TypeRef type;
      CTypeQualifiers qualifiers;
    };

   private:
    MemberVarInfo() = delete;
    MemberVarInfo(TensorMemberVar info) : member(info) {}
    MemberVarInfo(UnconnectedMemberVar info) : member(info) {}

   public:
    /// Creates an instance that describes a member variable that is generated
    /// for a poplar tensor. The resulting member variable will be connected to
    /// the tensor.
    static MemberVarInfo create(TypeRef elementType, poplar::Tensor tensor,
                                VertexInOutType::Direction dir) {
      return MemberVarInfo(TensorMemberVar{elementType, tensor, dir});
    }
    /// Creates an instance that describes a member variable that is not
    /// connected to a tensor. The resulting member variable will be an
    /// additional member variable of the vertex that poplar does not know
    /// about.
    static MemberVarInfo create(TypeRef type, CTypeQualifiers qualifiers) {
      return MemberVarInfo(UnconnectedMemberVar{type, qualifiers});
    }

    bool isTensorMemberVar() const {
      return std::holds_alternative<TensorMemberVar>(member);
    }
    bool isUnconnectedMemberVar() const {
      return std::holds_alternative<UnconnectedMemberVar>(member);
    }
    const TensorMemberVar& tensorMemberVar() const {
      assert(isTensorMemberVar());
      return std::get<TensorMemberVar>(member);
    }
    const UnconnectedMemberVar& unconnectedMemberVar() const {
      assert(isUnconnectedMemberVar());
      return std::get<UnconnectedMemberVar>(member);
    }

   private:
    std::variant<TensorMemberVar, UnconnectedMemberVar> member;
  };

  /**
   * @brief Constructs and emits a Vertex with a given name, types, directions,
   * and compute function generator.
   * @param name The name of the vertex.
   * @param memberVars The member variables of the vertex. This can be tensors
   * that are connected to the vertex or additional member variables that are
   * not connected to tensors.
   * @param computeFunctionGenerator A callback that generates the compute
   * function for the vertex. The callback must accept a vector of \ref Value,
   * one for each tensor and the additional member variables. The callback must
   * return a \ref Function named "compute" that returns a boolean.
   */
  Vertex(std::string name, std::vector<MemberVarInfo> memberVars,
         VertexKind kind,
         std::function<Function(std::vector<Value>)> computeFunctionGenerator);

  ~Vertex() { currentVertex = nullptr; }

  const std::vector<MemberVariable>& fields() const { return fields_; }

 private:
  /// Adds a function to the current vertex. This function is called by the
  /// constructor of the \ref Function class and registers the function with the
  /// current vertex.
  static void addFunctionToCurrentVertex(Function function);
  friend class Function;

  std::vector<MemberVariable> fields_;
  std::vector<Function> functions_;
  static Vertex* currentVertex;
};
}  // namespace graphene::codedsl