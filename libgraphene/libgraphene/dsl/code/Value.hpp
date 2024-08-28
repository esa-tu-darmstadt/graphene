#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Type.hpp"

namespace graphene::codedsl {

/**
 * @brief Represents a value in the CodeDSL language.
 */
class Value {
 public:
  /**
   * @brief Constructs a Value from a literal.
   * @tparam T The type of the literal.
   * @param value The literal value.
   */
  template <DataType T>
  Value(T value)
      : Value(Value(Type::VOID, std::to_string(value)).cast(getType<T>())) {}

  /**
   * @brief Gets a void value.
   * @return A Value representing void.
   */
  static Value getVoid();

  /** @brief Returns the type of the value. */
  TypeRef type() const;

  /** @brief Returns the expression string of the value. */
  std::string expr() const;

  /** @brief Emits the value in the generated code. */
  void emitValue() const;

  /**
   * @brief Casts the value to a different type.
   * @param type The type to cast to.
   * @return A new Value with the casted type.
   */
  Value cast(TypeRef type);

  /// ------------------ Functions ------------------
  /// These are the functions that may be implemented by a type.
  /// If a function is not implemented by the type, calling it will throw an
  /// exception.
  Value operator[](Value index);
  Value size() const;

  /**
   * @brief Assignment operator for assignable values.
   * @param other The Value to assign.
   * @return A reference to this Value.
   */
  void assign(const Value& other, bool endWithSemicolon = true);
  Value& operator=(const Value& other) {
    assign(other);
    return *this;
  }

 protected:
  explicit Value(TypeRef type, std::string expr, bool isAssignable = false);

 private:
  std::string expr_;
  TypeRef type_;
  bool isAssignable_ = false;
};

/**
 * @brief Represents a variable in the CodeDSL language.
 */
class Variable : public Value {
 public:
  using Value::operator=;
  using Value::Value;

  /**
   * @brief Constructs a Variable with a given type.
   * @param type The type of the variable.
   */
  Variable(TypeRef type);

  /**
   * @brief Constructs a Variable with a given type and initializer.
   * @param type The type of the variable.
   * @param initializer The initializer Value.
   */
  Variable(TypeRef type, Value initializer);

  /**
   * @brief Constructs a Variable from an initializer Value.
   * @param initializer The initializer Value.
   */
  Variable(Value initializer);
};

/**
 * @brief Represents a member variable in the CodeDSL language.
 * @details In contrast to \ref Variable, the declaration of a MemberVariable
 * must be requested explicitly.
 */
class MemberVariable : public Value {
 public:
  /**
   * @brief Constructs a MemberVariable with a given type.
   * @param type The type of the member variable.
   */
  MemberVariable(TypeRef type);

  /**
   * @brief Constructs a MemberVariable with a given type and initializer.
   * @param type The type of the member variable.
   * @param initializer The initializer Value.
   */
  MemberVariable(TypeRef type, Value initializer);

  /**
   * @brief Emits the declaration of the member variable.
   */
  void declare() const;

 private:
  std::optional<Value> initializer_;
};

/**
 * @brief Represents a function parameter in the CodeDSL language.
 */
class Parameter : public Value {
 public:
  /**
   * @brief Constructs a Parameter with a given index and type.
   * @param index The index of the parameter.
   * @param type The type of the parameter.
   */
  Parameter(int index, TypeRef type);
};

/**
 * @brief Represents a void value in the CodeDSL language.
 */
class Void : public Value {
 public:
  Void();
};

/**
 * @brief Represents an expression in the CodeDSL language.
 * @details Expressions are not assignable.
 */
class Expression : public Value {
 public:
  /**
   * @brief Constructs an Expression with a given type and expression string.
   * @param type The type of the expression.
   * @param expr The expression string.
   */
  Expression(TypeRef type, std::string expr);
};

}  // namespace graphene::codedsl