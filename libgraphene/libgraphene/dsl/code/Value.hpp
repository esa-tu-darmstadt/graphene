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

#pragma once

#include <cstdint>
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
   * @brief Constructs a Value from a literal. The type of the Value is inferred
   * from the literal.
   * @tparam T The type of the literal.
   * @param value The literal value.
   */
  template <DataType T>
  Value(T value)
      : Value(Value(Type::VOID, std::to_string(value)).cast(getType<T>())) {}

  // 64-bit integer literals (which are "default" on 64-bit host systems) are
  // converted to 32-bit, so that the user does not need to write `(uint32_t)1`
  // everywhere. Consequently, it is currently not possible to create a 64-bit
  // integer Value.
  Value(int64_t value) : Value(static_cast<int32_t>(value)) {
    if (value != static_cast<int64_t>(static_cast<int32_t>(value))) {
      throw std::runtime_error(
          "Integer overflow while converting this literal to 32-bit integer");
    }
  }
  Value(uint64_t value) : Value(static_cast<uint32_t>(value)) {
    if (value != static_cast<uint64_t>(static_cast<uint32_t>(value))) {
      throw std::runtime_error(
          "Integer overflow while converting this literal to 32-bit unsigned "
          "integer");
    }
  }

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

  /**
   * @brief Reinterprets the value as a different type.
   * @param type The type to reinterpret as.
   * @return A new Value with the reinterpreted type.
   */
  Value reinterpretCast(TypeRef type);

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

  /**
   * @brief Constructs a non-const Variable with a given type.
   * @param type The type of the variable.
   */
  explicit Variable(TypeRef type, CTypeQualifiers qualifiers = {});

  /**
   * @brief Constructs a Variable with a given type and initializer.
   * @param type The type of the variable.
   * @param initializer The initializer Value.
   */
  Variable(TypeRef type, Value initializer, CTypeQualifiers qualifiers = {});

  /**
   * @brief Constructs a Variable from an initializer Value.
   * @param initializer The initializer Value.
   */
  Variable(Value initializer, CTypeQualifiers qualifiers = {});

  /**
   * @brief Constructs a Variable with a given initial literal value.
   * @tparam T The type of the literal.
   * @param value The literal value.
   */
  template <DataType T>
  Variable(T value, CTypeQualifiers qualifiers = {})
      : Variable(Value(value), qualifiers) {}
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
  MemberVariable(TypeRef type, CTypeQualifiers qualifiers = {});

  /**
   * @brief Constructs a MemberVariable with a given type and initializer.
   * @param type The type of the member variable.
   * @param initializer The initializer Value.
   */
  MemberVariable(TypeRef type, Value initializer,
                 CTypeQualifiers qualifiers = {});

  /**
   * @brief Emits the declaration of the member variable.
   */
  void declare() const;

 private:
  std::optional<Value> initializer_;
  CTypeQualifiers qualifiers_;
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