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

#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace graphene {

/**
 * @brief A specialized associative container implementing map-like semantics
 * using vector storage
 *
 * VectorMap provides O(1) or O(N) access time for key-value pairs where keys
 * are size_t indices, depending. The implementation utilizes a vector as the
 * underlying storage mechanism, where the index serves as the key. This
 * approach optimizes for scenarios where:
 *   1. Keys are dense or semi-dense size_t values
 *   2. Direct index-based access is the primary operation
 *   3. Memory overhead of sparse standard maps is undesirable
 *
 * Zero values (T{}) represent absent elements, enabling sparse storage while
 * maintaining vector-based performance characteristics.
 *
 * @tparam T The value type stored in the container
 *
 * Implementation Notes:
 * - Vector resizing occurs automatically when accessing indices beyond current
 * capacity
 * - Iterator implementation skips zero values, providing map-like iteration
 * semantics
 * - Memory efficiency depends on key density and value type size
 *
 * Time Complexity:
 * - Access (operator[]): O(1) amortized
 * - Insertion: O(1) amortized for push_back, O(n) for mid-container resize
 * - Iteration: O(n) where n is the vector size
 */
template <typename T>
class VectorMap {
 public:
  using key_type = std::size_t;
  using mapped_type = T;

  // A "map-like" value_type is pair<const key_type, mapped_type>
  // but physically we only store T in the vector.
  using value_type = std::pair<const key_type, mapped_type>;

 private:
  std::vector<T> data_;
  //------------------------------------------------
  //  A single base class that can be either const
  //  or non-const, controlled by IsConst.
  //------------------------------------------------
  template <bool IsConst>
  class iterator_base {
   private:
    // 1) Figure out which vector type we refer to.
    using container_type =
        std::conditional_t<IsConst, const std::vector<T>, std::vector<T>>;

    // 2) Pick the right iterator type (const or non-const).
    using base_iter =
        std::conditional_t<IsConst, typename std::vector<T>::const_iterator,
                           typename std::vector<T>::iterator>;

    // 3) The "mapped type" we yield is T or const T.
    using mapped_ref = std::conditional_t<IsConst, const T, T>;

    //------------------------------------------------
    // Because we want to skip zero-values, we store:
    //  - an actual iterator (`current_`)
    //  - a sentinel (`end_`)
    //  - the numeric index (`index_`)
    //------------------------------------------------
    base_iter current_;
    base_iter end_;
    key_type index_;

    //------------------------------------------------
    // We'll do the same "proxy reference" trick:
    //   a pair<const key_type, (const) T&>
    // for the "non-const" iterator, that’s `T&`.
    // for the "const" iterator, that’s `const T&`.
    //
    // We'll store it inside a union buffer, so
    // operator* can return a stable reference.
    //------------------------------------------------
    struct ProxyRef {
      const key_type key;
      mapped_ref& value;
      // Allow conversion to a real std::pair for convenience
      operator std::pair<const key_type, mapped_ref&>() const {
        return {key, value};
      }
    };

    // We'll hold that ProxyRef in a raw buffer inside the iterator
    // so it has a stable address for `operator*()`.
    mutable std::aligned_storage_t<sizeof(ProxyRef), alignof(ProxyRef)> cache_;

    // Helper to interpret our buffer as ProxyRef
    ProxyRef& cacheRef() const {
      return *reinterpret_cast<ProxyRef*>(
          const_cast<void*>(static_cast<const void*>(&cache_)));
    }

    // Skip all zeroes
    void skipZeros() {
      while (current_ != end_ && *current_ == T{}) {
        ++current_;
        ++index_;
      }
    }

   public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    // value_type is always pair<const key_type, T> (like a map).
    // But for reference, we yield a ProxyRef& or
    // a ProxyRef& with const T inside.
    using value_type = VectorMap::value_type;
    using reference = ProxyRef&;
    using pointer = ProxyRef*;

    // Default ctor -> singular iterator
    iterator_base() : current_(), end_(), index_(0) {}

    // Real ctor
    iterator_base(base_iter curr, base_iter end, key_type idx)
        : current_(curr), end_(end), index_(idx) {
      skipZeros();
    }

    // operator* -> re-init the in-place ProxyRef, return a reference
    reference operator*() const {
      // Construct the ProxyRef in place.
      new (&cache_) ProxyRef{index_, *current_};
      return cacheRef();
    }

    // operator-> -> address of that ProxyRef
    pointer operator->() const { return &(**this); }

    // Pre-increment
    iterator_base& operator++() {
      if (current_ != end_) {
        ++current_;
        ++index_;
        skipZeros();
      }
      return *this;
    }
    // Post-increment
    iterator_base operator++(int) {
      iterator_base tmp(*this);
      ++(*this);
      return tmp;
    }

    bool operator==(const iterator_base& other) const {
      return current_ == other.current_;
    }
    bool operator!=(const iterator_base& other) const {
      return !(*this == other);
    }
  };  // end class iterator_base

 public:
  using iterator = iterator_base<false>;
  using const_iterator = iterator_base<true>;

  // Default constructor
  VectorMap() = default;

  // Initializer list constructor. Sets `map[0] = values[0]`, `map[1] =
  // values[1]`, etc.
  VectorMap(std::initializer_list<T> values) : data_(values) {}

  // Initializer list constructor for key-value pairs. Sets `map[key] = value`
  // for each pair in the list.
  VectorMap(std::initializer_list<std::pair<size_t, size_t>> entries) {
    for (const auto& [key, value] : entries) {
      (*this)[key] = value;
    }
  }

  // Vector constructor
  explicit VectorMap(std::vector<T> values) : data_(std::move(values)) {}

  // Expose begin/end
  iterator begin() { return iterator(data_.begin(), data_.end(), 0); }
  const_iterator begin() const {
    return const_iterator(data_.begin(), data_.end(), 0);
  }
  iterator end() { return iterator(data_.end(), data_.end(), data_.size()); }
  const_iterator end() const {
    return const_iterator(data_.end(), data_.end(), data_.size());
  }

  // Access methods
  mapped_type& at(size_t index) {
    if (index >= data_.size()) {
      data_.resize(index + 1);
    }
    return data_[index];
  }

  const mapped_type& at(size_t index) const {
    if (index >= data_.size()) {
      static const T zero{};
      return zero;
    }
    return data_[index];
  }

  mapped_type& operator[](std::size_t i) { return at(i); }
  const mapped_type& operator[](std::size_t i) const { return at(i); }

  // Capacity
  size_t capacity() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  // Modifiers
  void clear() { data_.clear(); }

  // Erase a specific key
  void erase(size_t index) {
    if (index < data_.size()) {
      data_[index] = T{};
    }
  }

  // Count non-zero elements
  size_t count() const {
    size_t count = 0;
    for (const auto& item : data_) {
      if (item != T{}) {
        ++count;
      }
    }
    return count;
  }

  // Reserve capacity so that the map can store an element at maxKey without
  // resizing
  void reserve(size_t maxKey) {
    if (maxKey >= data_.size()) {
      data_.resize(maxKey + 1);
    }
  }

  // Comparison operators
  bool operator==(const VectorMap& other) const {
    // Make sure that the common elements are equal, and that the rest are
    // zero
    size_t min_size = std::min(this->capacity(), other.capacity());
    if (!std::equal(this->data_.begin(), this->data_.begin() + min_size,
                    other.data_.begin())) {
      return false;
    }

    // Check that the rest of the elements are zero
    if (this->capacity() != other.capacity()) {
      auto& largerData =
          this->capacity() > other.capacity() ? this->data_ : other.data_;
      for (size_t i = min_size; i < largerData.size(); ++i) {
        if (largerData[i] != T{}) {
          return false;
        }
      }
    }

    return true;
  }

  size_t maxKey() const {
    for (ssize_t i = data_.size() - 1; i >= 0; --i) {
      if (data_[i] != T{}) {
        return i;
      }
    }
    throw std::runtime_error("No non-zero elements");
  }

  size_t minKey() const {
    for (size_t i = 0; i < data_.size(); ++i) {
      if (data_[i] != T{}) {
        return i;
      }
    }
    throw std::runtime_error("No non-zero elements");
  }
};

}  // namespace graphene