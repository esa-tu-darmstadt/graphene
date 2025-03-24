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

#include <gtest/gtest.h>

#include <iterator>

#include "libgraphene/common/VectorMap.hpp"

using namespace graphene;

class VectorMapTest : public ::testing::Test {
 protected:
  VectorMap<int> int_map;
  VectorMap<std::string> string_map;

  void SetUp() override {
    // Initialize test fixtures
    int_map[1] = 10;
    int_map[5] = 50;
    int_map[10] = 100;

    string_map[2] = "two";
    string_map[4] = "four";
  }
};

// Basic Operations Tests
TEST_F(VectorMapTest, DefaultConstructor) {
  VectorMap<int> map;
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(map.capacity(), 0);
}

// Construct from initializer lists
TEST_F(VectorMapTest, InitializerListConstructors) {
  VectorMap<int> map1({1, 2, 3});
  EXPECT_EQ(map1[0], 1);
  EXPECT_EQ(map1[1], 2);
  EXPECT_EQ(map1[2], 3);
  EXPECT_EQ(map1.count(), 3);

  VectorMap<int> map2({{1, 10}, {2, 20}});
  EXPECT_EQ(map2[0], 0);
  EXPECT_EQ(map2[1], 10);
  EXPECT_EQ(map2[2], 20);
  EXPECT_EQ(map2.count(), 2);
}

TEST_F(VectorMapTest, AccessOperator) {
  EXPECT_EQ(int_map[1], 10);
  EXPECT_EQ(int_map[5], 50);
  EXPECT_EQ(int_map[10], 100);

  // Access non-existent elements
  EXPECT_EQ(int_map[2], 0);  // Default-constructed value
  EXPECT_EQ(int_map[20], 0);
}

TEST_F(VectorMapTest, ConstAccessOperator) {
  const VectorMap<int>& const_map = int_map;
  EXPECT_EQ(const_map[1], 10);
  EXPECT_EQ(const_map[5], 50);
  EXPECT_EQ(const_map[20], 0);  // Out of bounds returns default value
}

TEST_F(VectorMapTest, AtMethod) {
  EXPECT_EQ(int_map.at(1), 10);
  EXPECT_EQ(int_map.at(5), 50);

  // Access non-existent element should resize
  EXPECT_EQ(int_map.at(15), 0);
  EXPECT_GE(int_map.capacity(), 16);
}

TEST_F(VectorMapTest, ConstAtMethod) {
  const VectorMap<int>& const_map = int_map;
  EXPECT_EQ(const_map.at(1), 10);
  EXPECT_EQ(const_map.at(5), 50);
  EXPECT_EQ(const_map.at(20), 0);
}

// Iterator Tests
TEST_F(VectorMapTest, IteratorBasics) {
  std::vector<std::pair<size_t, int>> expected = {{1, 10}, {5, 50}, {10, 100}};

  size_t index = 0;
  for (const auto& [key, value] : int_map) {
    ASSERT_LT(index, expected.size());
    EXPECT_EQ(key, expected[index].first);
    EXPECT_EQ(value, expected[index].second);
    ++index;
  }
  EXPECT_EQ(index, expected.size());
}

// Const Iterator Basic Test
TEST_F(VectorMapTest, ConstIteratorBasics) {
  std::vector<std::pair<size_t, int>> expected = {{1, 10}, {5, 50}, {10, 100}};

  const VectorMap<int>& const_map = int_map;

  size_t index = 0;
  for (const auto [key, value] : const_map) {
    ASSERT_LT(index, expected.size());
    EXPECT_EQ(key, expected[index].first);
    EXPECT_EQ(value, expected[index].second);
    ++index;
  }
  EXPECT_EQ(index, expected.size());
}

TEST_F(VectorMapTest, IteratorModification) {
  // Modify values through iterator
  for (auto [key, value] : int_map) {
    value *= 2;
  }

  EXPECT_EQ(int_map[1], 20);
  EXPECT_EQ(int_map[5], 100);
  EXPECT_EQ(int_map[10], 200);
}

TEST_F(VectorMapTest, EqualityOperator) {
  VectorMap<int> copy_map = int_map;
  EXPECT_EQ(int_map, copy_map);

  copy_map[1] = 0;
  EXPECT_NE(int_map, copy_map);
}

TEST_F(VectorMapTest, IteratorSkipsZeros) {
  int_map[2] = 0;  // Add zero value
  int_map[3] = 0;  // Add another zero value

  std::vector<size_t> visited_keys;
  for (const auto& [key, value] : int_map) {
    visited_keys.push_back(key);
    EXPECT_NE(value, 0);  // Iterator should skip zeros
  }

  EXPECT_EQ(visited_keys.size(), 3);  // Only non-zero elements
  EXPECT_EQ(visited_keys[0], 1);
  EXPECT_EQ(visited_keys[1], 5);
  EXPECT_EQ(visited_keys[2], 10);
}

// Modification Tests
TEST_F(VectorMapTest, EraseOperation) {
  int_map.erase(5);
  EXPECT_EQ(int_map[5], 0);

  // Verify iterator skips erased element
  std::vector<size_t> keys;
  for (const auto& [key, value] : int_map) {
    keys.push_back(key);
  }
  EXPECT_EQ(keys.size(), 2);
  EXPECT_EQ(keys[0], 1);
  EXPECT_EQ(keys[1], 10);
}

TEST_F(VectorMapTest, ClearOperation) {
  int_map.clear();
  EXPECT_TRUE(int_map.empty());
  EXPECT_EQ(int_map.capacity(), 0);

  // Verify iterator behavior on empty container
  EXPECT_EQ(int_map.begin(), int_map.end());
}

TEST_F(VectorMapTest, Count) {
  EXPECT_EQ(int_map.count(), 3);

  int_map[15] = 0;                // Add zero value
  EXPECT_EQ(int_map.count(), 3);  // Count should not change

  int_map[20] = 200;  // Add non-zero value
  EXPECT_EQ(int_map.count(), 4);
}

// Edge Cases
TEST_F(VectorMapTest, ZeroValueHandling) {
  int_map[7] = 70;
  int_map[7] = 0;  // Set to zero

  // Verify zero is treated as non-existent
  bool found_seven = false;
  for (const auto& [key, value] : int_map) {
    if (key == 7) found_seven = true;
  }
  EXPECT_FALSE(found_seven);
}

TEST_F(VectorMapTest, LargeIndices) {
  size_t large_index = 1000000;
  int_map[large_index] = 42;

  EXPECT_EQ(int_map[large_index], 42);
  EXPECT_GE(int_map.capacity(), large_index + 1);
}

// Type Compatibility Tests
TEST_F(VectorMapTest, ComplexTypes) {
  struct ComplexType {
    int x;
    bool operator==(const ComplexType& other) const { return x == other.x; }
    bool operator!=(const ComplexType& other) const {
      return !(*this == other);
    }
  };

  VectorMap<ComplexType> complex_map;
  complex_map[1] = ComplexType{42};
  complex_map[2] = ComplexType{0};  // Should be treated as non-existent

  size_t count = 0;
  for (const auto& [key, value] : complex_map) {
    EXPECT_EQ(value.x, 42);
    ++count;
  }
  EXPECT_EQ(count, 1);
}

// Test maxKey and minKey methods
TEST_F(VectorMapTest, MaxKey) {
  EXPECT_EQ(int_map.maxKey(), 10);
  int_map[20] = 200;
  EXPECT_EQ(int_map.maxKey(), 20);
}

TEST_F(VectorMapTest, MinKey) {
  EXPECT_EQ(int_map.minKey(), 1);
  int_map[0] = 200;
  EXPECT_EQ(int_map.minKey(), 0);
}

TEST_F(VectorMapTest, Copy) {
  // Copy constructor
  VectorMap<int> copy_map = int_map;
  EXPECT_EQ(int_map, copy_map);

  // Assignment operator
  VectorMap<int> empty_map;
  EXPECT_TRUE(empty_map.empty());
  empty_map = int_map;
  EXPECT_EQ(int_map, empty_map);
}

static_assert(std::forward_iterator<std::map<int, int>::iterator>);

// static_assert(std::forward_iterator<VectorMap<int>::iterator>);
// static_assert(std::forward_iterator<VectorMap<int>::const_iterator>);