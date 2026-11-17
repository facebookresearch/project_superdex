/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/reflection_error.h>

#include <stdexcept>
#include <string>

using namespace mochi;

namespace {

class Tank {
 public:
  int capacity = 10;

  int const& GetLevel() const {
    return _level;
  }
  void SetLevel(int level) {
    _level = level;
  }
  void SetCheckedLevel(int const& level, Error& error) {
    MOCHI_ERROR_IF(level > capacity, error, "Level exceeds capacity.");
    MOCHI_ERROR_RETURN(error);
    _level = level;
  }
  int GetHeadroom(Error& error) const noexcept {
    MOCHI_ERROR_IF(_level > capacity, error, "Tank is overfull.");
    MOCHI_ERROR_RETURN(error, 0);
    return capacity - _level;
  }
  // A static member function has a free function pointer type.
  static int GetFillPercent(Tank const& tank, Error& error) {
    MOCHI_ERROR_IF(tank._level > tank.capacity, error, "Tank is overfull.");
    MOCHI_ERROR_RETURN(error, 0);
    return tank._level * 100 / tank.capacity;
  }
  int Fill(int amount, Error& error) noexcept {
    SetCheckedLevel(_level + amount, error);
    return _level;
  }
  int GetCapacity() const& {
    return capacity;
  }
  void SetCapacity(int newCapacity, Error& error) & {
    MOCHI_ERROR_IF(newCapacity < _level, error, "Capacity is below the level.");
    MOCHI_ERROR_RETURN(error);
    capacity = newCapacity;
  }

  MOCHI_STRUCT_BEGIN(Tank)
  MOCHI_FIELD(capacity)
  MOCHI_PROPERTY("level", &Tank::GetLevel, &Tank::SetLevel)
  MOCHI_PROPERTY_WITH_ERROR("checkedLevel", &Tank::GetLevel, &Tank::SetCheckedLevel)
  MOCHI_PROPERTY_WITH_ERROR("headroom", &Tank::GetHeadroom)
  MOCHI_PROPERTY_WITH_ERROR("fillPercent", &Tank::GetFillPercent)
  MOCHI_PROPERTY_WITH_ERROR("checkedCapacity", &Tank::GetCapacity, &Tank::SetCapacity)
  MOCHI_METHOD_WITH_ERROR("fill(amount)", &Tank::Fill)
  MOCHI_METHOD("getLevel()", &Tank::GetLevel)
  MOCHI_STRUCT_END()

 private:
  int _level = 0;
};

} // namespace

static SReflect::FieldTypeInfo const& TankField(char const* name) {
  SReflect::FieldTypeInfo const* field = Tank::GetTypeInfo().FindField(name);
  if (field == nullptr) {
    throw std::invalid_argument(std::string("Tank has no field named ") + name);
  }
  return *field;
}

static int ReadInt(Tank const& tank, char const* name) {
  int value = 0;
  TankField(name).GetValue(&tank, &value);
  return value;
}

static void WriteInt(Tank& tank, char const* name, int value) {
  EXPECT_TRUE(TankField(name).SetValue(&tank, &value));
}

// Returns what() of the std::runtime_error that fn throws, or an empty string if it does not throw.
template <class Fn>
static std::string RuntimeErrorMessage(Fn const& fn) {
  try {
    fn();
  } catch (std::runtime_error const& e) {
    return e.what();
  }
  return {};
}

TEST(ReflectionMacros, RegistersLikeSrMacros) {
  EXPECT_FALSE(TankField("level").IsReadOnly());
  EXPECT_FALSE(TankField("checkedLevel").IsReadOnly());
  EXPECT_FALSE(TankField("checkedCapacity").IsReadOnly());
  EXPECT_TRUE(TankField("headroom").IsReadOnly());
  EXPECT_TRUE(TankField("fillPercent").IsReadOnly());

  SReflect::MethodInfo const* getLevel = Tank::GetTypeInfo().FindMethod("getLevel");
  ASSERT_NE(nullptr, getLevel);
  EXPECT_TRUE(getLevel->_isConst);
  EXPECT_TRUE(getLevel->_args.empty());
}

TEST(ReflectionMacros, SetterWithError) {
  Tank tank;
  WriteInt(tank, "checkedLevel", 4);
  EXPECT_EQ(4, tank.GetLevel());
  EXPECT_EQ(4, ReadInt(tank, "checkedLevel"));

  EXPECT_EQ("Level exceeds capacity.", RuntimeErrorMessage([&] {
              WriteInt(tank, "checkedLevel", tank.capacity + 1);
            }));
  EXPECT_EQ(4, tank.GetLevel());

  // Ref-qualified accessors.
  WriteInt(tank, "checkedCapacity", 8);
  EXPECT_EQ(8, ReadInt(tank, "checkedCapacity"));
  EXPECT_EQ("Capacity is below the level.", RuntimeErrorMessage([&] {
              WriteInt(tank, "checkedCapacity", 3);
            }));
  EXPECT_EQ(8, tank.capacity);
}

TEST(ReflectionMacros, GettersWithError) {
  Tank tank;
  tank.SetLevel(4);
  EXPECT_EQ(6, ReadInt(tank, "headroom"));
  EXPECT_EQ(40, ReadInt(tank, "fillPercent"));

  tank.SetLevel(tank.capacity + 1);
  EXPECT_EQ("Tank is overfull.", RuntimeErrorMessage([&] { ReadInt(tank, "headroom"); }));
  EXPECT_EQ("Tank is overfull.", RuntimeErrorMessage([&] { ReadInt(tank, "fillPercent"); }));
}

TEST(ReflectionMacros, MethodWithError) {
  SReflect::MethodInfo const* fill = Tank::GetTypeInfo().FindMethod("fill");
  ASSERT_NE(nullptr, fill);
  ASSERT_EQ(1u, fill->_args.size());
  EXPECT_EQ("amount", fill->_args[0]._name);
  EXPECT_FALSE(fill->_isConst);

  Tank tank;
  int amount = 4;
  int result = 0;
  void* args[] = {&amount};
  EXPECT_TRUE(fill->Call(&tank, &result, args));
  EXPECT_EQ(4, result);
  EXPECT_EQ(4, tank.GetLevel());

  amount = tank.capacity;
  EXPECT_EQ(
      "Level exceeds capacity.", RuntimeErrorMessage([&] { fill->Call(&tank, &result, args); }));
  EXPECT_EQ(4, tank.GetLevel());
}
