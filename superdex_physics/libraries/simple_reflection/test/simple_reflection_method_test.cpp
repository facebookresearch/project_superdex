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

#include <simple_reflection/simple_reflection.h>

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SReflectMethodTest {
namespace {

// A supported type with no default constructor
struct NoDefault {
  explicit NoDefault(int32_t v) : value(v) {}
  int32_t value;

  SR_BeginStruct(SReflectMethodTest::NoDefault);
  SR_Field(value);
  SR_EndStruct()
};

struct Base {
  Base() = default;
  Base(Base const&) = default;
  Base& operator=(Base const&) = default;
  virtual ~Base() = default;
  [[nodiscard]] virtual std::string Describe() const {
    return "base";
  }
  [[nodiscard]] int32_t Twice(int32_t n) const noexcept {
    return 2 * n;
  }

  SR_BeginStruct(SReflectMethodTest::Base);
  // A virtual member function, inherited by Machine, which overrides it
  SR_Method("describe()", &Base::Describe);
  SR_EndStruct()
};

// Covers most kinds of method. The comment above each one says which case it covers.
struct Machine : Base {
  int32_t total = 0;

  [[nodiscard]] std::string Describe() const override {
    return "machine";
  }
  int32_t Add(int32_t n) {
    if (n < 0) {
      throw std::invalid_argument("negative");
    }
    return total += n;
  }
  void Absorb(NoDefault const& v) & noexcept {
    total += v.value;
  }
  static std::string Label(Machine const& m, std::string prefix) {
    prefix += std::to_string(m.total);
    return prefix;
  }
  [[nodiscard]] int32_t const& GetTotal() const& {
    return total;
  }

  SR_BeginStruct(SReflectMethodTest::Machine);
  SR_BaseClass(SReflectMethodTest::Base);
  SR_Field(total);
  // A non-const member function that throws, with an attribute
  SR_Method("add(n)", &Machine::Add);
  SRA_Description("Adds n to the total");
  // A const noexcept member function of the base class
  SR_Method("twice(n)", &Base::Twice);
  // An lvalue-qualified noexcept member function with an argument type that isn't default
  // constructible
  SR_Method("absorb(value)", &Machine::Absorb);
  // A return type that isn't default constructible
  SR_Method("snapshot()", [](Machine const& m) { return NoDefault{m.total}; });
  // A const noexcept generic lambda with an out-parameter
  SR_Method(
      "divmod(divisor, remainder)",
      [](auto const& m, int32_t divisor, int32_t& remainder) noexcept {
        remainder = m.total % divisor;
        return m.total / divisor;
      });
  // A non-const generic lambda that returns void
  SR_Method("scale(factor)", [](auto& m, double factor) {
    m.total = static_cast<int32_t>(m.total * factor);
  });
  // A noexcept lambda whose argument and return types are the class being defined
  SR_Method("merge(other)", [](Machine const& m, Machine const& other) noexcept {
    Machine merged;
    merged.total = m.total + other.total;
    return merged;
  });
  // A function pointer (a static member behaves like a free function) with a by-value argument
  // that it modifies
  SR_Method("label(prefix)", &Machine::Label);
  // A const&-qualified member function that returns a const reference
  SR_Method("getTotal()", &Machine::GetTotal);
  SR_EndStruct()
};

} // namespace

// SR_Method checks its name at compile time, and splits it at registration.
static_assert(SReflect::detail::CountMethodArgNames(" f\t( a ,b\t) ") == 2);
static_assert(SReflect::detail::MethodNamePart(" f\t( a ,b\t) ", 0) == "f");
static_assert(SReflect::detail::MethodNamePart(" f\t( a ,b\t) ", 2) == "b");
static_assert(SReflect::detail::CountMethodArgNames("f( )") == 0);
static_assert(SReflect::detail::CountMethodArgNames("f") < 0);
static_assert(SReflect::detail::CountMethodArgNames("f(a,)") < 0);
static_assert(SReflect::detail::CountMethodArgNames("f(a, a)") < 0);
static_assert(SReflect::detail::CountMethodArgNames("f()x") < 0);

static SReflect::MethodInfo const& Method(char const* name) {
  SReflect::MethodInfo const* method = Machine::GetTypeInfo().FindMethod(name);
  if (method == nullptr) {
    throw std::runtime_error(std::string{"missing method "} + name);
  }
  return *method;
}

// Format a method like "divmod(int32 divisor, int32& remainder) const -> int32".
static std::string Signature(SReflect::MethodInfo const& method) {
  std::string result = method._name + "(";
  for (SReflect::MethodArgInfo const& arg : method._args) {
    result += &arg == method._args.data() ? "" : ", ";
    result += std::string{arg.GetTypeInfo()._name} + (arg._isNonConstRef ? "& " : " ") + arg._name;
  }
  result += method._isConst ? ") const" : ")";
  if (SReflect::TypeInfo const* returnType = method.GetReturnTypeInfo()) {
    result += std::string{" -> "} + returnType->_name;
  }
  return result;
}

// Call a method through MethodInfo::Call with the addresses of the given arguments.
template <class Object, class... Args>
static bool CallMethod(Object& obj, char const* name, void* returnValue, Args&... args) {
  std::array<void*, sizeof...(Args) + 1> argPtrs = {&args..., nullptr};
  return Method(name).Call(
      &obj, returnValue, SReflect::Span<void* const>(argPtrs.data(), sizeof...(Args)));
}

TEST(SReflectMethod, Registration) {
  std::vector<std::string> signatures;
  for (SReflect::MethodInfo const* method : Machine::GetTypeInfo()._methods) {
    signatures.push_back(Signature(*method));
  }
  std::vector<std::string> const expected = {
      "describe() const -> string",
      "add(int32 n) -> int32",
      "twice(int32 n) const -> int32",
      "absorb(NoDefault value)",
      "snapshot() const -> NoDefault",
      "divmod(int32 divisor, int32& remainder) const -> int32",
      "scale(double factor)",
      "merge(Machine other) const -> Machine",
      "label(string prefix) const -> string",
      "getTotal() const -> int32",
  };
  EXPECT_EQ(expected, signatures);

  // The attribute after add goes on add, not on the field before it
  EXPECT_TRUE(Method("add").HasAttribute<SReflect::Attribute_Description>());
  EXPECT_FALSE(
      Machine::GetTypeInfo().FindField("total")->HasAttribute<SReflect::Attribute_Description>());

  // Fields and methods are looked up separately
  EXPECT_EQ(nullptr, Machine::GetTypeInfo().FindMethod("total"));
  EXPECT_EQ(nullptr, Machine::GetTypeInfo().FindField("add"));
}

TEST(SReflectMethod, Call) {
  Machine machine;
  int32_t n = 5;
  int32_t result = 0;
  EXPECT_TRUE(CallMethod(machine, "add", &result, n));
  EXPECT_EQ(5, result);
  EXPECT_TRUE(CallMethod(machine, "add", nullptr, n)); // Discards the return value
  double factor = 1.5;
  EXPECT_TRUE(CallMethod(machine, "scale", nullptr, factor));
  EXPECT_EQ(15, machine.total);
  int32_t negative = -1;
  EXPECT_THROW(CallMethod(machine, "add", nullptr, negative), std::invalid_argument);
  EXPECT_EQ(15, machine.total);

  // Const methods work on a const object
  Machine const& constMachine = machine;
  int32_t divisor = 4;
  int32_t remainder = 0;
  EXPECT_TRUE(CallMethod(constMachine, "divmod", &result, divisor, remainder));
  EXPECT_EQ(std::make_pair(3, 3), std::make_pair(result, remainder));
  EXPECT_TRUE(CallMethod(constMachine, "twice", &result, n));
  EXPECT_EQ(10, result);
  EXPECT_TRUE(CallMethod(constMachine, "getTotal", &result));
  EXPECT_EQ(15, result);
  std::string text;
  EXPECT_TRUE(CallMethod(constMachine, "describe", &text));
  EXPECT_EQ("machine", text);
  Machine other;
  other.total = 1;
  Machine merged;
  EXPECT_TRUE(CallMethod(constMachine, "merge", &merged, other));
  EXPECT_EQ(16, merged.total);

  // Label modifies its copy of prefix, not the caller's
  std::string prefix = "total=";
  EXPECT_TRUE(CallMethod(constMachine, "label", &text, prefix));
  EXPECT_EQ("total=15", text);
  EXPECT_EQ("total=", prefix);

  NoDefault part{2};
  EXPECT_TRUE(CallMethod(machine, "absorb", nullptr, part));
  NoDefault snapshot{0};
  EXPECT_TRUE(CallMethod(constMachine, "snapshot", &snapshot));
  EXPECT_EQ(17, snapshot.value);
}

TEST(SReflectMethod, CallReturnsFalseWithoutCalling) {
  Machine machine;
  int32_t n = 5;
  std::array<void*, 2> tooManyArgs = {&n, &n};
  std::array<void*, 1> nullArg = {nullptr};
  EXPECT_FALSE(Method("add").Call(&machine, nullptr, tooManyArgs));
  EXPECT_FALSE(Method("add").Call(&machine, nullptr, nullArg));
  EXPECT_FALSE(CallMethod(std::as_const(machine), "add", nullptr, n));
  double factor = 2.0;
  EXPECT_FALSE(CallMethod(std::as_const(machine), "scale", nullptr, factor));
  EXPECT_EQ(0, machine.total);
}

TEST(SReflectMethod, MemberNamesMustBeUnique) {
  // Calls what SR_Field, SR_Method and SR_BaseClass do, on StructTypeInfos that the test owns. The
  // test deletes the members it adds successfully, because a StructTypeInfo doesn't own them.
  auto const addField = [](SReflect::StructTypeInfo& info, char const* name) {
    SReflect::detail::AddField(info, name, 0, SReflect::GetTypeInfo<int32_t>());
  };
  auto const addMethod = [](SReflect::StructTypeInfo& info, char const* nameWithArgs) {
    SReflect::detail::AddMethod<Machine, 0>(info, nameWithArgs, [](Machine const& /*unused*/) {});
  };

  std::unique_ptr<SReflect::StructTypeInfo> const info{
      SReflect::MakeStructTypeInfo<Machine>("SReflectMethodTest::MemberNameUniqueness")};
  SReflect::detail::AddBaseClass(*info, Base::GetTypeInfo(), 0); // Inherits "describe"
  EXPECT_THROW(addField(*info, "describe"), std::runtime_error);
  EXPECT_THROW(addMethod(*info, "describe()"), std::runtime_error);
  addField(*info, "total");
  std::unique_ptr<SReflect::FieldTypeInfo> const total{info->_fields.back()};
  EXPECT_THROW(addMethod(*info, "total()"), std::runtime_error);
  addMethod(*info, "reset()");
  std::unique_ptr<SReflect::MethodInfo> const reset{info->_methods.back()};
  EXPECT_THROW(addField(*info, "reset"), std::runtime_error);

  // Machine has a field named "total" and a method named "add"
  std::unique_ptr<SReflect::StructTypeInfo> const derived{
      SReflect::MakeStructTypeInfo<NoDefault>("SReflectMethodTest::BaseMemberNameUniqueness")};
  addField(*derived, "add");
  std::unique_ptr<SReflect::FieldTypeInfo> const add{derived->_fields.back()};
  EXPECT_THROW(
      SReflect::detail::AddBaseClass(*derived, Machine::GetTypeInfo(), 0), std::runtime_error);
  addMethod(*derived, "total()");
  std::unique_ptr<SReflect::MethodInfo> const derivedTotal{derived->_methods.back()};
  EXPECT_THROW(
      SReflect::detail::AddBaseClass(*derived, Machine::GetTypeInfo(), 0), std::runtime_error);
}

} // namespace SReflectMethodTest
