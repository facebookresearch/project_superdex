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
#include <mochi_core/utils/reflection.h>
#include <mochi_physics/src/mochi_attributes.h>
#include <mochi_physics/src/mochi_capture.h>
#include <mochi_physics/src/mochi_common_components.h> // For CActorInfo (used for JSON only)
#include <mochi_physics/src/mochi_ecs_registry.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <type_traits>

using namespace mochi;
using namespace mochi::capture;

namespace mochi::attribute {
namespace {

//----------------------------------------------------------------------------------------
// Test Attributes (used to test RestorePartialState)
//----------------------------------------------------------------------------------------

struct TestAttributeA : Attribute {
  MOCHI_STRUCT_WITH_BASE(mochi::attribute::TestAttributeA, mochi::Attribute);
};

struct TestAttributeB : Attribute {
  MOCHI_STRUCT_WITH_BASE(mochi::attribute::TestAttributeB, mochi::Attribute);
};

} // namespace
} // namespace mochi::attribute

namespace mochi_capture_test {
namespace {

//----------------------------------------------------------------------------------------
// ECS Test Components
//----------------------------------------------------------------------------------------

// ECS component with trivial data.
struct CComponentA : NoCopy {
  float fval{};
  int32_t ival{};

  MOCHI_STRUCT_BEGIN(mochi_capture_test::CComponentA);
  MOCHI_ATTRIBUTE(CaptureState)
  MOCHI_ATTRIBUTE(CaptureStateCtx)
  MOCHI_ATTRIBUTE(TestAttributeA)
  MOCHI_FIELD(fval);
  MOCHI_FIELD(ival);
  MOCHI_STRUCT_END();
};
static_assert(sizeof(CComponentA) == 8);
static_assert(std::is_trivially_copyable_v<CComponentA>);

// ECS component with non-trivial data.
struct CComponentB : NoCopy {
  int32_t ival{};
  DynamicArray<std::string> strings;

  MOCHI_STRUCT_BEGIN(mochi_capture_test::CComponentB);
  MOCHI_ATTRIBUTE(CaptureState)
  MOCHI_ATTRIBUTE(CaptureStateCtx)
  MOCHI_ATTRIBUTE(TestAttributeB)
  MOCHI_FIELD(ival)
  MOCHI_FIELD(strings)
  MOCHI_STRUCT_END()
};
static_assert(!std::is_trivially_copyable_v<CComponentB>);

// ECS component with both TestAttriuteA and TestAttributeB
struct CComponentAB : NoCopy {
  int32_t ival{};

  MOCHI_STRUCT_BEGIN(mochi_capture_test::CComponentAB)
  MOCHI_ATTRIBUTE(CaptureState)
  MOCHI_ATTRIBUTE(CaptureStateCtx)
  MOCHI_ATTRIBUTE(TestAttributeA)
  MOCHI_ATTRIBUTE(TestAttributeB)
  MOCHI_FIELD(ival)
  MOCHI_STRUCT_END()
};

// ECS component with some padding bytes, not owned by any reflection field.
struct CComponentWithPadding {
  double dval{}; // 8 byte alignment
  int32_t ival{}; // 4 byte alignment

  // If you deleted this variable, it would not change the size of the struct.
  // It must round up 16 bytes because of the 8-byte alignment requirement.
  int32_t padding{};

  MOCHI_STRUCT_BEGIN(mochi_capture_test::CComponentWithPadding);
  MOCHI_ATTRIBUTE(CaptureState) MOCHI_ATTRIBUTE(CaptureStateCtx);
  MOCHI_FIELD(dval);
  MOCHI_FIELD(ival);
  // No MOCHI_FIELD(padding)
  MOCHI_STRUCT_END();
};
static_assert(alignof(CComponentWithPadding) == 8);
static_assert(sizeof(CComponentWithPadding) == 16);
static_assert(std::is_trivially_copyable_v<CComponentWithPadding>);

// An ECS component for which state is NOT captured
struct CNonCapturedComponent {
  int value;

  MOCHI_STRUCT_BEGIN(mochi_capture_test::CNonCapturedComponent);
  // No MOCHI_ATTRIBUTE(CaptureState)
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

//----------------------------------------------------------------------------------------
// Test Fixture
//----------------------------------------------------------------------------------------

class MochiCapture : public testing::Test {
 public:
  void SetUp() override {
    ecs::InitializeComponentRegistryOnce(reg);
    ecs::RegisterComponent<CComponentA>(reg);
    ecs::RegisterComponent<CComponentB>(reg);
    ecs::RegisterComponent<CComponentAB>(reg);
    ecs::RegisterComponent<CComponentWithPadding>(reg);
    ecs::RegisterComponent<CNonCapturedComponent>(reg);
    ecs::RegisterComponent<CActorInfo>(reg);
    capture::InitializeOnce(reg);
    ecs::FinalizeComponentRegistration(reg);
  }

  entt::registry reg;
};

} // namespace
} // namespace mochi_capture_test
using namespace mochi_capture_test;

//----------------------------------------------------------------------------------------
// Test Cases
//----------------------------------------------------------------------------------------

TEST_F(MochiCapture, ComponentTypeInfo) {
  // Verify that our components have the reflection properties we intended
  EXPECT_TRUE(SReflect::GetTypeInfo<CComponentA>().IsMemCopySafe());
  EXPECT_FALSE(SReflect::GetTypeInfo<CComponentB>().IsMemCopySafe());
}

TEST_F(MochiCapture, Empty) {
  // Capture a scene with nothing in it.
  DynamicArray<uint8_t> state0, state1;
  CaptureState(reg, state0, test::ExpectOK{});
  EXPECT_NE(0, state0.size()); // Still has a header and footer
  CaptureState(reg, state1, test::ExpectOK{});
  EXPECT_EQ(state0.size(), state1.size());
  RestoreState(reg, state0, test::ExpectOK{});
  RestoreState(reg, state1, test::ExpectOK{});
  EXPECT_TRUE(IsEqualState(reg, state0, state1));
}

TEST_F(MochiCapture, TrivialCtxComponent) {
  DynamicArray<uint8_t> emptyState, state0, state1, state2;
  CaptureState(reg, emptyState, test::ExpectOK{});

  // Set a global context component and capture
  auto& compA = reg.set<CComponentA>();
  compA.fval = 1.23f;
  compA.ival = 42;
  CaptureState(reg, state0, test::ExpectOK{});
  EXPECT_NE(0, state0.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state0));

  // Change the values and capture again
  compA.fval = 2.34f;
  compA.ival = 7;
  CaptureState(reg, state1, test::ExpectOK{});
  EXPECT_NE(0, state1.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state1));
  EXPECT_FALSE(IsEqualState(reg, state0, state1));

  // Restore state0
  RestoreState(reg, state0, test::ExpectOK{});
  EXPECT_EQ(1.23f, compA.fval);
  EXPECT_EQ(42, compA.ival);

  // Restore state1
  RestoreState(reg, state1, test::ExpectOK{});
  EXPECT_EQ(2.34f, compA.fval);
  EXPECT_EQ(7, compA.ival);

  // Set a global context component that will NOT get captured.
  reg.set<CNonCapturedComponent>();

  // Capture state2 (same as state1 because CNonCapturedComponent doesn't count)
  CaptureState(reg, state2, test::ExpectOK{});
  EXPECT_EQ(state1.size(), state2.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state2));
  EXPECT_FALSE(IsEqualState(reg, state0, state2));
  EXPECT_TRUE(IsEqualState(reg, state1, state2));
}

TEST_F(MochiCapture, NonTrivialCtxComponent) {
  DynamicArray<uint8_t> emptyState, state0, state1, state2;
  CaptureState(reg, emptyState, test::ExpectOK{});

  // Set a global context component and capture
  auto& compB = reg.set<CComponentB>();
  compB.strings = {"one", "two"};
  CaptureState(reg, state0, test::ExpectOK{});
  EXPECT_NE(0, state0.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state0));

  // Change the values and capture again
  compB.strings = {"three", "four"};
  CaptureState(reg, state1, test::ExpectOK{});
  EXPECT_NE(0, state1.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state1));
  EXPECT_FALSE(IsEqualState(reg, state0, state1));

  // Restore state0
  RestoreState(reg, state0, test::ExpectOK{});
  EXPECT_EQ(compB.strings, (DynamicArray<std::string>{"one", "two"}));

  // Restore state1
  RestoreState(reg, state1, test::ExpectOK{});
  EXPECT_EQ(compB.strings, (DynamicArray<std::string>{"three", "four"}));

  // Set a global context component that will NOT get captured.
  reg.set<CNonCapturedComponent>();

  // Capture state2 (same as state1 because CNonCapturedComponent doesn't count)
  CaptureState(reg, state2, test::ExpectOK{});
  EXPECT_EQ(state1.size(), state2.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state2));
  EXPECT_FALSE(IsEqualState(reg, state0, state2));
  EXPECT_TRUE(IsEqualState(reg, state1, state2));
}

TEST_F(MochiCapture, TrivialEntityComponents) {
  DynamicArray<uint8_t> emptyState, state0, state1, state2;
  CaptureState(reg, emptyState, test::ExpectOK{});

  // Create a few entities
  DynamicArray<entt::entity> entities = {reg.create(), reg.create(), reg.create(), reg.create()};

  // Emplace a component on half of them
  auto& compA0 = reg.emplace<CComponentA>(entities[0]);
  compA0.fval = 1.23f;
  compA0.ival = 4;
  auto& compA2 = reg.emplace<CComponentA>(entities[2]);
  compA2.fval = 5.67f;
  compA2.ival = 8;
  CaptureState(reg, state0, test::ExpectOK{});
  EXPECT_NE(0, state0.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state0));

  // Change the values and capture again
  compA0.fval = 1.11;
  compA2.fval = 2.22;
  CaptureState(reg, state1, test::ExpectOK{});
  EXPECT_NE(0, state1.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state1));
  EXPECT_FALSE(IsEqualState(reg, state0, state1));

  // Restore state0
  RestoreState(reg, state0, test::ExpectOK{});
  EXPECT_EQ(1.23f, compA0.fval);
  EXPECT_EQ(4, compA0.ival);
  EXPECT_EQ(5.67f, compA2.fval);
  EXPECT_EQ(8, compA2.ival);

  // Restore state1
  RestoreState(reg, state1, test::ExpectOK{});
  EXPECT_EQ(1.11f, compA0.fval);
  EXPECT_EQ(4, compA0.ival);
  EXPECT_EQ(2.22f, compA2.fval);
  EXPECT_EQ(8, compA2.ival);

  // Emplace a component that will NOT get captured.
  for (auto e : entities) {
    reg.emplace<CNonCapturedComponent>(e);
  }

  // Capture state2 (same as state1 because CNonCapturedComponent doesn't count)
  CaptureState(reg, state2, test::ExpectOK{});
  EXPECT_EQ(state1.size(), state2.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state2));
  EXPECT_FALSE(IsEqualState(reg, state0, state2));
  EXPECT_TRUE(IsEqualState(reg, state1, state2));

  // Destroy the entities
  for (auto e : entities) {
    reg.destroy(e);
  }
}

TEST_F(MochiCapture, NonTrivialEntityComponents) {
  DynamicArray<uint8_t> emptyState, state0, state1, state2;
  CaptureState(reg, emptyState, test::ExpectOK{});

  // Create a few entities
  DynamicArray<entt::entity> entities = {reg.create(), reg.create(), reg.create(), reg.create()};

  // Emplace a component on half of them
  auto& compB0 = reg.emplace<CComponentB>(entities[0]);
  compB0.strings = {"one"};
  auto& compB2 = reg.emplace<CComponentB>(entities[2]);
  compB2.strings = {"two"};
  CaptureState(reg, state0, test::ExpectOK{});
  EXPECT_NE(0, state0.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state0));

  // Change the values and capture again
  compB0.strings = {"one", "again"};
  compB2.strings = {"new thing"};
  CaptureState(reg, state1, test::ExpectOK{});
  EXPECT_NE(0, state1.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state1));
  EXPECT_FALSE(IsEqualState(reg, state0, state1));

  // Restore state0
  RestoreState(reg, state0, test::ExpectOK{});
  EXPECT_EQ((DynamicArray<std::string>{"one"}), compB0.strings);
  EXPECT_EQ((DynamicArray<std::string>{"two"}), compB2.strings);

  // Restore state1
  RestoreState(reg, state1, test::ExpectOK{});
  EXPECT_EQ((DynamicArray<std::string>{"one", "again"}), compB0.strings);
  EXPECT_EQ((DynamicArray<std::string>{"new thing"}), compB2.strings);

  // Emplace a component that will NOT get captured.
  for (auto e : entities) {
    reg.emplace<CNonCapturedComponent>(e);
  }

  // Capture state2 (same as state1 because CNonCapturedComponent doesn't count)
  CaptureState(reg, state2, test::ExpectOK{});
  EXPECT_EQ(state1.size(), state2.size());
  EXPECT_FALSE(IsEqualState(reg, emptyState, state2));
  EXPECT_FALSE(IsEqualState(reg, state0, state2));
  EXPECT_TRUE(IsEqualState(reg, state1, state2));

  // Destroy the entities
  for (auto e : entities) {
    reg.destroy(e);
  }

  // Back to the empty state
  DynamicArray<uint8_t> emptyAgain;
  CaptureState(reg, emptyAgain, test::ExpectOK{});
  EXPECT_EQ(emptyState.size(), emptyAgain.size());
  EXPECT_TRUE(IsEqualState(reg, emptyState, emptyAgain));
}

TEST_F(MochiCapture, ManyEntities) {
  // Create a large number of entities with the components.
  // The component data will be split across multiple pages.
  constexpr int kCount = ENTT_PACKED_PAGE * 5 / 2; // 2.5 pages
  DynamicArray<entt::entity> entities(kCount);
  for (int i = 0; i < kCount; ++i) {
    entities[i] = reg.create();
    reg.emplace<CComponentA>(entities[i]).ival = i;
    reg.emplace<CComponentB>(entities[i]).strings.push_back(Format("%6d", i));
  }
  DynamicArray<uint8_t> state0;
  CaptureState(reg, state0, test::ExpectOK{});

  // Modify the values and capture again.
  // NOTE: We choose to keep the string lengths the same, which keeps the capture data size the
  // same. This, forces IsEqualState to compare the strings, rather than exiting early because of
  // the size mismatch.
  for (int i = 0; i < kCount; ++i) {
    reg.get<CComponentA>(entities[i]).ival = i + 1;
    reg.get<CComponentB>(entities[i]).strings[0] = Format("%6d", i + 1);
  }
  DynamicArray<uint8_t> state1;
  CaptureState(reg, state1, test::ExpectOK{});
  EXPECT_EQ(state0.size(), state1.size());
  EXPECT_FALSE(IsEqualState(reg, state0, state1));

  // Restore state0
  RestoreState(reg, state0, test::ExpectOK{});
  for (int i = 0; i < kCount; ++i) {
    EXPECT_EQ(i, reg.get<CComponentA>(entities[i]).ival);
    EXPECT_EQ(Format("%6d", i), reg.get<CComponentB>(entities[i]).strings[0]);
  }

  // Restore state1
  RestoreState(reg, state1, test::ExpectOK{});
  for (int i = 0; i < kCount; ++i) {
    EXPECT_EQ(i + 1, reg.get<CComponentA>(entities[i]).ival);
    EXPECT_EQ(Format("%6d", i + 1), reg.get<CComponentB>(entities[i]).strings[0]);
  }

  // Re-capture the same state
  DynamicArray<uint8_t> state2;
  CaptureState(reg, state2, test::ExpectOK{});
  EXPECT_EQ(state1.size(), state2.size());
  EXPECT_FALSE(IsEqualState(reg, state0, state2));
  EXPECT_TRUE(IsEqualState(reg, state1, state2));
}

TEST_F(MochiCapture, Padding) {
  // CComponentWithPadding contains padding bytes that are not part of any serialized field.
  // The whole thing can be quickly captured via memcpy, including the padding values, but those
  // values should not affect the results of IsEqualState.

  DynamicArray<entt::entity> entities{reg.create(), reg.create(), reg.create()};
  DynamicArray<CComponentWithPadding*> components(entities.size());
  for (size_t i = 0; i < entities.size(); ++i) {
    components[i] = &reg.emplace<CComponentWithPadding>(entities[i]);
    components[i]->dval = static_cast<double>(i);
    components[i]->ival = i;
    components[i]->padding = 111;
  }

  auto& ctx = reg.set<CComponentWithPadding>();
  ctx.dval = 1.23;
  ctx.ival = 42;
  ctx.padding = 222;

  DynamicArray<uint8_t> state0;
  CaptureState(reg, state0, test::ExpectOK{});

  // If we modify those padding values, the state buffers will contain different bytes, but we will
  // still consider them to be equal.
  for (auto* c : components) {
    c->padding = 333;
  }
  ctx.padding = 444;
  DynamicArray<uint8_t> state1;
  CaptureState(reg, state1, test::ExpectOK{});
  EXPECT_EQ(state0.size(), state1.size());
  EXPECT_NE(state0, state1); // Some bytes are different
  EXPECT_TRUE(IsEqualState(reg, state0, state1)); // Still "equal"
}

TEST_F(MochiCapture, MissingCtxComponent) {
  // Capture with a ctx components
  reg.set<CComponentA>();
  DynamicArray<uint8_t> state0;
  CaptureState(reg, state0, test::ExpectOK{});

  // Remove the ctx component
  reg.unset<CComponentA>();

  // Can't restore state0 without that component
  RestoreState(reg, state0, test::ExpectNotOK{});

  // If we add the component back, then it will work because the capture system can't tell the
  // difference, and doesn't care.
  reg.set<CComponentA>();
  RestoreState(reg, state0, test::ExpectOK{});

  // If we replace the component with something else, it will not work.
  reg.unset<CComponentA>();
  reg.set<CComponentB>();
  RestoreState(reg, state0, test::ExpectNotOK{});
}

TEST_F(MochiCapture, EntityMismatch) {
  constexpr int kNumEntities = 10;
  DynamicArray<uint8_t> state;
  DynamicArray<entt::entity> entities;
  entities.reserve(kNumEntities);

  for (int i = 0; i < kNumEntities; ++i) {
    // Create some entities with components
    entities.resize(kNumEntities);
    for (auto& e : entities) {
      e = reg.create();
      reg.emplace<CComponentA>(e);
    }

    // Capture state
    state.clear();
    CaptureState(reg, state, test::ExpectOK{});
    RestoreState(reg, state, test::ExpectOK{}); // It works

    // Remove CComponentA from the ith entity without deleting the rest of the entity
    reg.erase<CComponentA>(entities[i]);

    // Can't restore without that component
    RestoreState(reg, state, test::ExpectNotOK{});

    // Re-add the component and try to restore
    Error restoreError;
    reg.emplace<CComponentA>(entities[i]);
    RestoreState(reg, state, restoreError);

    // Should RestoreState succeed in this case? It depends on whether or not the components in the
    // storage pool exactly match the original order. Because of EnTT's "swap-and-pop"
    // implementation, the order will only match if it was the last component in the pool.
    if (i == kNumEntities - 1) {
      EXPECT_OK(restoreError);
    } else {
      EXPECT_NOT_OK(restoreError);
    }

    // Capture again
    state.clear();
    CaptureState(reg, state, test::ExpectOK{});
    RestoreState(reg, state, test::ExpectOK{}); // It works

    // Destroy the ith entity
    reg.destroy(entities[i]);

    // Can't restore state. Number of entities with CComponentA has decreased.
    RestoreState(reg, state, test::ExpectNotOK{});

    // Re-create the ith entity
    entities[i] = reg.create();
    reg.emplace<CComponentA>(entities[i]);

    // Still can't restore state because the entt::entity identifier has changed
    RestoreState(reg, state, test::ExpectNotOK{});

    // Capture again
    state.clear();
    CaptureState(reg, state, test::ExpectOK{});
    RestoreState(reg, state, test::ExpectOK{}); // It works

    // Add another entity with CComponentA
    auto newGuy = reg.create();
    reg.emplace<CComponentA>(newGuy);

    // Can't restore because the number of entities with CComponentA has increased
    RestoreState(reg, state, test::ExpectNotOK{});

    // Cleanup
    reg.destroy(newGuy);
    for (auto e : entities) {
      reg.destroy(e);
    }
  }
}

// An ECS system with observable side effects
static void IncrementComponentA(CComponentA& comp) {
  comp.ival += 1;
}

// Another ECS system with observable side effects
static void DoubleComponentA(CComponentA& comp) {
  comp.ival *= 2;
}

TEST_F(MochiCapture, RegisterPostRestoreCallback) {
  capture::RegisterPostRestoreSystem(IncrementComponentA, reg);
  capture::RegisterPostRestoreSystem(DoubleComponentA, reg);

  // Create an entity
  auto e0 = reg.create();
  auto& compA0 = reg.emplace<CComponentA>(e0);
  compA0.ival = 10;

  // Capture
  DynamicArray<uint8_t> state;
  CaptureState(reg, state, test::ExpectOK{});
  EXPECT_EQ(10, compA0.ival);

  // Restore
  RestoreState(reg, state, test::ExpectOK{});

  // Observe that the two post-restore systems were invoked, in the order registered.
  EXPECT_EQ(22, compA0.ival);

  // Add another entity
  auto e1 = reg.create();
  reg.emplace<CComponentA>(e1);

  // Fail to restore state because of the new entity
  RestoreState(reg, state, test::ExpectNotOK{});

  // Observe that the post-restore systems were not invoked.
  EXPECT_EQ(22, compA0.ival); // no change)
}

TEST_F(MochiCapture, CaptureStateToJson) {
  // Set up some context
  reg.set<CComponentA>().ival = 111;
  reg.set<CComponentB>().strings.push_back("woot");

  // Rigid actor "Bob" with CComponentA
  auto e0 = reg.create();
  reg.emplace<CActorInfo>(e0, "Bob", ActorType::Rigid);
  reg.emplace<CComponentA>(e0).ival = 222;

  // Rigid actor "Bob" (again) with CComponentB
  auto e1 = reg.create();
  reg.emplace<CActorInfo>(e1, "Bob", ActorType::Rigid);
  reg.emplace<CComponentB>(e1).strings.push_back("RigidStuff");

  // Soft actor with no name and CComponentB
  auto e2 = reg.create();
  reg.emplace<CActorInfo>(e2, "", ActorType::Soft);
  reg.emplace<CComponentB>(e2).strings.push_back("SoftStuff");

  // Soft actor with no name and both components
  auto e3 = reg.create();
  reg.emplace<CActorInfo>(e3, "", ActorType::Soft);
  reg.emplace<CComponentA>(e3).ival = 333;
  reg.emplace<CComponentB>(e3).strings.push_back("MoreSoftStuff");

  // Expected multi-line JSON
  std::string expectedJson = Format(
      R"({
  "actors": {
    "Bob": {
      "_handle": %u,
      "_name": "Bob",
      "_type": "Rigid",
      "components": {
        "CComponentA": {
          "fval": 0,
          "ival": 222
        }
      }
    },
    "Bob2": {
      "_handle": %u,
      "_name": "Bob",
      "_type": "Rigid",
      "components": {
        "CComponentB": {
          "ival": 0,
          "strings": [
            "RigidStuff"
          ]
        }
      }
    },
    "Soft": {
      "_handle": %u,
      "_name": "",
      "_type": "Soft",
      "components": {
        "CComponentB": {
          "ival": 0,
          "strings": [
            "SoftStuff"
          ]
        }
      }
    },
    "Soft2": {
      "_handle": %u,
      "_name": "",
      "_type": "Soft",
      "components": {
        "CComponentA": {
          "fval": 0,
          "ival": 333
        },
        "CComponentB": {
          "ival": 0,
          "strings": [
            "MoreSoftStuff"
          ]
        }
      }
    }
  },
  "scene": {
    "CComponentA": {
      "fval": 0,
      "ival": 111
    },
    "CComponentB": {
      "ival": 0,
      "strings": [
        "woot"
      ]
    }
  }
})",
      e0,
      e1,
      e2,
      e3);

  std::string actualJson = CaptureStateToJson(reg, /*prettyMultiLine*/ false, test::ExpectOK{});

  // Ignore white space for the purpose of this test.
  expectedJson.erase(
      std::remove_if(expectedJson.begin(), expectedJson.end(), isspace), expectedJson.end());
  actualJson.erase(std::remove_if(actualJson.begin(), actualJson.end(), isspace), actualJson.end());
  EXPECT_STREQ(expectedJson.c_str(), actualJson.c_str());
}

TEST_F(MochiCapture, RestorePartialState) {
  // Add some ctx components
  reg.set<CComponentA>().ival = 11;
  reg.set<CComponentB>().ival = 22;
  reg.set<CComponentAB>().ival = 33;
  reg.set<CComponentWithPadding>().ival = 44;

  // Add an entity with various components
  auto const e = reg.create();
  reg.emplace<CComponentA>(e).ival = 55;
  reg.emplace<CComponentB>(e).ival = 66;
  reg.emplace<CComponentAB>(e).ival = 77;
  reg.emplace<CComponentWithPadding>(e).ival = 88;

  // Capture
  DynamicArray<uint8_t> state0;
  CaptureState(reg, state0, test::ExpectOK{});

  // Expect state 0 values
  auto expectState0 = [&]() {
    EXPECT_EQ(11, reg.ctx<CComponentA const>().ival);
    EXPECT_EQ(22, reg.ctx<CComponentB const>().ival);
    EXPECT_EQ(33, reg.ctx<CComponentAB const>().ival);
    EXPECT_EQ(44, reg.ctx<CComponentWithPadding const>().ival);
    EXPECT_EQ(55, reg.get<CComponentA const>(e).ival);
    EXPECT_EQ(66, reg.get<CComponentB const>(e).ival);
    EXPECT_EQ(77, reg.get<CComponentAB const>(e).ival);
    EXPECT_EQ(88, reg.get<CComponentWithPadding const>(e).ival);
  };
  expectState0();

  // Modify data
  reg.ctx<CComponentA>().ival = 111;
  reg.ctx<CComponentB>().ival = 222;
  reg.ctx<CComponentAB>().ival = 333;
  reg.ctx<CComponentWithPadding>().ival = 444;
  reg.get<CComponentA>(e).ival = 555;
  reg.get<CComponentB>(e).ival = 666;
  reg.get<CComponentAB>(e).ival = 777;
  reg.get<CComponentWithPadding>(e).ival = 888;

  // Capture again
  DynamicArray<uint8_t> state1;
  CaptureState(reg, state1, test::ExpectOK{});

  // Restore full state 0
  DynamicArray<SReflect::TypeId> excludedAttriutes;
  RestorePartialState(reg, state0, MakeConstSpan(excludedAttriutes), test::ExpectOK{});
  expectState0();

  // Restore state 1, excluding components with TestAttributeA
  //  Included: CComponentB, CComponentWithPadding
  //  Excluded: CComponentA, CComponentAB
  excludedAttriutes.push_back(attribute::TestAttributeA::GetTypeId());
  RestorePartialState(reg, state1, MakeConstSpan(excludedAttriutes), test::ExpectOK{});
  EXPECT_EQ(11, reg.ctx<CComponentA const>().ival);
  EXPECT_EQ(222, reg.ctx<CComponentB const>().ival); // restored
  EXPECT_EQ(33, reg.ctx<CComponentAB const>().ival);
  EXPECT_EQ(444, reg.ctx<CComponentWithPadding const>().ival); // restored
  EXPECT_EQ(55, reg.get<CComponentA const>(e).ival);
  EXPECT_EQ(666, reg.get<CComponentB const>(e).ival); // restored
  EXPECT_EQ(77, reg.get<CComponentAB const>(e).ival);
  EXPECT_EQ(888, reg.get<CComponentWithPadding const>(e).ival);

  // Restore full stat 0
  RestoreState(reg, state0, test::ExpectOK{});
  expectState0();

  // Restore state 1, excluding components with TestAttributeB
  //  Included: CComponentA, CComponentWithPadding
  //  Excluded: CComponentB, CComponentAB
  excludedAttriutes = {attribute::TestAttributeB::GetTypeId()};
  RestorePartialState(reg, state1, MakeConstSpan(excludedAttriutes), test::ExpectOK{});
  EXPECT_EQ(111, reg.ctx<CComponentA const>().ival); // restored
  EXPECT_EQ(22, reg.ctx<CComponentB const>().ival);
  EXPECT_EQ(33, reg.ctx<CComponentAB const>().ival);
  EXPECT_EQ(444, reg.ctx<CComponentWithPadding const>().ival); // restored
  EXPECT_EQ(555, reg.get<CComponentA const>(e).ival); // restored
  EXPECT_EQ(66, reg.get<CComponentB const>(e).ival);
  EXPECT_EQ(77, reg.get<CComponentAB const>(e).ival);
  EXPECT_EQ(888, reg.get<CComponentWithPadding const>(e).ival); // restored

  // Restore full stat 0
  RestoreState(reg, state0, test::ExpectOK{});
  expectState0();

  // Restore state 1, excluding components with TestAttributeA or TestAttributeB
  //  Included: CComponentWithPadding
  //  Excluded: CComponentA, CComponentB, CComponentAB
  excludedAttriutes = {
      attribute::TestAttributeA::GetTypeId(), attribute::TestAttributeB::GetTypeId()};
  RestorePartialState(reg, state1, MakeConstSpan(excludedAttriutes), test::ExpectOK{});
  EXPECT_EQ(11, reg.ctx<CComponentA const>().ival);
  EXPECT_EQ(22, reg.ctx<CComponentB const>().ival);
  EXPECT_EQ(33, reg.ctx<CComponentAB const>().ival);
  EXPECT_EQ(444, reg.ctx<CComponentWithPadding const>().ival); // restored
  EXPECT_EQ(55, reg.get<CComponentA const>(e).ival);
  EXPECT_EQ(66, reg.get<CComponentB const>(e).ival);
  EXPECT_EQ(77, reg.get<CComponentAB const>(e).ival);
  EXPECT_EQ(888, reg.get<CComponentWithPadding const>(e).ival); // restored
}
