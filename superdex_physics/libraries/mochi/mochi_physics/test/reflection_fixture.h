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

#pragma once

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <string>
#include <string_view>

namespace mochi::reflection_fixture {

struct Leaf {
  int count = 0;
};

struct DataView {
  Leaf leaf;
  Span<real const> values;
};

struct SamplesView;

struct Samples {
  Samples() = default;
  Samples(SamplesView const& view);
  DynamicArray<real> values;
};

struct SamplesView {
  SamplesView() = default;
  SamplesView(Samples const& samples) : values(MakeConstSpan(samples.values)) {}
  Span<real const> values;
};

inline Samples::Samples(SamplesView const& view) : values(view.values.begin(), view.values.end()) {}

struct BaseData {
  real GetValue() const {
    return _value;
  }
  void SetValue(real value, Error& error) {
    MOCHI_ERROR_IF_NOT(value >= 0_r, error, "Value must be nonnegative");
    MOCHI_ERROR_RETURN(error);
    _value = value;
  }

 protected:
  mutable real _value = 1_r;
};

struct HiddenSetterData : BaseData {
  void SetValue(int value) {
    _ignored = value;
  }

 private:
  int _ignored = 0;
};

struct Data : BaseData {
  real& GetReference() const {
    return _value;
  }
  std::string_view GetLabel() const {
    return _label;
  }
  void SetLabel(std::string_view label) {
    _label = label;
  }
  bool IsReady() const {
    return _ready;
  }
  void Enable(bool ready) {
    _ready = ready;
  }
  DataView GetData() const {
    return {{static_cast<int>(_values.size())}, MakeConstSpan(_values)};
  }
  SamplesView GetSampleView() const {
    SamplesView view;
    view.values = MakeConstSpan(_values);
    return view;
  }
  int CountSamples(SamplesView samples) const {
    return static_cast<int>(samples.values.size());
  }
  real GetChecked(Error& error) const {
    MOCHI_ERROR_IF_NOT(false, error, "Checked value is unavailable");
    return 0_r;
  }
  void* GetPointer() const {
    return pointer;
  }
  Span<real const> Scale(Span<real> values) const {
    for (auto& value : values) {
      value *= 2_r;
    }
    return values;
  }
  void SetSamples(Span<real const> values) {
    _values.assign(values.begin(), values.end());
  }
  int Read(Span<real const> const&) const {
    return 1;
  }
  int Read(Span<real const>&) const {
    return 2;
  }
  static Data Default() {
    return {};
  }

  void* pointer = nullptr;

 private:
  std::string _label;
  bool _ready = false;
  DynamicArray<real> _values = {1_r, 2_r};
};

struct VirtualBase {
  virtual ~VirtualBase() = default;
  virtual int Run() const {
    return 1;
  }
  virtual int GetValue() const {
    return _value;
  }
  virtual void SetValue(int value) {
    _value = value;
  }

 protected:
  int _value = 0;
};

struct VirtualDerived : VirtualBase {
  int Run() const override {
    return 3;
  }
  int GetValue() const override {
    return _value + 10;
  }
  void SetValue(int value) override {
    _value = value * 2;
  }
};

struct FieldAndAccessors {
  int value = 0;
  int GetValue() const {
    return value + 1;
  }
  void SetValue(int input) {
    value = input * 2;
  }
};

struct DerivedField : BaseData {
  real localValue = 3_r;
};

} // namespace mochi::reflection_fixture

#include "reflection_fixture.generated.h"
