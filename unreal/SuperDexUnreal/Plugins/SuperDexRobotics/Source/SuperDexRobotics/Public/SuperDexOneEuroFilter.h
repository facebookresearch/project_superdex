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

#include "CoreMinimal.h"

/**
 * A simple, adaptive low-pass filter that can be used with variable time steps, implemented as
 * described by Casiez et al. (2012) in:
 * https://dl.acm.org/doi/abs/10.1145/2207676.2208639
 *
 * To tune this filter, first set Beta to zero and adjust MinCutoff to the largest value that will
 * eliminate noise and jitter during slow motions (signal change). Then increase Beta until filter
 * lag is reduced to desired amount during fast motions (high signal velocity). DCutoff should
 * ideally be adjusted to provide a smooth estimate of velocity (signal derivative).
 *
 * T is a scalar (float or double) or an FVector.
 */
template <typename T>
struct TSuperDexOneEuroFilter {
  /**
   * Construct an instance of the One Euro filter with the given parameters.
   * @param InMinCutoff   The minimum cutoff frequency of the filter, in Hz.
   * @param InBeta        The filter adaptation slope [1 / signal unit] or [Hz / unit velocity].
   * @param InDcutoff     The cutoff frequency for smoothed signal derivative estimation, in Hz.
   */
  TSuperDexOneEuroFilter(double InMinCutoff = 1.0, double InBeta = 0.0, double InDcutoff = 1.0)
      : MinCutoff(InMinCutoff),
        Beta(InBeta),
        Dcutoff(InDcutoff),
        PreviousValue(T{}),
        PreviousDerivative(T{}),
        bFirstTime(true) {}

  /**
   * Run one iteration of the filter with the given time step.
   * @param Value       The next value in the signal.
   * @param DeltaTime   The time step between the current and previous values, in seconds.
   * @return            The filtered value.
   */
  T Filter(T Value, double DeltaTime) {
    if (bFirstTime) {
      PreviousValue = Value;
      PreviousDerivative = T{};
      bFirstTime = false;
      return Value;
    }
    T Derivative = (Value - PreviousValue) / DeltaTime;
    T SmoothedDerivative = ExponentialSmoothing(Derivative, Alpha(Dcutoff), PreviousDerivative);
    double Cutoff = Beta * Magnitude(SmoothedDerivative) + MinCutoff;
    T SmoothedValue = ExponentialSmoothing(Value, Alpha(Cutoff), PreviousValue);
    PreviousValue = SmoothedValue;
    PreviousDerivative = SmoothedDerivative;
    return SmoothedValue;
  }

  // ~ Static filter parameters; see constructor for description ~
  double MinCutoff;
  double Beta;
  double Dcutoff;

 private:
  static double Alpha(double Cutoff) {
    double Tau = 1.0 / (2 * PI * Cutoff);
    return 1.0 / (1.0 + Tau);
  }

  static T ExponentialSmoothing(T Value, double Alpha, T PreviousValue) {
    return Alpha * Value + (1.0 - Alpha) * PreviousValue;
  }

  static double Magnitude(double In) {
    return FMath::Abs(In);
  }

  static double Magnitude(const FVector& In) {
    return In.Length();
  }

  // ~ Dynamic filter state ~
  T PreviousValue;
  T PreviousDerivative;
  bool bFirstTime;
};
