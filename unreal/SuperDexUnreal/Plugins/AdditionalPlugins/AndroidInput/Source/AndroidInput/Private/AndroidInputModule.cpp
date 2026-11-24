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

#include "AndroidInputModule.h"

#include "Async/Async.h"
#include "Engine/Engine.h"
#include "Engine/GameInstance.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/PlayerInput.h"
#include "GenericPlatform/GenericPlatformInputDeviceMapper.h"
#include "InputCoreTypes.h"
#include "Misc/ScopeLock.h"

#if PLATFORM_ANDROID
#include <android/input.h>
#include <android/keycodes.h>
#include <android_native_app_glue.h>

extern struct android_app* GNativeAndroidApp;
#endif // PLATFORM_ANDROID

#define LOCTEXT_NAMESPACE "FAndroidInputModule"

// Custom Android Input keys
const FKey FAndroidInputKeys::AndroidPrintScreen("AndroidInput_PrintScreen");

void FAndroidInputKeys::RegisterKeys() {
  EKeys::AddMenuCategoryDisplayInfo(
      "AndroidInput",
      LOCTEXT("AndroidInputSubCategory", "Android Input"),
      TEXT("GraphEditor.KeyEvent_16x"));

  EKeys::AddKey(FKeyDetails(
      AndroidPrintScreen,
      LOCTEXT("AndroidInput_PrintScreen", "Android Input Print Screen Key"),
      FKeyDetails::GamepadKey,
      TEXT("AndroidInput")));
}

#if PLATFORM_ANDROID
static constexpr float AxisDeadzone = 0.01f;

#define UE_ANDROIDINPUT_KEYMAP(AndroidKey, UnrealKey) {AndroidKey, UnrealKey}

// clang-format off
static const TMap<int32_t, FKey> GamepadKeyMap = {
    // Face buttons
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_A,      EKeys::Gamepad_FaceButton_Bottom),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_B,      EKeys::Gamepad_FaceButton_Right),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_X,      EKeys::Gamepad_FaceButton_Left),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_Y,      EKeys::Gamepad_FaceButton_Top),
    // D-Pad
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_DPAD_UP,       EKeys::Gamepad_DPad_Up),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_DPAD_DOWN,     EKeys::Gamepad_DPad_Down),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_DPAD_LEFT,     EKeys::Gamepad_DPad_Left),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_DPAD_RIGHT,    EKeys::Gamepad_DPad_Right),
    // Shoulders and triggers
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_L1,     EKeys::Gamepad_LeftShoulder),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_R1,     EKeys::Gamepad_RightShoulder),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_L2,     EKeys::Gamepad_LeftTrigger),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_R2,     EKeys::Gamepad_RightTrigger),
    // Thumbsticks
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_THUMBL, EKeys::Gamepad_LeftThumbstick),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_THUMBR, EKeys::Gamepad_RightThumbstick),
    // Start/Select
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_START,  EKeys::Gamepad_Special_Right),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BUTTON_SELECT, EKeys::Gamepad_Special_Left),
};

static const TMap<int32_t, FKey> KeyboardKeyMap = {
    // Letters A-Z
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_A, EKeys::A), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_B, EKeys::B), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_C, EKeys::C),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_D, EKeys::D), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_E, EKeys::E), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F, EKeys::F),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_G, EKeys::G), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_H, EKeys::H), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_I, EKeys::I),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_J, EKeys::J), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_K, EKeys::K), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_L, EKeys::L),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_M, EKeys::M), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_N, EKeys::N), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_O, EKeys::O),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_P, EKeys::P), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_Q, EKeys::Q), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_R, EKeys::R),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_S, EKeys::S), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_T, EKeys::T), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_U, EKeys::U),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_V, EKeys::V), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_W, EKeys::W), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_X, EKeys::X),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_Y, EKeys::Y), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_Z, EKeys::Z),
    // Numbers 0-9
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_0, EKeys::Zero),  UE_ANDROIDINPUT_KEYMAP(AKEYCODE_1, EKeys::One),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_2, EKeys::Two),   UE_ANDROIDINPUT_KEYMAP(AKEYCODE_3, EKeys::Three),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_4, EKeys::Four),  UE_ANDROIDINPUT_KEYMAP(AKEYCODE_5, EKeys::Five),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_6, EKeys::Six),   UE_ANDROIDINPUT_KEYMAP(AKEYCODE_7, EKeys::Seven),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_8, EKeys::Eight), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_9, EKeys::Nine),
    // Function keys F1-F12
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F1,  EKeys::F1),  UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F2,  EKeys::F2),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F3,  EKeys::F3),  UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F4,  EKeys::F4),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F5,  EKeys::F5),  UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F6,  EKeys::F6),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F7,  EKeys::F7),  UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F8,  EKeys::F8),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F9,  EKeys::F9),  UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F10, EKeys::F10),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F11, EKeys::F11), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_F12, EKeys::F12),
    // Numpad 0-9
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_0, EKeys::NumPadZero),  UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_1, EKeys::NumPadOne),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_2, EKeys::NumPadTwo),   UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_3, EKeys::NumPadThree),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_4, EKeys::NumPadFour),  UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_5, EKeys::NumPadFive),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_6, EKeys::NumPadSix),   UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_7, EKeys::NumPadSeven),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_8, EKeys::NumPadEight), UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_9, EKeys::NumPadNine),
    // Numpad operators
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_ADD,      EKeys::Add),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_SUBTRACT, EKeys::Subtract),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_MULTIPLY, EKeys::Multiply),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_DIVIDE,   EKeys::Divide),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_DOT,      EKeys::Decimal),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUMPAD_ENTER,    EKeys::Enter),
    // Modifier keys
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_SHIFT_LEFT,  EKeys::LeftShift),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_SHIFT_RIGHT, EKeys::RightShift),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_CTRL_LEFT,   EKeys::LeftControl),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_CTRL_RIGHT,  EKeys::RightControl),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_ALT_LEFT,    EKeys::LeftAlt),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_ALT_RIGHT,   EKeys::RightAlt),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_META_LEFT,   EKeys::LeftCommand),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_META_RIGHT,  EKeys::RightCommand),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_CAPS_LOCK,   EKeys::CapsLock),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_NUM_LOCK,    EKeys::NumLock),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_SCROLL_LOCK, EKeys::ScrollLock),
    // Special keys
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_SPACE,       EKeys::SpaceBar),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_ENTER,       EKeys::Enter),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_ESCAPE,      EKeys::Escape),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_TAB,         EKeys::Tab),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_DEL,         EKeys::BackSpace),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_FORWARD_DEL, EKeys::Delete),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_INSERT,      EKeys::Insert),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_MOVE_HOME,   EKeys::Home),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_MOVE_END,    EKeys::End),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_PAGE_UP,     EKeys::PageUp),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_PAGE_DOWN,   EKeys::PageDown),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BREAK,       EKeys::Pause),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_SYSRQ,       FAndroidInputKeys::AndroidPrintScreen),
    // Arrow keys
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_DPAD_UP,    EKeys::Up),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_DPAD_DOWN,  EKeys::Down),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_DPAD_LEFT,  EKeys::Left),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_DPAD_RIGHT, EKeys::Right),
    // Punctuation and symbols
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_GRAVE,         EKeys::Tilde),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_MINUS,         EKeys::Hyphen),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_EQUALS,        EKeys::Equals),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_LEFT_BRACKET,  EKeys::LeftBracket),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_RIGHT_BRACKET, EKeys::RightBracket),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_BACKSLASH,     EKeys::Backslash),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_SEMICOLON,     EKeys::Semicolon),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_APOSTROPHE,    EKeys::Apostrophe),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_COMMA,         EKeys::Comma),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_PERIOD,        EKeys::Period),
    UE_ANDROIDINPUT_KEYMAP(AKEYCODE_SLASH,         EKeys::Slash),
};
// clang-format on

#undef UE_ANDROIDINPUT_KEYMAP

FKey AndroidKeyCodeToGamepadFKey(int32_t KeyCode) {
  const FKey* Found = GamepadKeyMap.Find(KeyCode);
  return Found ? *Found : EKeys::Invalid;
}

FKey AndroidKeyCodeToKeyboardFKey(int32_t KeyCode) {
  const FKey* Found = KeyboardKeyMap.Find(KeyCode);
  return Found ? *Found : EKeys::Invalid;
}

void DispatchGamepadButtonEvent(const FKey& Key, bool bPressed) {
  if (!Key.IsValid()) {
    return;
  }

  AsyncTask(ENamedThreads::GameThread, [Key, bPressed]() {
    if (GEngine == nullptr) {
      return;
    }

    FInputDeviceId DeviceId = IPlatformInputDeviceMapper::Get().GetDefaultInputDevice();

    for (const FWorldContext& Context : GEngine->GetWorldContexts()) {
      UWorld* World = Context.World();
      if (World == nullptr) {
        continue;
      }

      for (FConstPlayerControllerIterator It = World->GetPlayerControllerIterator(); It; ++It) {
        APlayerController* PC = It->Get();
        if (PC != nullptr && PC->PlayerInput != nullptr) {
          FInputKeyParams Params(
              Key, bPressed ? IE_Pressed : IE_Released, bPressed ? 1.0 : 0.0, true, DeviceId);
          PC->InputKey(Params);
        }
      }
    }
  });
}

void DispatchGamepadAxisEvent(const FKey& Axis, float Value) {
  if (!Axis.IsValid()) {
    return;
  }

  AsyncTask(ENamedThreads::GameThread, [Axis, Value]() {
    if (GEngine == nullptr) {
      return;
    }

    FInputDeviceId DeviceId = IPlatformInputDeviceMapper::Get().GetDefaultInputDevice();

    for (const FWorldContext& Context : GEngine->GetWorldContexts()) {
      UWorld* World = Context.World();
      if (World == nullptr) {
        continue;
      }

      for (FConstPlayerControllerIterator It = World->GetPlayerControllerIterator(); It; ++It) {
        APlayerController* PC = It->Get();
        if (PC != nullptr && PC->PlayerInput != nullptr) {
          FInputKeyParams Params(Axis, Value, 0.0f, 1, true, DeviceId);
          PC->InputKey(Params);
        }
      }
    }
  });
}

void ProcessMotionEvent(AInputEvent* Event) {
  FAndroidInputModule* Module = FAndroidInputModule::Get();
  if (Module == nullptr) {
    return;
  }

  FScopeLock Lock(&Module->AxisStateLock);

  auto CheckAndDispatchAxis = [&Module](const FKey& Axis, float& StoredValue, float NewValue) {
    if (FMath::Abs(NewValue - StoredValue) > AxisDeadzone) {
      StoredValue = NewValue;
      DispatchGamepadAxisEvent(Axis, NewValue);
    }
  };

  float LeftX = AMotionEvent_getAxisValue(Event, AMOTION_EVENT_AXIS_X, 0);
  float LeftY = AMotionEvent_getAxisValue(Event, AMOTION_EVENT_AXIS_Y, 0);
  // NB: Z is second thumbstick X
  float RightX = AMotionEvent_getAxisValue(Event, AMOTION_EVENT_AXIS_Z, 0);
  // NB: RZ is second thumbstick Y
  float RightY = AMotionEvent_getAxisValue(Event, AMOTION_EVENT_AXIS_RZ, 0);
  float LeftTrigger = AMotionEvent_getAxisValue(Event, AMOTION_EVENT_AXIS_LTRIGGER, 0);
  float RightTrigger = AMotionEvent_getAxisValue(Event, AMOTION_EVENT_AXIS_RTRIGGER, 0);

  CheckAndDispatchAxis(EKeys::Gamepad_LeftX, Module->AxisState.LeftStickX, LeftX);
  CheckAndDispatchAxis(EKeys::Gamepad_LeftY, Module->AxisState.LeftStickY, LeftY);
  CheckAndDispatchAxis(EKeys::Gamepad_RightX, Module->AxisState.RightStickX, RightX);
  CheckAndDispatchAxis(EKeys::Gamepad_RightY, Module->AxisState.RightStickY, RightY);
  CheckAndDispatchAxis(EKeys::Gamepad_LeftTriggerAxis, Module->AxisState.LeftTrigger, LeftTrigger);
  CheckAndDispatchAxis(
      EKeys::Gamepad_RightTriggerAxis, Module->AxisState.RightTrigger, RightTrigger);
}

void ProcessGamepadInput(struct android_app* App, struct android_poll_source* Source) {
  AInputEvent* Event = nullptr;
  while (AInputQueue_getEvent(App->inputQueue, &Event) >= 0) {
    int32_t EventType = AInputEvent_getType(Event);

    bool bIsSupportedInput = false;

    if (EventType == AINPUT_EVENT_TYPE_KEY) {
      int32_t KeyCode = AKeyEvent_getKeyCode(Event);
      int32_t InputSource = AInputEvent_getSource(Event);
      FKey Key = EKeys::Invalid;

      if ((InputSource & AINPUT_SOURCE_GAMEPAD) == AINPUT_SOURCE_GAMEPAD ||
          (InputSource & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK) {
        Key = AndroidKeyCodeToGamepadFKey(KeyCode);
        bIsSupportedInput = true;
      } else if ((InputSource & AINPUT_SOURCE_KEYBOARD) == AINPUT_SOURCE_KEYBOARD) {
        Key = AndroidKeyCodeToKeyboardFKey(KeyCode);
        bIsSupportedInput = true;
      }

      if (Key.IsValid()) {
        int32_t Action = AKeyEvent_getAction(Event);
        if (Action == AKEY_EVENT_ACTION_DOWN) {
          DispatchGamepadButtonEvent(Key, true);
        } else if (Action == AKEY_EVENT_ACTION_UP) {
          DispatchGamepadButtonEvent(Key, false);
        }
      }
    } else if (EventType == AINPUT_EVENT_TYPE_MOTION) {
      int32_t InputSource = AInputEvent_getSource(Event);
      if ((InputSource & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK ||
          (InputSource & AINPUT_SOURCE_GAMEPAD) == AINPUT_SOURCE_GAMEPAD) {
        ProcessMotionEvent(Event);
        bIsSupportedInput = true;
      }
    }

    if (!bIsSupportedInput && AInputQueue_preDispatchEvent(App->inputQueue, Event)) {
      continue;
    }

    int32_t Handled = 0;
    if (App->onInputEvent != nullptr) {
      Handled = App->onInputEvent(App, Event);
    }
    AInputQueue_finishEvent(App->inputQueue, Event, Handled);
  }
}

void InstallInputHook() {
  if (GNativeAndroidApp != nullptr) {
    GNativeAndroidApp->inputPollSource.process = &ProcessGamepadInput;
    UE_LOG(LogAndroidInput, Log, TEXT("Android input hook installed successfully"));
  } else {
    UE_LOG(LogAndroidInput, Warning, TEXT("GNativeAndroidApp is null, cannot install input hook"));
  }
}

#endif // PLATFORM_ANDROID

DEFINE_LOG_CATEGORY(LogAndroidInput);

void FAndroidInputModule::StartupModule() {
  FAndroidInputKeys::RegisterKeys();
#if PLATFORM_ANDROID
  InstallInputHook();
#else
  UE_LOG(
      LogAndroidInput,
      Log,
      TEXT("AndroidInput module loaded (non-Android platform, hook not installed)"));
#endif
}

void FAndroidInputModule::ShutdownModule() {
  UE_LOG(LogAndroidInput, Display, TEXT("FAndroidInputModule::ShutdownModule"));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FAndroidInputModule, AndroidInput)
