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

#include "Engine/Engine.h"

#include "SuperDexPoseUtils.generated.h"

// Handy macros to print text to the screen (not the Output Log). _K versions
// take any positive integer `Key` which will cause messages previously submitted
// with the same `Key` to be pushed off screen immediately. This is useful if you
// want to print a message from TickComponent without spamming the message queue.

// clang-format off

#define SUPERDEX_SCREEN_PRINTF(Format, ...) do { if (GEngine) { GEngine->AddOnScreenDebugMessage(static_cast<uint64>(-1), 2.f, FColor::White, FString::Printf(TEXT(Format), ##__VA_ARGS__), false, FVector2D(2,2)); } } while (0)
#define SUPERDEX_SCREEN_PRINTF_K(Key, Format, ...) do { if (GEngine) { GEngine->AddOnScreenDebugMessage(static_cast<uint64>(Key), 2.f, FColor::White, FString::Printf(TEXT(Format), ##__VA_ARGS__), true, FVector2D(2,2)); } } while (0)
#define SUPERDEX_SCREEN_PRINTF_C(Color, Format, ...) do { if (GEngine) { GEngine->AddOnScreenDebugMessage(static_cast<uint64>(-1), 2.f, Color, FString::Printf(TEXT(Format), ##__VA_ARGS__), false, FVector2D(2,2)); } } while (0)
#define SUPERDEX_SCREEN_PRINTF_KC(Key, Color, Format, ...) do { if (GEngine) { GEngine->AddOnScreenDebugMessage(static_cast<uint64>(Key), 2.f, Color, FString::Printf(TEXT(Format), ##__VA_ARGS__), true, FVector2D(2,2)); } } while (0)

// clang-format on

//------------------------------------------------------------------------------

/** Centimeters per meter. */
constexpr float MetersToCm = 100.f;

/** Converts an ANSI C string to an FString. */
inline FString ToFString(char const* Str) {
  return ANSI_TO_TCHAR(Str);
}

/**
 * Converts a vector from a right-handed frame to the left-handed Unreal frame by swapping Y and Z.
 */
inline FVector RightHandedToLeftHanded(FVector const& V) {
  return FVector(V.X, V.Z, V.Y);
}

/**
 * Converts a rotation from a right-handed frame to the left-handed Unreal frame (see the FVector
 * overload).
 */
inline FQuat RightHandedToLeftHanded(FQuat const& Q) {
  return FQuat(-Q.X, -Q.Z, -Q.Y, Q.W);
}

/**
 * Converts a transform's rotation and translation from a right-handed frame to the left-handed
 * Unreal frame; the result has unit scale.
 */
inline FTransform RightHandedToLeftHanded(FTransform const& T) {
  return FTransform(
      RightHandedToLeftHanded(T.GetRotation()), RightHandedToLeftHanded(T.GetTranslation()));
}

/**
 * Find the first child component of Root of type T, searching all descendants if
 * bIncludeAllDescendents is true and only direct children otherwise.
 */
template <typename T>
T* GetFirstChildComponentByType(USceneComponent const* Root, bool bIncludeAllDescendents) {
  if (Root) {
    TArray<USceneComponent*> Children;
    Root->GetChildrenComponents(bIncludeAllDescendents, Children);
    for (USceneComponent* Child : Children) {
      if (T* MaybeResult = Cast<T>(Child)) {
        return MaybeResult;
      }
    }
  }
  return nullptr;
}

/**
 * Find the first child component under a given root component with the given type and with a name
 * that starts with the given string. This is useful for finding components at runtime based on name
 * and parent relationship.
 */
template <typename T>
T* GetFirstChildComponentByTypeAndNameStartsWith(
    USceneComponent const* Root,
    const TCHAR* Prefix,
    bool bIncludeAllDescendents) {
  if (Root) {
    TArray<USceneComponent*> Children;
    Root->GetChildrenComponents(bIncludeAllDescendents, Children);
    for (USceneComponent* Child : Children) {
      if (T* MaybeResult = Cast<T>(Child)) {
        FString Name = MaybeResult->GetName();
        if (Name.StartsWith(Prefix, ESearchCase::IgnoreCase)) {
          return MaybeResult;
        }
      }
    }
  }
  return nullptr;
}

/**
 * Find the first child component under a given root component with the given type and with a name
 * that contains the given string. This is useful for finding components at runtime based on name
 * and parent relationship.
 */
template <typename T>
T* GetFirstChildComponentByTypeAndNameContains(
    USceneComponent const* Root,
    const TCHAR* Substr,
    bool bIncludeAllDescendents) {
  if (Root) {
    TArray<USceneComponent*> Children;
    Root->GetChildrenComponents(bIncludeAllDescendents, Children);
    for (USceneComponent* Child : Children) {
      if (T* MaybeResult = Cast<T>(Child)) {
        FString Name = MaybeResult->GetName();
        if (Name.Contains(Substr, ESearchCase::CaseSensitive)) {
          return MaybeResult;
        }
      }
    }
  }
  return nullptr;
}

/** Sets the visibility of Root and all of its descendant components. */
inline void SetSelfAndChildrenVisible(USceneComponent* Root, bool bVisible) {
  TArray<USceneComponent*> Children;
  Root->SetVisibility(bVisible);
  Root->GetChildrenComponents(true, Children);
  for (USceneComponent* Child : Children) {
    Child->SetVisibility(bVisible);
  }
}

/** Find the first component of type T on Root whose name is Name. */
template <typename T>
T* FindComponentByName(AActor const* Root, FString Name) {
  TArray<T*> Components;
  Root->GetComponents(Components);
  for (T* Component : Components) {
    if (Component->GetName() == Name) {
      return Component;
    }
  }
  return nullptr;
}

/** Find the first component of type T on Root whose FName is Name. */
template <typename T>
T* FindComponentByFName(AActor const* Root, FName Name) {
  TArray<T*> Components;
  Root->GetComponents(Components);
  for (T* Component : Components) {
    if (Component->GetFName() == Name) {
      return Component;
    }
  }
  return nullptr;
}

/** Find the first component of type T on Root whose name starts with Prefix, ignoring case. */
template <typename T>
T* FindComponentStartsWithName(AActor const* Root, FString Prefix) {
  TArray<T*> Components;
  Root->GetComponents(Components);
  for (T* Component : Components) {
    FString Name = Component->GetName();
    if (Name.StartsWith(Prefix, ESearchCase::IgnoreCase)) {
      return Component;
    }
  }
  return nullptr;
}

/** Find the first actor of type T attached to Parent. */
template <typename T>
T* FindChildActorByType(AActor* Parent) {
  TArray<AActor*> ChildActors;
  Parent->GetAttachedActors(ChildActors);
  for (AActor* ChildActor : ChildActors) {
    T* ChildActorT = Cast<T>(ChildActor);
    if (ChildActorT != nullptr) {
      return ChildActorT;
    }
  }
  return nullptr;
}

/** Shuffles InArray in place with FMath::RandRange. */
template <class T>
void ShuffleArray(T& InArray) {
  const int32 LastIndex = InArray.Num() - 1;
  for (int32 i = 0; i < LastIndex; ++i) {
    int32 Index = FMath::RandRange(0, LastIndex);
    if (i != Index) {
      InArray.Swap(i, Index);
    }
  }
}

/**
 * Adds and registers a new component to an Actor at runtime. This can be used
 * to add components from another component in e.g. BeginPlay or TickComponent,
 * but not from constructors (expect crashes). Optionally pass a PreRegistrationSetup
 * lambda to be executed before the component is registered or attached. This allows you
 * to set component properties before the component's BeginPlay will be called.
 */
template <typename TActorComponent>
TActorComponent* AddComponentRuntime(
    AActor* Actor,
    TFunction<void(TActorComponent*)> PreRegistrationSetup = nullptr) {
  TActorComponent* Component = NewObject<TActorComponent>(Actor);
  if (Component) {
    if (PreRegistrationSetup) {
      PreRegistrationSetup(Component);
    }
    Component->RegisterComponent();
  }
  return Component;
}

/** Like the overload above, and then attaches the component to Parent (if any) at SocketName. */
template <typename TSceneComponent>
TSceneComponent* AddComponentRuntime(
    AActor* Actor,
    USceneComponent* Parent,
    FAttachmentTransformRules AttachmentRules,
    FName SocketName = NAME_None,
    TFunction<void(TSceneComponent*)> PreRegistrationSetup = nullptr) {
  TSceneComponent* Component = NewObject<TSceneComponent>(Actor);
  if (Component) {
    if (PreRegistrationSetup) {
      PreRegistrationSetup(Component);
    }
    Component->RegisterComponent();
    if (Parent) {
      Component->AttachToComponent(Parent, AttachmentRules, SocketName);
    }
  }
  return Component;
}

/**
 * Converts a UENUM value to FString in the form "TEnum::EnumValue".
 * If bStripClassName = true, returns "EnumValue".
 */
template <typename TEnum>
FString UEnumToString(TEnum EnumValue, bool bStripClassName = false) {
  const UEnum* EnumPtr = StaticEnum<TEnum>();
  if (!EnumPtr) {
    return FString("Invalid");
  }
  FString Str = EnumPtr->GetNameByValue((int64)EnumValue).ToString();
  if (bStripClassName) {
    Str.Split("::", nullptr, &Str);
  }
  return Str;
}

/** Relative-transform helpers for scene component hierarchies. */
UCLASS(BlueprintType)
class SUPERDEXPOSE_API USuperDexPoseUtilsFunctionLibrary : public UBlueprintFunctionLibrary {
  GENERATED_BODY()

 public:
  /**
   * Recursively walks up the tree to get the transform from the descendent component to the
   * ancestor component. If it runs into the root before reaching the ancestor it will stop and
   * return the relative transform from the descendent component to the actor's root.
   */
  static FTransform GetRelativeTransform_DescendentToAncestor(
      USceneComponent* AncestorComp,
      USceneComponent* DescendentComp);

  /**
   * Recursively walks up the tree to get the transform relative to its owning actor's root
   * component.
   */
  static FTransform GetRelativeTransform_CompToRoot(USceneComponent* Comp);
};
