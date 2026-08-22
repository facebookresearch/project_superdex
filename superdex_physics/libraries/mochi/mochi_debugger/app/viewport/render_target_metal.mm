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

// Metal texture helper functions for render_target.cpp
// This file provides Metal texture creation/destruction that can be called from C++

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <imguios/application.h>

// Forward declarations for extern "C" functions
extern "C" {
void* GetMetalDevice();
void* CreateMetalTexture(void* devicePtr, int width, int height);
void* CreateMetalTextureRefForImGui(void* filamentOwnedTexture);
void DestroyMetalTexture(void* texturePtr);
}

extern "C" {

// Get the Metal device from ImGuiOS
void* GetMetalDevice() {
  // ImGuios::GetMetalDevice() returns id<MTLDevice> as void*
  return ImGuios::GetMetalDevice();
}

// Create a Metal texture for use with both Filament and ImGui
// Returns the texture with ownership transferred via CFBridgingRetain
// The caller (Filament) will release it via CFBridgingRelease
void* CreateMetalTexture(void* devicePtr, int width, int height) {
  if (!devicePtr || width <= 0 || height <= 0) {
    return nullptr;
  }

  auto device = (__bridge id<MTLDevice>)devicePtr;

  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  descriptor.storageMode = MTLStorageModePrivate;

  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (!texture) {
    return nullptr;
  }

  // Transfer ownership to Filament via CFBridgingRetain
  // Filament will call CFBridgingRelease when it destroys the texture
  return (void*)CFBridgingRetain(texture);
}

// Create a second reference to the texture for ImGui display
// This is needed because Filament takes ownership of the first reference
void* CreateMetalTextureRefForImGui(void* filamentOwnedTexture) {
  if (!filamentOwnedTexture) {
    return nullptr;
  }
  // The filamentOwnedTexture is already retained by Filament
  // We need an additional retain for ImGui to use
  auto texture = (__bridge id<MTLTexture>)filamentOwnedTexture;
  return (void*)CFBridgingRetain(texture);
}

// Release the ImGui reference to the texture
void DestroyMetalTexture(void* texturePtr) {
  if (texturePtr) {
    // Release the retained reference
    CFBridgingRelease(texturePtr);
  }
}

} // extern "C"
