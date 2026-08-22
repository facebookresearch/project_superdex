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
#include <superdex_physics.h>
#include <superdex_robotics/superdex_robotics.h>

namespace superdex::robotics {

/**
 * @brief Interface for loading bot data from an abstract source. Implementations provide the
 * concrete I/O strategy (e.g., local filesystem, in-memory assets).
 *
 * @note See @ref FileBotLoader and SuperDexStudioBotLoader for file- and memory-based examples,
 * respectively.
 * @note Pass an @ref IBotLoader to @ref superdex::robotics::LoadBotPrefab to control how files and
 * shapes are resolved.
 */
struct MOCHI_API IBotLoader {
  virtual ~IBotLoader() = default;

  /**
   * @brief Determine the @ref BotFileType of a Mochi Bot source.
   * @param[in] path Abstract path to the Mochi Bot source.
   * @param[in,out] error Error status. Check @ref Error::IsOK for success.
   * @return The detected @ref BotFileType. Do not use the result if @p error is set.
   */
  virtual BotFileType GetBotFileType(std::string_view path, superdex::Error& error) const = 0;

  /**
   * @brief  Load flat @ref BotPrefab from a Mochi Bot source.
   *
   * @param[in] path Abstract path to the Mochi Bot source.
   * @param[in,out] error Error status. Check @ref Error::IsOK for success.
   * @return Loaded @ref BotPrefab, or default-constructed on failure.
   *
   * @note Implementations MUST call @ref RebuildBotData before returning BotPrefab!
   */
  virtual BotPrefab LoadBotPrefab(std::string_view path, superdex::Error& error) const = 0;

  /**
   * @brief Load @ref ModBotPrefab from a Mochi Bot source.
   *
   * @param[in] path Abstract path to the Mochi Bot source.
   * @param[in,out] error Error status. Check @ref Error::IsOK for success.
   * @return Loaded @ref ModBotPrefab, or default-constructed on failure.
   */
  virtual ModBotPrefab LoadModBotPrefab(std::string_view path, superdex::Error& error) const = 0;

  /**
   * @brief  Load a Mochi shape from a model resource.
   *
   * @param[in] path File path to the shape asset.
   * @param[in] bakeScale Scale to bake into the shape geometry.
   * @param[in] bakeTransform Transform to bake into the shape geometry.
   * @param[in] context Mochi context used to create the shape.
   * @param[in,out] error Error status. Check @ref Error::IsOK for success.
   * @return Handle to the loaded shape, or invalid handle on failure.
   */
  virtual ShapeHandle LoadShape(
      std::string_view path,
      Real3 const& bakeScale,
      TransformRT const& bakeTransform,
      Context* context,
      superdex::Error& error) const = 0;
};

/**
 * @brief Default file-system implementation of @ref IBotLoader. Loads @ref  BotPrefab and @ref
 * ModBotPrefab from .superdex_bot files and shape assets from .mochi.h5 models directly from the
 * local filesystem.
 */
struct MOCHI_API FileBotLoader : IBotLoader {
  ~FileBotLoader() override = default;
  BotFileType GetBotFileType(std::string_view path, superdex::Error& error) const override;
  BotPrefab LoadBotPrefab(std::string_view path, superdex::Error& error) const override;
  ModBotPrefab LoadModBotPrefab(std::string_view path, superdex::Error& error) const override;
  ShapeHandle LoadShape(
      std::string_view path,
      Real3 const& bakeScale,
      TransformRT const& bakeTransform,
      Context* context,
      superdex::Error& error) const override;
};

/**
 * @brief Load bot parameters from an abstract source using a custom loader (e.g. SuperDexStudio,
 * Unreal). If the path refers to a @ref ModBotPrefab, the mod bot recipe is evaluated to
 * produce the final @ref BotPrefab. Otherwise, the parameters are loaded directly.
 *
 * @param[in] path File path to the .superdex_bot file.
 * @param[in] loader Loader used to read files and shapes.
 * @param[in] validate If true, the bot will be validated on load. If it is a built-bot it will be
 * validated after each modification stage.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Loaded or built @ref BotPrefab, or default-constructed on failure.
 */
MOCHI_API BotPrefab LoadBotPrefab(
    std::string_view path,
    IBotLoader const& loader,
    bool validate,
    superdex::Error& error);

} // namespace superdex::robotics
