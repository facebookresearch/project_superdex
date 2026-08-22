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

#include <mochi_core/linear_algebra/matrix.h>

using namespace mochi;

void mochi::details::InitMatrixTypeInfo(
    size_t size,
    size_t alignment,
    SReflect::TypeInfo const& innerTypeInfo,
    int rowsAtCompileTime,
    int colsAtCompileTime,
    mochi::krylov::Direction majorDirection,
    mochi::krylov::Ownership ownership,
    int leadingDimAtCompileTime,
    SReflect::MatrixTypeInfo* outTypeInfo) {
  // Get the names of the enum values
  char const* majorDirStr = SReflect::EnumToString(majorDirection);
  char const* ownershipStr = SReflect::EnumToString(ownership);
  MOCHI_ASSERT(majorDirStr, "Unknown krylov::Direction enum value");
  MOCHI_ASSERT(ownershipStr, "Unknown krylov::Ownership enum value");

  // Format the matrix type name
  char name[256];
  snprintf(
      name,
      sizeof(name),
      "Matrix<%s,%d,%d,Direction::%s,Ownership::%s,%d>",
      innerTypeInfo._name,
      rowsAtCompileTime,
      colsAtCompileTime,
      majorDirStr,
      ownershipStr,
      leadingDimAtCompileTime);

  // Format the name again, this time with namespaces
  char nameWithNamespace[256];
  snprintf(
      nameWithNamespace,
      sizeof(nameWithNamespace),
      "mochi::Matrix<%s,%d,%d,mochi::krylov::Direction::%s,mochi::krylov::Ownership::%s,%d>",
      innerTypeInfo._name,
      rowsAtCompileTime,
      colsAtCompileTime,
      majorDirStr,
      ownershipStr,
      leadingDimAtCompileTime);

  // TypeInfo fields
  outTypeInfo->_coreType = SReflect::CoreType::CT_matrix;
  outTypeInfo->_alignment = alignment;
  outTypeInfo->_sizeInBytes = size;
  outTypeInfo->_name = SReflect::detail::MakeTypeName(name);
  outTypeInfo->_nameWithNamespace = SReflect::detail::MakeTypeName(nameWithNamespace);
  outTypeInfo->_typeId = SReflect::ComputeTypeId(outTypeInfo->_nameWithNamespace);

  // MatrixTypeInfo fields
  outTypeInfo->_innerTypeInfo = &innerTypeInfo;
  outTypeInfo->_isRowMajor = (majorDirection == krylov::Direction::RowMajor);
  outTypeInfo->_isNumRowsDynamic = rowsAtCompileTime < 0;
  outTypeInfo->_isNumColumnsDynamic = colsAtCompileTime < 0;
  outTypeInfo->_fixedNumRows =
      outTypeInfo->_isNumRowsDynamic ? 0 : static_cast<size_t>(rowsAtCompileTime);
  outTypeInfo->_fixedNumColumns =
      outTypeInfo->_isNumColumnsDynamic ? 0 : static_cast<size_t>(colsAtCompileTime);
}
