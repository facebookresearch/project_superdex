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

#include "mochi_hdf5.h"

#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/group_rw.h>
#include <mochi_core/utils/hdf5_utils.h>
#include <mochi_core/utils/monadic.h>
#include <mochi_core/utils/profile.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace mochi {

#if MOCHI_USE_HDF5

template <typename T>
static std::optional<RowMatrix<T>>
LoadRowMajorMatrix(H5::Group const& group, std::string_view name, Error& error) {
  // constraints
  constexpr bool is_int = std::is_same_v<T, int>;
  constexpr bool is_real = std::is_same_v<T, real>;
  static_assert(is_int || is_real, "currently only supporting int and real");

  try {
    auto dataSet = group.openDataSet(name.data());
    H5::DataSpace dataSpace = dataSet.getSpace();
    int numDims = dataSpace.getSimpleExtentNdims();
    MOCHI_ERROR_IF(numDims != 2, error, "Dataset has incorrect number of dimensions.");
    MOCHI_ERROR_RETURN(error, {});
    hsize_t dims[2] = {};
    dataSpace.getSimpleExtentDims(dims);
    RowMatrix<T> result(dims[0], dims[1]);
    if constexpr (std::is_same_v<T, int>) {
      dataSet.read(result.Data(), MOCHI_H5T_NATIVE_INT);
    }
    if constexpr (std::is_same_v<T, real>) {
      dataSet.read(result.Data(), MOCHI_H5T_NATIVE_REAL);
    }

    return result;
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }
}

template <typename T, typename ContainerT = DynamicArray<T>>
static std::optional<ContainerT>
LoadArray(H5::Group const& group, std::string_view name, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  try {
    auto basisDataSet = group.openDataSet(name.data());
    H5::DataSpace dataSpace = basisDataSet.getSpace();
    int numDims = dataSpace.getSimpleExtentNdims();
    MOCHI_ERROR_IF(numDims != 1, error, "Array has incorrect number of dimensions.");
    MOCHI_ERROR_RETURN(error, {});

    hsize_t dims[1] = {};
    dataSpace.getSimpleExtentDims(dims);

    ContainerT result(dims[0]);
    if constexpr (std::is_same_v<T, int>) {
      basisDataSet.read(result.data(), MOCHI_H5T_NATIVE_INT);
    } else if constexpr (std::is_same_v<T, real>) {
      basisDataSet.read(result.data(), MOCHI_H5T_NATIVE_REAL);
    } else {
      static_assert(std::is_same_v<T, void>, "Invalid argument for T!");
    }

    return result;
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }
}

template <typename T, size_t N>
static std::optional<DynamicArray<NdArray<T, N>>>
LoadVectorOfNdArray(H5::Group const& group, std::string_view name, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  try {
    auto basisDataSet = group.openDataSet(name.data());
    H5::DataSpace dataSpace = basisDataSet.getSpace();
    int numDims = dataSpace.getSimpleExtentNdims();
    MOCHI_ERROR_IF(numDims != 2, error, "Array has incorrect number of dimensions.");
    MOCHI_ERROR_RETURN(error, {});

    hsize_t dims[2] = {};
    dataSpace.getSimpleExtentDims(dims);
    MOCHI_ERROR_IF(dims[1] != N, error, "Second dimension of array has wrong size.");
    MOCHI_ERROR_RETURN(error, {});

    DynamicArray<NdArray<T, N>> result(dims[0]);
    Span<T> span = Flatten(MakeSpan(result));
    if constexpr (std::is_same_v<T, int>) {
      basisDataSet.read(span.data(), MOCHI_H5T_NATIVE_INT);
    } else if constexpr (std::is_same_v<T, real>) {
      basisDataSet.read(span.data(), MOCHI_H5T_NATIVE_REAL);
    } else {
      static_assert(std::is_same_v<T, void>, "Invalid argument for T!");
    }
    return result;
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }
}

[[nodiscard]] static size_t
GetFixedStringByteCount(H5::StrType const& strType, hsize_t count, Error& error) {
  MOCHI_ERROR_IF(
      strType.isVariableStr(), error, "Variable-length string datasets are not supported.");
  MOCHI_ERROR_RETURN(error, 0);

  size_t const strSize = strType.getSize();
  MOCHI_ERROR_IF(strSize == 0, error, "String dataset has invalid string size.");
  MOCHI_ERROR_RETURN(error, 0);
  MOCHI_ERROR_IF(
      count > std::numeric_limits<size_t>::max() / strSize, error, "String dataset is too large.");
  MOCHI_ERROR_RETURN(error, 0);
  return count * strSize;
}

namespace {

template <typename ResultT, typename UserDataT>
struct LoadDictionaryVisitorData {
  std::unordered_map<std::string, ResultT>& outDictionary;
  UserDataT const& userData;
  Error& error;
};

struct EmptyUserData {};

struct LoadBshUserData {
  ModelData const& model;
};

struct LoadRomUserData {
  Real3 bakeScale{};
  TransformRT bakeTransform{};
};

} // namespace

template <
    typename ResultT,
    typename UserDataT,
    typename H5GroupTypePossiblyConst,
    std::optional<ResultT>(loadFunc)(H5GroupTypePossiblyConst& group, UserDataT data, Error& error)>
static int LoadDictionaryVisitor(
    H5::H5Object& obj,
    H5std_string attrName,
    H5O_info2_t const* oinfo,
    void* operatorData) {
  auto* data = reinterpret_cast<LoadDictionaryVisitorData<ResultT, UserDataT>*>(operatorData);
  MOCHI_ERROR_RETURN(data->error, 0);

  if (attrName == ".") {
    return 0;
  }

  if (oinfo->type == H5O_TYPE_GROUP) {
    auto* group = dynamic_cast<H5::Group*>(&obj);
    MOCHI_ASSERT(group, "Expected an H5::Group based on the type ID");

    auto itemGroup = group->openGroup(attrName);
    auto loadedItem = loadFunc(itemGroup, data->userData, data->error);
    MOCHI_ERROR_RETURN(data->error, 0);

    if (loadedItem) {
      auto fullName = itemGroup.getObjName();
      auto shortName = [&fullName]() -> std::string {
        auto findChar = fullName.rfind('/');
        if (findChar == std::string::npos) {
          return fullName;
        } else {
          return fullName.substr(findChar + 1, std::string::npos);
        }
      }();

      data->outDictionary.emplace(shortName, std::move(*loadedItem));
    }
  }

  return 0;
}

template <
    typename ResultT,
    typename UserDataT,
    typename H5GroupTypePossiblyConst,
    std::optional<ResultT>(loadFunc)(H5GroupTypePossiblyConst& group, UserDataT data, Error& error)>
static std::unordered_map<std::string, ResultT>
LoadDictionary(H5::Group group, UserDataT const& userData, Error& error) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ERROR_RETURN(error, {});
  std::unordered_map<std::string, ResultT> result;
  LoadDictionaryVisitorData<ResultT, UserDataT> data{result, userData, error};
  group.visit(
      H5_INDEX_NAME,
      H5_ITER_NATIVE,
      &LoadDictionaryVisitor<ResultT, UserDataT, H5GroupTypePossiblyConst, loadFunc>,
      &data,
      H5O_INFO_ALL);
  return result;
}

template <
    typename ResultT,
    typename UserDataT,
    std::optional<ResultT>(loadFunc)(H5::Group const& group, UserDataT data, Error& error)>
static std::unordered_map<std::string, ResultT> LoadDictionaryIfExists(
    H5::Group const& file,
    std::string const& name,
    UserDataT const& data,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  if (file.nameExists(name.c_str())) {
    H5::Group group = file.openGroup(name.c_str());
    return LoadDictionary<ResultT, UserDataT, H5::Group const, loadFunc>(group, data, error);
  } else {
    return {};
  }
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

// Helper to lookup the H5::DataType of some supported type T (can't be a constexpr unfortunately)
template <typename T>
static H5::DataType GetH5Type() {
  if constexpr (std::is_same_v<T, float>) {
    return H5::PredType::NATIVE_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return H5::PredType::NATIVE_DOUBLE;
  } else {
    static_assert(std::is_same_v<T, int>, "Data type not yet supported");
    return H5::PredType::NATIVE_INT;
  }
}

// Load a scalar attribute from H5Object
template <typename T>
static void
LoadAttribute(H5::H5Object const& obj, char const* attributeName, T& outData, Error& error) {
  MOCHI_ERROR_RETURN(error);
  try {
    MOCHI_ERROR_IF(!obj.attrExists(attributeName), error, "Missing required H5 attribute");
    MOCHI_ERROR_RETURN(error);
    auto attr = obj.openAttribute(attributeName);
    MOCHI_ERROR_IF(attr.getSpace().getSimpleExtentNdims() != 0, error, "Expected a scalar.");
    MOCHI_ERROR_RETURN(error);
    attr.read(GetH5Type<T>(), &outData);
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

// Load an NdArray attribute from H5Object
template <typename T, size_t D0, size_t... DIMS>
static void LoadAttribute(
    H5::H5Object const& obj,
    char const* attributeName,
    NdArray<T, D0, DIMS...>& outData,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  try {
    MOCHI_ERROR_IF(!obj.attrExists(attributeName), error, "Missing required H5 attribute");
    MOCHI_ERROR_RETURN(error);
    using ResultType = NdArray<T, D0, DIMS...>;
    auto attr = obj.openAttribute(attributeName);
    int numDims = attr.getSpace().getSimpleExtentNdims();
    MOCHI_ERROR_IF(numDims != ResultType::num_dims, error, "Invalid attribute dimensions");
    MOCHI_ERROR_RETURN(error);
    hsize_t hdims[ResultType::num_dims] = {};
    attr.getSpace().getSimpleExtentDims(hdims);
    for (int i = 0; i < numDims; ++i) {
      MOCHI_ERROR_IF((int)hdims[i] != ResultType::dims[i], error, "Invalid attribute dimensions");
    }
    MOCHI_ERROR_RETURN(error);
    attr.read(GetH5Type<T>(), Flatten(outData).data());
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

// Load a string attribute from H5Object
static void LoadAttribute(
    H5::H5Object const& obj,
    char const* attributeName,
    std::string& outData,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  try {
    MOCHI_ERROR_IF(!obj.attrExists(attributeName), error, "Missing required H5 attribute");
    MOCHI_ERROR_RETURN(error);
    auto attr = obj.openAttribute(attributeName);
    MOCHI_ERROR_IF(attr.getSpace().getSimpleExtentNdims() != 0, error, "Expected a scalar.");
    MOCHI_ERROR_RETURN(error);
    H5::StrType stype = attr.getStrType();
    attr.read(stype, outData);
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

static bool LoadRomTransformLayerSettings(H5::Group const& group, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  bool needsRigidTransformLayer = false;

  if (group.attrExists("transform_layer")) {
    std::string transformLayerStr;
    LoadAttribute(group, "transform_layer", transformLayerStr, error);

    if (transformLayerStr == "none") {
      needsRigidTransformLayer = false;
    } else if (transformLayerStr == "pivot") {
      needsRigidTransformLayer = true;
    } else {
      MOCHI_ERROR_SET(error, "Unrecognized transform layer type.");
    }
    MOCHI_ERROR_RETURN(error, {});
  }

  return needsRigidTransformLayer;
}

// Signed scale and rotation decompositions are not unique, so validate their combined linear map.
[[nodiscard]] static bool IsScalarLinearTransform(Real3 const& scale, Quaternion const& rotation) {
  real const magnitude = Max(Abs(scale));
  if (!IsFinite(magnitude) || magnitude == 0_r) {
    return false;
  }

  Matrix3x3r const linearTransform =
      Dot(ToMatrix3x3(Normalize(rotation)), DiagonalMatrix<3>(scale));
  real const scalar = Trace(linearTransform) * (1_r / 3_r);
  return NearEqual(
      linearTransform, DiagonalMatrix<3>(scalar), kDefaultNearEqualEpsilon<real> * magnitude);
}

[[nodiscard]] static std::optional<LinearRomData>
LoadLinearRom(H5::Group const& group, LoadRomUserData const& data, Error& error) {
  MOCHI_ERROR_IF(
      !IsScalarLinearTransform(data.bakeScale, data.bakeTransform.GetRotation()),
      error,
      "Unsupported linear ROM bake transform: combined rotation and scale must reduce to a nonzero uniform scale.");
  MOCHI_ERROR_RETURN(error, {});

  bool const needsRigidTransformLayer = LoadRomTransformLayerSettings(group, error);
  MOCHI_ERROR_RETURN(error, {});

  std::optional<RowMatrix<real>> basis;
  try {
    auto basisDataSet = group.openDataSet("basis");
    H5::DataSpace dataSpace = basisDataSet.getSpace();
    int numDims = dataSpace.getSimpleExtentNdims();
    MOCHI_ERROR_IF(numDims != 2, error, "Basis has incorrect number of dimensions.");
    MOCHI_ERROR_RETURN(error, {});

    hsize_t dims[2] = {};
    dataSpace.getSimpleExtentDims(dims);

    auto tmpBasis = RowMatrix<real>(dims[0], dims[1]);
    basisDataSet.read(tmpBasis.Data(), MOCHI_H5T_NATIVE_REAL);
    basis.emplace(std::move(tmpBasis));
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }

  return LinearRomData{
      .needsRigidTransformLayer = needsRigidTransformLayer, .basis = std::move(*basis)};
}

static ai::MlpLayer<real> CreateMlpLayerDataFromH5Group(H5::Group const& group, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  auto weightRowMajorOpt = LoadRowMajorMatrix<real>(group, "weight", error);
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ASSERT(weightRowMajorOpt);
  Matrix<real> W = weightRowMajorOpt.value();

  auto biasOpt = LoadArray<real, ColumnVector<real>>(group, "bias", error);
  MOCHI_ERROR_RETURN(error, {});
  // Check if activation group exists, default to identity if not
  if (group.nameExists("activation")) {
    auto activationGroup = group.openGroup("activation");
    std::string kind;
    LoadAttribute(activationGroup, "kind", kind, error);
    MOCHI_ERROR_RETURN(error, {});
    MOCHI_ERROR_IF(
        kind != "elu" && kind != "identity",
        error,
        "Only ELU and identity activation are supported.");
    MOCHI_ERROR_RETURN(error, {});
    if (kind == "elu") {
      real alpha = 0;
      LoadAttribute(activationGroup, "alpha", alpha, error);
      MOCHI_ERROR_RETURN(error, {});
      return ai::MlpLayer<real>(
          std::move(W), std::move(biasOpt.value()), ai::ELUActivation<real>{alpha});
    }
  }

  return ai::MlpLayer<real>{
      std::move(W), std::move(biasOpt.value()), ai::IdentityActivation<real>()};
}

struct MlpLayerVisitorData {
  std::map<int, ai::MlpLayer<real>>& outLayers;
  Error& error;
};

static int MlpLayerVistor(
    H5::H5Object& obj,
    H5std_string childName,
    H5O_info2_t const* oinfo,
    void* userData) {
  auto* data = reinterpret_cast<MlpLayerVisitorData*>(userData);
  MOCHI_ERROR_RETURN(data->error, 0);
  if (childName == ".") {
    return 0;
  }

  if (oinfo->type == H5O_TYPE_GROUP) {
    auto group = dynamic_cast<H5::Group*>(&obj)->openGroup(childName);
    std::string name;
    LoadAttribute(group, "name", name, data->error);
    MOCHI_ERROR_IF(name.empty(), data->error, "Empty MLP layer name");
    MOCHI_ERROR_RETURN(data->error, 0);

    if (name == "Linear") {
      int layerIndex = -1;
      LoadAttribute(group, "index", layerIndex, data->error);
      MOCHI_ERROR_IF(layerIndex < 0, data->error, "Invalid MLP layer index");
      MOCHI_ERROR_RETURN(data->error, 0);
      data->outLayers[layerIndex] = CreateMlpLayerDataFromH5Group(group, data->error);
      MOCHI_ERROR_RETURN(data->error, 0);
    }
  }

  return 0;
}

static std::optional<std::tuple<ai::Mlp<real>, ColumnVector<real>>>
LoadCromEncoderModel(H5::Group& group, real scale, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  try {
    //
    // 1. load layers
    //
    std::map<int, ai::MlpLayer<real>> mlpLayers;
    auto visitorData = MlpLayerVisitorData{mlpLayers, error};
    group.visit(H5_INDEX_NAME, H5_ITER_NATIVE, &MlpLayerVistor, &visitorData, H5O_INFO_ALL);
    MOCHI_ERROR_IF(mlpLayers.empty(), error, "No MLP layers found");
    MOCHI_ERROR_RETURN(error, {});

    //
    // 2. load input standardization parameters
    //
    auto inputStandardizeGroup = group.openGroup("standardizeQ");
    auto meanOpt = LoadArray<real, ColumnVector<real>>(inputStandardizeGroup, "mean", error);
    auto stdOpt = LoadArray<real, ColumnVector<real>>(inputStandardizeGroup, "std", error);
    MOCHI_ERROR_RETURN(error, {});
    MOCHI_ASSERT(
        meanOpt.has_value() && stdOpt, "LoadArray should have returned a value or set an error.");
    MOCHI_ERROR_IF(
        meanOpt->size() != stdOpt->size(),
        error,
        "The \"mean\" and \"std\" datasets must be the same size");
    MOCHI_ERROR_RETURN(error, {});

    // combine the mean and std into a single vector (like decoder) and scale
    auto const count = meanOpt->size();
    ColumnVector<real> meanAndStdevValuesForInputStandardize(count * 2);
    meanAndStdevValuesForInputStandardize.TopRows(count) = scale * meanOpt.value();
    meanAndStdevValuesForInputStandardize.BottomRows(count) = scale * stdOpt.value();

    //
    // 3. create vector of mlp layers to accommodate the Mlp constructor
    //
    DynamicArray<ai::MlpLayer<real>> mlpLayersVec;
    mlpLayersVec.reserve(mlpLayers.size());
    for (auto& [k, v] : mlpLayers) {
      mlpLayersVec.emplace_back(v);
    }
    MOCHI_ASSERT(!mlpLayersVec.empty());

    //
    // 4. create Mlp
    //
    ai::Mlp<real> network(std::move(mlpLayersVec));

    return std::make_tuple(network, meanAndStdevValuesForInputStandardize);
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }
}

static std::optional<std::tuple<ai::Mlp<real>, ColumnVector<real>>>
LoadCromDecoderModel(H5::Group& group, real scale, Error& error) {
  MOCHI_ERROR_IF(
      !IsFinite(scale) || NearZero(scale),
      error,
      "CROM decoder scale must be finite and non-zero.");
  MOCHI_ERROR_RETURN(error, {});

  try {
    //
    // 1. load layers
    //
    std::map<int, ai::MlpLayer<real>> mlpLayers;
    auto visitorData = MlpLayerVisitorData{mlpLayers, error};
    group.visit(H5_INDEX_NAME, H5_ITER_NATIVE, &MlpLayerVistor, &visitorData, H5O_INFO_ALL);
    MOCHI_ERROR_IF(mlpLayers.empty(), error, "No MLP layers found.");
    MOCHI_ERROR_RETURN(error, {});

    //
    // 2. load final output standardization
    //
    auto finalScalingGroup = group.openGroup("invStandardizeQ");
    auto meanOpt = LoadArray<real, ColumnVector<real>>(finalScalingGroup, "mean", error);
    auto stdOpt = LoadArray<real, ColumnVector<real>>(finalScalingGroup, "std", error);
    MOCHI_ERROR_RETURN(error, {});
    MOCHI_ASSERT(
        meanOpt.has_value() && stdOpt, "LoadArray should have returned a value or set an error.");
    MOCHI_ERROR_IF(
        meanOpt->size() != stdOpt->size(),
        error,
        "The \"mean\" and \"std\" datasets must be the same size");
    MOCHI_ERROR_RETURN(error, {});

    // combine the mean and std into a single vector
    auto const count = meanOpt.value().size();
    ColumnVector<real> meanAndStdevValuesForInverseStandardizeOutput(count * 2);
    meanAndStdevValuesForInverseStandardizeOutput.TopRows(count) = meanOpt.value();
    meanAndStdevValuesForInverseStandardizeOutput.BottomRows(count) = stdOpt.value();

    //
    // 3. create vector of mlp layers to accommodate the Mlp constructor
    //
    DynamicArray<ai::MlpLayer<real>> mlpLayersVec;
    mlpLayersVec.reserve(mlpLayers.size());
    for (auto& [k, v] : mlpLayers) {
      mlpLayersVec.emplace_back(v);
    }

    //
    // 4. bake the scale of the shape into the network
    //
    MOCHI_ERROR_IF(
        mlpLayersVec.front().WeightsView().Cols() < 3,
        error,
        "CROM decoder's first layer must have at least 3 input columns.");
    MOCHI_ERROR_RETURN(error, {});
    mlpLayersVec.front().WeightsView().RightCols(3) *=
        (1_r / scale); // Last 3 columns of the first layer correspond to node positions.
    meanAndStdevValuesForInverseStandardizeOutput *= scale;

    //
    // 5. create Mlp
    //
    ai::Mlp<real> network(std::move(mlpLayersVec));

    return std::make_tuple(network, meanAndStdevValuesForInverseStandardizeOutput);
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }
}

std::optional<std::tuple<ai::Mlp<real>, ColumnVector<real>>> hdf5::LoadCromDecoderModel(
    std::string_view filePath,
    std::string_view groupName,
    real scale,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  try {
    std::lock_guard lock{hdf5::GetGlobalMutex()};

    H5::H5File file = OpenFileForRead(filePath, error);
    MOCHI_ERROR_RETURN(error, {});

    std::string const groupNameString{groupName};
    MOCHI_ERROR_IF(
        !file.nameExists(groupNameString.c_str()), error, "Group not found for CROM decoder model");
    MOCHI_ERROR_RETURN(error, {});

    H5::Group groupToReadFrom = file.openGroup(groupNameString.c_str());
    return LoadCromDecoderModel(groupToReadFrom, scale, error);
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }
}

static bool IsUniformScale(Real3 const& scale) {
  return (scale[0] == scale[1]) && (scale[1] == scale[2]);
}

static std::optional<NeuralNetCromData>
LoadNeuralNetCrom(H5::Group const& group, LoadRomUserData const& data, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(
      !IsUniformScale(data.bakeScale),
      error,
      "When baking scale into a Neural CROM, that scale must be uniform on all axes.");
  MOCHI_ERROR_IF(
      data.bakeTransform != TransformRT::Identity(),
      error,
      "Cannot bake rotation nor translation into a Neural CROM.");
  MOCHI_ERROR_RETURN(error, {});

  real const scale = data.bakeScale[0];
  bool const needsRigidTransformLayer = LoadRomTransformLayerSettings(group, error);

  int latentStateSize = 0;
  LoadAttribute(group, "latent_state_size", latentStateSize, error);
  MOCHI_ERROR_RETURN(error, {});

  auto latentStateForZeroDisplacement =
      LoadArray<real, ColumnVector<real>>(group, "latent_state_for_zero_displacement", error);
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ASSERT(latentStateForZeroDisplacement);

  // Load decoder (required) and encoder (optional) from hierarchical structure
  MOCHI_ERROR_IF_NOT(
      group.nameExists("decoder"), error, "Neural CROM must contain 'decoder' subgroup");
  MOCHI_ERROR_RETURN(error, {});

  // Load decoder from decoder/ subgroup
  H5::Group decoderGroup = group.openGroup("decoder");
  auto cromDecoderTuple = LoadCromDecoderModel(decoderGroup, scale, error);
  MOCHI_ERROR_RETURN(error, {});
  auto&& [decoderNetwork, meanAndStdevValuesForInverseStandardizeOutput] =
      std::move(*cromDecoderTuple);

  // Load encoder from encoder/ subgroup if it exists (optional)
  std::optional<ai::Mlp<real>> encoderNetwork;
  std::optional<ColumnVector<real>> meanAndStdevValuesForInputStandardize;

  if (group.nameExists("encoder")) {
    H5::Group encoderGroup = group.openGroup("encoder");
    auto cromEncoderTuple = LoadCromEncoderModel(encoderGroup, scale, error);
    MOCHI_ERROR_RETURN(error, {});
    encoderNetwork = std::move(std::get<0>(*cromEncoderTuple));
    meanAndStdevValuesForInputStandardize = std::move(std::get<1>(*cromEncoderTuple));
  }

  return NeuralNetCromData{
      .needsRigidTransformLayer = needsRigidTransformLayer,
      .latentStateSize = latentStateSize,
      .encoder = std::move(encoderNetwork),
      .meanAndStdevForInputStandardize = std::move(meanAndStdevValuesForInputStandardize),
      .decoder = std::move(decoderNetwork),
      .meanAndStdevForOutputInverseStandardize =
          std::move(meanAndStdevValuesForInverseStandardizeOutput),
      .latentStateForZeroDisplacement = std::move(latentStateForZeroDisplacement.value())};
}

static std::optional<RomData> LoadRom(H5::Group const& group, LoadRomUserData data, Error& error) {
  // originally here we had a mochi_error_if the group is missing a "type" attribute
  // but this won't work now because LoadRom uses the visit method of h5 gorups, which
  // is applied recursively to load into a dictionary, but doing this conflicts with the scenarion
  // when a given ROM model has multiple subgroups that define that model, for example the
  // nn_crom. In fact, the nn_crom is currently defined such that a layer is a subgroup. So it
  // seems more appropriate to just load something if it contains "type". Maybe the whole ROM
  // loading needs to be improved.

  if (group.attrExists("type")) {
    std::string typeStr;
    LoadAttribute(group, "type", typeStr, error);

    if (typeStr == "linear" || typeStr == "linear biharmonic") {
      return LoadLinearRom(group, data, error);
    } else if (typeStr == "neural_net_crom") {
      return LoadNeuralNetCrom(group, data, error);
    } else if (typeStr == "polynomial_crom" || typeStr == "gmm") {
      // GMMs are deprecated. Polynomial CROMs are directly generated in Mochi. For backwards
      // compatibility of existing H5 files, it is not an error for either of them to exist in H5.
      return {};
    } else {
      MOCHI_ERROR_SET(error, "Invalid ROM type.");
      return {};
    }
  } else {
    return {};
  }
}

static std::optional<SampleMeshInfo>
LoadSampleMesh(H5::Group const& group, EmptyUserData, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  auto volumeElements = LoadArray<int>(group, "volume_elements", error);
  auto volumeElementWeights = LoadArray<real>(group, "volume_element_weights", error);
  auto boundaryElements = LoadArray<int>(group, "boundary_elements", error);
  auto boundaryElementWeights = LoadArray<real>(group, "boundary_element_weights", error);
  MOCHI_ERROR_RETURN(error, {});

  MOCHI_ERROR_IF(
      volumeElements->size() != volumeElementWeights->size(),
      error,
      "Sample mesh volume element and weight counts must match.");
  MOCHI_ERROR_IF(
      boundaryElements->size() != boundaryElementWeights->size(),
      error,
      "Sample mesh boundary element and weight counts must match.");
  MOCHI_ERROR_IF_NOT(
      IsUnique(MakeConstSpan(*volumeElements)),
      error,
      "Sample mesh volume elements must be unique.");
  MOCHI_ERROR_IF_NOT(
      IsUnique(MakeConstSpan(*boundaryElements)),
      error,
      "Sample mesh boundary elements must be unique.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(MakeConstSpan(*volumeElementWeights)),
      error,
      "Sample mesh volume element weights must be finite.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(MakeConstSpan(*boundaryElementWeights)),
      error,
      "Sample mesh boundary element weights must be finite.");
  MOCHI_ERROR_RETURN(error, {});

  return SampleMeshInfo{
      std::move(*volumeElements),
      std::move(*volumeElementWeights),
      std::move(*boundaryElements),
      std::move(*boundaryElementWeights)};
}

static std::optional<ContactSamplesBsh>
LoadBsh(H5::Group const& group, LoadBshUserData data, Error& error) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ERROR_RETURN(error, {});

  auto children = LoadArray<int>(group, "children", error);
  auto childrenRanges = LoadVectorOfNdArray<int, 2>(group, "children_ranges", error);
  auto ids = LoadArray<int>(group, "ids", error);
  auto roots = LoadArray<int>(group, "roots", error);
  MOCHI_ERROR_RETURN(error, {});

  // Build a TetrahedralMesh to get the boundary indices. This is inefficient since the mesh will be
  // constructed again loading completes.
  TetrahedralMesh tetMesh(
      Unflatten<Real3 const>(MakeConstSpan(data.model.mesh->coordinates)),
      Unflatten<Int4 const>(MakeConstSpan(data.model.mesh->connectivity)));

  // Relabelling capabilities: map BSH face ids to tetrahedral-mesh triangle elements.
  MOCHI_ERROR_IF_NOT(
      group.nameExists("ids_to_triangle_elements"),
      error,
      "BSH is missing the dataset 'ids_to_triangle_elements'. "
      "This dataset is required to map BSH face ids to tetrahedral-mesh triangle elements.");
  MOCHI_ERROR_RETURN(error, {});

  auto idToTriangleElements = LoadVectorOfNdArray<int, 3>(group, "ids_to_triangle_elements", error);
  auto relabelling = Transform(std::move(idToTriangleElements), [&](DynamicArray<Int3> const& in) {
    auto boundaryConnectivity = tetMesh.GetBoundaryFacesConnectivity();
    MOCHI_ERROR_IF(
        isize(in) != isize(boundaryConnectivity),
        error,
        "ids_to_triangle_elements length does not match the number of triangles on this tetrahedral mesh.");

    auto sortInt3 = [](Int3 a) -> Int3 {
      std::sort(a.begin(), a.end());
      return a;
    };

    struct Int3Hash {
      inline size_t operator()(Int3 const& a) const noexcept {
        auto baseHash = std::hash<int>{};
        return baseHash(a[0]) ^ baseHash(a[1]) ^ baseHash(a[2]);
      }
    };

    // Generate inverse map of (face id) -> (tri element)
    // to get the map (tri element) -> (face id).
    std::unordered_map<Int3, int, Int3Hash> meshFaceToId;
    for (int i = 0; i < boundaryConnectivity.size(); ++i) {
      meshFaceToId[sortInt3(boundaryConnectivity[i])] = i;
    }

    // Compose the above map with the map (bsh face id) -> (tri element) to
    // build a map (bsh face id) -> (face id)
    std::unordered_map<int, int> result;
    for (int i = 0; i < in.size(); ++i) {
      if (auto it = meshFaceToId.find(sortInt3(in[i])); it != meshFaceToId.end()) {
        result[i] = it->second;
      } else {
        MOCHI_ERROR_SET(error, "Face found in BSH that is not in tetrahedral mesh.");
      }
    }

    return result;
  });
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ASSERT(
      relabelling.has_value(), "LoadVectorOfNdArray should have returned a value or set an error.");

  // Reindex ids to match the tetrahedral mesh triangle elements.
  for (auto& id : *ids) {
    if (auto it = relabelling->find(id); it != relabelling->end()) {
      id = it->second;
    } else {
      MOCHI_ERROR_SET(error, "Invalid index found in BSH ids while trying to reindex.");
    }
  }
  MOCHI_ERROR_RETURN(error, {});

  return ContactSamplesBsh::LoadFromNumpyArrays(*children, *childrenRanges, *ids, *roots);
}

static void ReadMeshTransformsImpl(
    H5::H5File const& file,
    Span<Quaternion> outRotations,
    Span<Real3> outTranslations,
    Span<int> outLinkParents,
    Span<ArticulatedJointType> outJointTypes,
    Span<ArticulatedCycleJoint> outCycleJoints,
    Span<Real3> outJointAxes,
    Span<Real3> outJointXYZs,
    bool& outHasJointLimits,
    Span<Real3> outJointMinLimits,
    Span<Real3> outJointMaxLimits,
    bool& outHasJointNames,
    Span<std::array<char, hdf5::kMaxMeshTransformsNameLength>> outJointNames,
    bool& outHasLinkNames,
    Span<std::array<char, hdf5::kMaxMeshTransformsNameLength>> outLinkNames,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  try {
    int numLinks = isize(outRotations);
    int numJoints = isize(outJointTypes);
    int numCycles = numJoints - numLinks;
    MOCHI_ERROR_IF(outTranslations.size() != numLinks, error, "Span size inconsistent.");
    MOCHI_ERROR_IF(outLinkParents.size() != numLinks, error, "Span size inconsistent.");
    MOCHI_ERROR_IF(outJointTypes.size() != numJoints, error, "Span size inconsistent.");
    MOCHI_ERROR_IF(outCycleJoints.size() != numCycles, error, "Span size inconsistent.");
    MOCHI_ERROR_IF(outJointAxes.size() != numJoints, error, "Span size inconsistent.");
    MOCHI_ERROR_IF(outJointXYZs.size() != numJoints, error, "Span size inconsistent.");
    MOCHI_ERROR_IF(outJointMinLimits.size() != numJoints, error, "Span size inconsistent.");
    MOCHI_ERROR_IF(outJointMaxLimits.size() != numJoints, error, "Span size inconsistent.");
    MOCHI_ERROR_IF(outLinkNames.size() != numLinks, error, "Span size inconsistent.");
    MOCHI_ERROR_IF(outJointNames.size() != numJoints, error, "Span size inconsistent.");
    MOCHI_ERROR_RETURN(error);

    auto readDataSet = [file](
                           std::string const& name,
                           int dim0,
                           int dim1,
                           H5::PredType type,
                           void* outBuffer,
                           Error& error) {
      MOCHI_ERROR_RETURN(error);
      H5::DataSet dataSet = file.openDataSet(name);
      H5::DataSpace dataSpace = dataSet.getSpace();

      // Get dimensions of dataset
      int const numDims = dataSpace.getSimpleExtentNdims();
      MOCHI_ERROR_IF(numDims != 2, error, "Dataset has incorrect number of dimensions.");
      MOCHI_ERROR_RETURN(error);
      hsize_t dims[2] = {};
      dataSpace.getSimpleExtentDims(dims, nullptr);

      // check the size of the span match the file
      MOCHI_ERROR_IF(dims[0] != dim0, error, "Invalid number of data set elements.");
      MOCHI_ERROR_IF(dims[1] != dim1, error, "Invalid number of dimension for data set.");
      MOCHI_ERROR_RETURN(error);

      dataSet.read(outBuffer, type);
    };

    readDataSet("rotations", numLinks, 4, MOCHI_H5T_NATIVE_REAL, outRotations.data(), error);
    readDataSet("translations", numLinks, 3, MOCHI_H5T_NATIVE_REAL, outTranslations.data(), error);
    readDataSet("joint_xyzs", numJoints, 3, MOCHI_H5T_NATIVE_REAL, outJointXYZs.data(), error);

    // Support both "joint_axiss" and "joint_axes"
    bool hasJointAxesOption1 = file.nameExists("joint_axes");
    bool hasJointAxesOption2 = file.nameExists("joint_axiss");
    MOCHI_ERROR_IF(
        !hasJointAxesOption1 && !hasJointAxesOption2,
        error,
        "Missing both 'joint_axiss' and 'joint_axes' dataset");
    if (hasJointAxesOption1 && hasJointAxesOption2) {
      MOCHI_LOG_WARNING("Both 'joint_axes' and 'joint_axiss' found in HDF5. Using 'joint_axes'.");
    }
    std::string jointAxesName = hasJointAxesOption1 ? "joint_axes" : "joint_axiss";
    readDataSet(jointAxesName, numJoints, 3, MOCHI_H5T_NATIVE_REAL, outJointAxes.data(), error);
    MOCHI_ERROR_RETURN(error);

    {
      // Read and process joint_types dataset
      H5::DataSet jointTypesDataset = file.openDataSet("joint_types");
      H5::DataSpace jointTypesDataspace = jointTypesDataset.getSpace();

      // Get dimensions of joint_types dataset
      int const numDims = jointTypesDataspace.getSimpleExtentNdims();
      MOCHI_ERROR_IF(numDims != 1, error, "Dataset has incorrect number of dimensions.");
      MOCHI_ERROR_RETURN(error);
      hsize_t jointTypesDims[1] = {};
      jointTypesDataspace.getSimpleExtentDims(jointTypesDims, nullptr);

      // Determine the size of each string in the dataset
      H5T_class_t typeClass = jointTypesDataset.getTypeClass();
      MOCHI_ERROR_IF(typeClass != H5T_STRING, error, "Error: joint_types is not a string dataset.");
      MOCHI_ERROR_IF(jointTypesDims[0] != numJoints, error, "Invalid number of joints.");
      MOCHI_ERROR_RETURN(error);

      H5::StrType strType = jointTypesDataset.getStrType();
      size_t const stringByteCount = GetFixedStringByteCount(strType, jointTypesDims[0], error);
      MOCHI_ERROR_RETURN(error);
      size_t const strSize = strType.getSize();
      H5T_str_t const strPadding = strType.getStrpad();
      MOCHI_ERROR_IF(
          strPadding != H5T_STR_NULLTERM && strPadding != H5T_STR_NULLPAD &&
              strPadding != H5T_STR_SPACEPAD,
          error,
          "Unsupported string padding mode in joint_types dataset.");
      MOCHI_ERROR_RETURN(error);

      // Read the strings
      DynamicArray<char> flatJointTypes;
      flatJointTypes.resize_noinit(stringByteCount);
      jointTypesDataset.read(flatJointTypes.data(), strType);

      // Convert to std::string and output
      for (size_t i = 0; i < jointTypesDims[0]; ++i) {
        std::string jointTypeStr(&flatJointTypes[i * strSize], strSize);
        if (strPadding == H5T_STR_NULLTERM || strPadding == H5T_STR_NULLPAD) {
          if (auto const nullPos = jointTypeStr.find('\0'); nullPos != std::string::npos) {
            jointTypeStr.resize(nullPos);
          }
        } else {
          while (!jointTypeStr.empty() && jointTypeStr.back() == ' ') {
            jointTypeStr.pop_back();
          }
        }
        if (jointTypeStr == "free") {
          outJointTypes[i] = ArticulatedJointType::Free;
        } else if (jointTypeStr == "revolute") {
          outJointTypes[i] = ArticulatedJointType::Revolute;
        } else if (jointTypeStr == "hard" || jointTypeStr == "fixed") {
          outJointTypes[i] = ArticulatedJointType::Hard;
        } else if (jointTypeStr == "prismatic") {
          outJointTypes[i] = ArticulatedJointType::Prismatic;
        } else if (jointTypeStr == "spherical") {
          outJointTypes[i] = ArticulatedJointType::Spherical;
        } else if (jointTypeStr == "cycle") {
          outJointTypes[i] = ArticulatedJointType::Cycle;
        } else {
          MOCHI_ERROR_SET(error, "Unknown joint type");
          MOCHI_ERROR_RETURN(error);
        }
      }
    }

    // Read cycle joints if they exist
    if (numCycles > 0) {
      DynamicArray<int> cycleJointsVec;
      cycleJointsVec.resize_noinit(numCycles * 2);
      readDataSet(
          "cycle_joints", numCycles, 2, H5::PredType::NATIVE_INT, cycleJointsVec.data(), error);
      MOCHI_ERROR_RETURN(error);
      for (int i = 0; i < numCycles; ++i) {
        outCycleJoints[i].child = cycleJointsVec[i * 2];
        outCycleJoints[i].parent = cycleJointsVec[i * 2 + 1];
      }
    }

    // Open the joint_parents dataset (which corresponds to link parents; bad naming)
    // Support both "joint_parents" and "link_parents"
    bool hasLinkParentsOption1 = file.nameExists("link_parents");
    bool hasLinkParentsOption2 = file.nameExists("joint_parents");
    if (hasLinkParentsOption1 && hasLinkParentsOption2) {
      MOCHI_LOG_WARNING(
          "Both 'link_parents' and 'joint_parents' found in HDF5. Using 'link_parents'.");
    }
    std::string linkParentsName = hasLinkParentsOption1 ? "link_parents" : "joint_parents";
    MOCHI_ERROR_IF(
        !hasLinkParentsOption1 && !hasLinkParentsOption2, error, "Missing 'link_parents' dataset");
    MOCHI_ERROR_RETURN(error);
    H5::DataSet linkParentsDataset = file.openDataSet(linkParentsName);

    // Get dimensions of joint_parents dataset
    H5::DataSpace linkParentsDataspace = linkParentsDataset.getSpace();
    int const numDims = linkParentsDataspace.getSimpleExtentNdims();
    MOCHI_ERROR_IF(numDims != 1, error, "Dataset has incorrect number of dimensions.");
    MOCHI_ERROR_RETURN(error);
    hsize_t linkParentsDims[1] = {};
    linkParentsDataspace.getSimpleExtentDims(linkParentsDims, nullptr);

    MOCHI_ERROR_IF(linkParentsDims[0] != numLinks, error, "Invalid number of bodies.");
    MOCHI_ERROR_RETURN(error);

    // Read the joint_parents dataset into a vector
    linkParentsDataset.read(outLinkParents.data(), H5::PredType::NATIVE_INT);

    // See if we have joint_limits to read
    outHasJointLimits = file.nameExists("joint_lowers") && file.nameExists("joint_uppers");
    if (outHasJointLimits) {
      readDataSet(
          "joint_lowers", numJoints, 3, MOCHI_H5T_NATIVE_REAL, outJointMinLimits.data(), error);
      readDataSet(
          "joint_uppers", numJoints, 3, MOCHI_H5T_NATIVE_REAL, outJointMaxLimits.data(), error);
      MOCHI_ERROR_RETURN(error);
    }

    auto readStringDataSet =
        [&file, &error](
            std::string const& name,
            int num,
            Span<std::array<char, hdf5::kMaxMeshTransformsNameLength>> outStringsFixed) {
          H5::DataSet dataSet = file.openDataSet(name);
          H5::DataSpace dataSpace = dataSet.getSpace();

          // Get dimensions of the dataset
          int const numDims = dataSpace.getSimpleExtentNdims();
          MOCHI_ERROR_IF(numDims != 1, error, "Dataset has incorrect number of dimensions.");
          MOCHI_ERROR_RETURN(error);
          hsize_t dims[1] = {};
          dataSpace.getSimpleExtentDims(dims, nullptr);

          // Ensure the dataset is a string type
          H5T_class_t typeClass = dataSet.getTypeClass();
          MOCHI_ERROR_IF(typeClass != H5T_STRING, error, "Error: Dataset is not a string dataset.");
          MOCHI_ERROR_IF(dims[0] != num, error, "Invalid number of elements.");
          MOCHI_ERROR_RETURN(error);

          H5::StrType strType = dataSet.getStrType();
          size_t const stringByteCount = GetFixedStringByteCount(strType, dims[0], error);
          MOCHI_ERROR_RETURN(error);
          size_t const strSize = strType.getSize();
          H5T_str_t const strPadding = strType.getStrpad();
          MOCHI_ERROR_IF(
              strPadding != H5T_STR_NULLTERM && strPadding != H5T_STR_NULLPAD &&
                  strPadding != H5T_STR_SPACEPAD,
              error,
              "Unsupported string padding mode in dataset.");
          MOCHI_ERROR_RETURN(error);

          // Read the strings
          DynamicArray<char> flatStrings(stringByteCount);
          dataSet.read(flatStrings.data(), strType);

          DynamicArray<std::string> strings(outStringsFixed.size());
          // Convert to std::string and output
          for (size_t i = 0; i < dims[0]; ++i) {
            strings[i] = std::string(&flatStrings[i * strSize], strSize);
            if (strPadding == H5T_STR_NULLTERM || strPadding == H5T_STR_NULLPAD) {
              if (auto const nullPos = strings[i].find('\0'); nullPos != std::string::npos) {
                strings[i].resize(nullPos);
              }
            } else {
              while (!strings[i].empty() && strings[i].back() == ' ') {
                strings[i].pop_back();
              }
            }
          }
          // Copy to output
          for (size_t i = 0; i < strings.size(); ++i) {
            outStringsFixed[i].fill('\0');
            std::copy_n(
                strings[i].begin(),
                std::min(hdf5::kMaxMeshTransformsNameLength - 1, strings[i].length()),
                outStringsFixed[i].data());
          }
        };

    // Read joint_names dataset if it exists
    outHasJointNames = file.nameExists("joint_names");
    if (outHasJointNames) {
      readStringDataSet("joint_names", numJoints, outJointNames);
    }

    // Read link_names dataset if it exists
    outHasLinkNames = file.nameExists("link_names");
    if (outHasLinkNames) {
      readStringDataSet("link_names", numLinks, outLinkNames);
    }

  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

void hdf5::ReadMeshTransformsBytes(
    Span<char const> fileData,
    Span<Quaternion> outRotations,
    Span<Real3> outTranslations,
    Span<int> outLinkParents,
    Span<ArticulatedJointType> outJointTypes,
    Span<ArticulatedCycleJoint> outCycleJoints,
    Span<Real3> outJointAxes,
    Span<Real3> outJointXYZs,
    bool& outHasJointLimits,
    Span<Real3> outJointMinLimits,
    Span<Real3> outJointMaxLimits,
    bool& outHasJointNames,
    Span<std::array<char, kMaxMeshTransformsNameLength>> outJointNames,
    bool& outHasLinkNames,
    Span<std::array<char, kMaxMeshTransformsNameLength>> outLinkNames,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  auto file = hdf5::OpenFileBytesForRead(fileData, error);
  MOCHI_ERROR_RETURN(error);

  return ReadMeshTransformsImpl(
      file,
      outRotations,
      outTranslations,
      outLinkParents,
      outJointTypes,
      outCycleJoints,
      outJointAxes,
      outJointXYZs,
      outHasJointLimits,
      outJointMinLimits,
      outJointMaxLimits,
      outHasJointNames,
      outJointNames,
      outHasLinkNames,
      outLinkNames,
      error);
}

int hdf5::ReadMeshTransformsBytesBodyCount(Span<char const> fileData, Error& error) {
  MOCHI_ERROR_RETURN(error, 0);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  auto file = hdf5::OpenFileBytesForRead(fileData, error);
  MOCHI_ERROR_RETURN(error, 0);

  try {
    // Support both "joint_parents" and "link_parents"
    bool hasLinkParentsOption1 = file.nameExists("link_parents");
    bool hasLinkParentsOption2 = file.nameExists("joint_parents");
    if (hasLinkParentsOption1 && hasLinkParentsOption2) {
      MOCHI_LOG_WARNING(
          "Both 'link_parents' and 'joint_parents' found in HDF5. Using 'link_parents'.");
    }
    std::string linkParentsName = hasLinkParentsOption1 ? "link_parents" : "joint_parents";
    MOCHI_ERROR_IF(
        !hasLinkParentsOption1 && !hasLinkParentsOption2, error, "Missing 'link_parents' dataset");
    MOCHI_ERROR_RETURN(error, 0);
    H5::DataSet jointParentsDataset = file.openDataSet(linkParentsName);
    H5::DataSpace jointParentsDataspace = jointParentsDataset.getSpace();
    int const numDims = jointParentsDataspace.getSimpleExtentNdims();
    MOCHI_ERROR_IF(numDims != 1, error, "Dataset has incorrect number of dimensions.");
    MOCHI_ERROR_RETURN(error, 0);
    hsize_t jointParentsDims[1] = {};
    jointParentsDataspace.getSimpleExtentDims(jointParentsDims, nullptr);
    return (int)jointParentsDims[0];
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return 0;
  }
}

int hdf5::ReadMeshTransformsBytesJointCount(Span<char const> fileData, Error& error) {
  MOCHI_ERROR_RETURN(error, 0);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  auto file = hdf5::OpenFileBytesForRead(fileData, error);
  MOCHI_ERROR_RETURN(error, 0);

  try {
    // Support both "joint_axiss" and "joint_axes"
    bool hasJointAxesOption1 = file.nameExists("joint_axes");
    bool hasJointAxesOption2 = file.nameExists("joint_axiss");
    if (hasJointAxesOption1 && hasJointAxesOption2) {
      MOCHI_LOG_WARNING("Both 'joint_axes' and 'joint_axiss' found in HDF5. Using 'joint_axes'.");
    }
    std::string jointAxesName = hasJointAxesOption1 ? "joint_axes" : "joint_axiss";
    MOCHI_ERROR_IF(
        !hasJointAxesOption1 && !hasJointAxesOption2, error, "Missing 'joint_axes' dataset");
    MOCHI_ERROR_RETURN(error, 0);
    H5::DataSet jointAxesDataset = file.openDataSet(jointAxesName);
    H5::DataSpace jointAxesDataspace = jointAxesDataset.getSpace();
    int const numDims = jointAxesDataspace.getSimpleExtentNdims();
    MOCHI_ERROR_IF(numDims != 2, error, "Dataset has incorrect number of dimensions.");
    MOCHI_ERROR_RETURN(error, 0);
    hsize_t jointAxesDims[2] = {};
    jointAxesDataspace.getSimpleExtentDims(jointAxesDims, nullptr);
    return (int)jointAxesDims[0];
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return 0;
  }
}

static ExperimentalModelData LoadExperimentalModelData(
    H5::Group const& group,
    ModelData const& model,
    Real3 const& bakeScale,
    TransformRT const& bakeTransform,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ExperimentalModelData outData;

  // Load any ROM-related data (non-const group is because the nn_crom needs it)
  outData.romData = LoadDictionaryIfExists<RomData, LoadRomUserData, LoadRom>(
      group, "rom", {bakeScale, bakeTransform}, error);

  // Load any sample mesh data
  outData.sampleMeshes = LoadDictionaryIfExists<SampleMeshInfo, EmptyUserData, LoadSampleMesh>(
      group, "sample_meshes", {}, error);

  if (model.mesh && (model.mesh->nodesPerElement == 4)) {
    outData.bshs = LoadDictionaryIfExists<ContactSamplesBsh, LoadBshUserData, LoadBsh>(
        group, "bsh", {model}, error);
  }

  return outData;
}

ExperimentalModelData mochi::hdf5::LoadExperimentalModelDataFromFile(
    std::string_view filePath,
    ModelData const& model,
    Real3 const& bakeScale,
    TransformRT const& bakeTransform,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::lock_guard lock{hdf5::GetGlobalMutex()};
  auto file = hdf5::OpenFileForRead(filePath, error);
  return LoadExperimentalModelData(file, model, bakeScale, bakeTransform, error);
}

ExperimentalModelData mochi::hdf5::LoadExperimentalModelDataFromBytes(
    Span<char const> fileData,
    ModelData const& model,
    Real3 const& bakeScale,
    TransformRT const& bakeTransform,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::lock_guard lock{hdf5::GetGlobalMutex()};
  auto file = hdf5::OpenFileBytesForRead(fileData, error);
  return LoadExperimentalModelData(file, model, bakeScale, bakeTransform, error);
}

#else // if !MOCHI_USE_HDF5

/*********************************************************************************************
  Stubs Implementations in case !MOCHI_USE_HDF5
*/

void hdf5::ReadMeshTransformsBytes(
    Span<char const> /*fileData*/,
    Span<Quaternion> /*outRotations*/,
    Span<Real3> /*outTranslations*/,
    Span<int> /*outLinkParents*/,
    Span<ArticulatedJointType> /*outJointTypes*/,
    Span<ArticulatedCycleJoint> /*outCycleJoints*/,
    Span<Real3> /*outJointAxes*/,
    Span<Real3> /*outJointXYZs*/,
    bool& /*outHasJointLimits*/,
    Span<Real3> /*outJointMinLimits*/,
    Span<Real3> /*outJointMaxLimits*/,
    bool& /*outHasJointNames*/,
    Span<std::array<char, kMaxMeshTransformsNameLength>> /*outJointNames*/,
    bool& /*outHasLinkNames*/,
    Span<std::array<char, kMaxMeshTransformsNameLength>> /*outLinkNames*/,
    Error& error) {
  MOCHI_ERROR_SET(
      error,
      "Mochi was not built with HDF5 support. To use this file format, you will need to update "
      "your build system to link the HDF5 library and define MOCHI_USE_HDF5 to 1.");
}

int hdf5::ReadMeshTransformsBytesBodyCount(Span<char const> /* fileData */, Error& error) {
  MOCHI_ERROR_SET(
      error,
      "Mochi was not built with HDF5 support. To use this file format, you will need to update "
      "your build system to link the HDF5 library and define MOCHI_USE_HDF5 to 1.");
  return 0;
}

int hdf5::ReadMeshTransformsBytesJointCount(Span<char const> /* fileData */, Error& error) {
  MOCHI_ERROR_SET(
      error,
      "Mochi was not built with HDF5 support. To use this file format, you will need to update "
      "your build system to link the HDF5 library and define MOCHI_USE_HDF5 to 1.");
  return 0;
}

std::optional<std::tuple<ai::Mlp<real>, ColumnVector<real>>> hdf5::LoadCromDecoderModel(
    std::string_view /*filePath*/,
    std::string_view /*groupName*/,
    real /*scale*/,
    Error& error) {
  MOCHI_ERROR_SET(
      error,
      "Mochi was not built with HDF5 support. To use this file format, you will need to update "
      "your build system to link the HDF5 library and define MOCHI_USE_HDF5 to 1.");
  return std::nullopt;
}

#endif // !MOCHI_USE_HDF5

} // namespace mochi
