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

#include "mesh_cli_geometry.h"

#if MOCHI_USE_OCCT

#include "step_face_colors.h"

#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepGProp.hxx>
#include <BRepGProp_Face.hxx>
#include <BRepTools.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <CDM_CanCloseStatus.hxx>
#include <GProp_GProps.hxx>
#include <Geom_Surface.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Interface_Static.hxx>
#include <Message_ProgressRange.hxx>
#include <Precision.hxx>
#include <Quantity_Color.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <STEPControl_Controller.hxx>
#include <ShapeAnalysis_Surface.hxx>
#include <Standard_Failure.hxx>
#include <TDF_LabelSequence.hxx>
#include <TopAbs.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Vertex.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <XCAFDoc_VisMaterial.hxx>
#include <XCAFPrs.hxx>
#include <XCAFPrs_DocumentExplorer.hxx>
#include <XCAFPrs_IndexedDataMapOfShapeStyle.hxx>
#include <XCAFPrs_Style.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Vec.hxx>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>

namespace mochi::mesh::cli::occ {

namespace {

// Floor on how far [mm] a face may sit from the source face it is matched to. Covers the drift a
// fuzzy fuse or a STEP round-trip introduces, yet stays well below any meaningful feature size.
constexpr double kMinMatchTolerance = 1e-2;
// Multiple of a face's own tolerance also accepted as a match distance.
constexpr double kMatchToleranceScale = 10.0;
// Samples per parametric direction when a face's UV-box center falls outside the face.
constexpr int kInteriorSampleGrid = 6;

uint32_t PackRgb(Quantity_Color const& color) {
  double channels[3] = {};
  color.Values(channels[0], channels[1], channels[2], Quantity_TOC_sRGB);
  uint32_t packed = 0;
  for (double const channel : channels) {
    long const value = std::clamp<long>(std::lround(channel * 255.0), 0, 255);
    packed = (packed << 8) | static_cast<uint32_t>(value);
  }
  return packed;
}

// Same precedence as RWMesh_ShapeIterator::initShape, so colors agree with the exported glTF.
std::optional<uint32_t> ResolveRgb(XCAFPrs_Style const& style) {
  if (!style.Material().IsNull()) {
    return PackRgb(style.Material()->BaseColor().GetRGB());
  }
  if (style.IsSetColorSurf()) {
    return PackRgb(style.GetColorSurf());
  }
  if (style.IsSetColorCurv()) {
    return PackRgb(style.GetColorCurv());
  }
  return std::nullopt;
}

// Fills whatever @p style leaves unset from @p inherited, as RWMesh_ShapeIterator::dispatchStyles.
XCAFPrs_Style Inherit(XCAFPrs_Style style, XCAFPrs_Style const& inherited) {
  if (!style.IsSetColorCurv() && inherited.IsSetColorCurv()) {
    style.SetColorCurv(inherited.GetColorCurv());
  }
  if (!style.IsSetColorSurf() && inherited.IsSetColorSurf()) {
    style.SetColorSurf(inherited.GetColorSurfRGBA());
  }
  if (style.Material().IsNull() && !inherited.Material().IsNull()) {
    style.SetMaterial(inherited.Material());
  }
  return style;
}

// A point strictly inside @p face and the face's outward normal there (orientation applied).
struct FaceSample {
  gp_Pnt point;
  gp_Vec normal;
};

gp_Vec OutwardNormal(TopoDS_Face const& face, double u, double v) {
  BRepGProp_Face const surface(face);
  gp_Pnt point;
  gp_Vec normal;
  surface.Normal(u, v, point, normal);
  return normal;
}

// Outward normal of @p face at its closest parametric point to @p point.
gp_Vec OutwardNormalNear(TopoDS_Face const& face, gp_Pnt const& point) {
  ShapeAnalysis_Surface analysis(BRep_Tool::Surface(face));
  gp_Pnt2d const uv = analysis.ValueOfUV(point, Precision::Confusion());
  return OutwardNormal(face, uv.X(), uv.Y());
}

// Samples the UV-box center when it lies on the face, else the first interior point of a grid.
std::optional<FaceSample> SampleFaceInterior(TopoDS_Face const& face) {
  double uMin = 0.0;
  double uMax = 0.0;
  double vMin = 0.0;
  double vMax = 0.0;
  BRepTools::UVBounds(face, uMin, uMax, vMin, vMax);
  double const tolerance = BRep_Tool::Tolerance(face);
  BRepAdaptor_Surface const surface(face);

  auto sampleAt = [&](double u, double v) -> std::optional<FaceSample> {
    BRepClass_FaceClassifier const classifier(face, gp_Pnt2d(u, v), tolerance);
    if (classifier.State() != TopAbs_IN) {
      return std::nullopt;
    }
    return FaceSample{surface.Value(u, v), OutwardNormal(face, u, v)};
  };

  if (std::optional<FaceSample> const center = sampleAt(0.5 * (uMin + uMax), 0.5 * (vMin + vMax))) {
    return center;
  }
  for (int i = 1; i < kInteriorSampleGrid; ++i) {
    for (int j = 1; j < kInteriorSampleGrid; ++j) {
      double const u = uMin + (uMax - uMin) * i / kInteriorSampleGrid;
      double const v = vMin + (vMax - vMin) * j / kInteriorSampleGrid;
      if (std::optional<FaceSample> const sample = sampleAt(u, v)) {
        return sample;
      }
    }
  }
  return std::nullopt;
}

} // namespace

bool LoadStepDocument(
    std::string const& path,
    Handle(TDocStd_Document) & outDocument,
    CliError& error) {
  // Statics are unregistered, and silently ignored, until the controller is initialized.
  STEPControl_Controller::Init();
  Interface_Static::SetCVal("xstep.cascade.unit", "MM");
  // Fix for SolidWorks' single-part export with a non-identity coordinate transform. This flag is
  // benign in most cases.
  Interface_Static::SetCVal("read.step.root.transformation", "OFF");

  Handle(XCAFApp_Application) const application = XCAFApp_Application::GetApplication();
  application->NewDocument("BinXCAF", outDocument);
  MOCHI_MESH_CLI_ERROR_IF(
      outDocument.IsNull(), error, "Failed to create an OpenCascade XCAF document.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);

  STEPCAFControl_Reader reader;
  reader.SetColorMode(true);
  reader.SetMatMode(true);
  reader.SetLayerMode(true);
  reader.SetNameMode(true);
  reader.SetPropsMode(true);

  IFSelect_ReturnStatus status = IFSelect_RetFail;
  try {
    status = reader.ReadFile(path.c_str());
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "OpenCascade threw while reading the STEP file.");
    return false;
  }
  MOCHI_MESH_CLI_ERROR_IF(status != IFSelect_RetDone, error, "Failed to read STEP file.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);

  if (!reader.Transfer(outDocument, Message_ProgressRange())) {
    MOCHI_MESH_CLI_ERROR_SET(error, "Failed to transfer the STEP file into an XCAF document.");
    return false;
  }

  TDF_LabelSequence freeShapes;
  XCAFDoc_DocumentTool::ShapeTool(outDocument->Main())->GetFreeShapes(freeShapes);
  MOCHI_MESH_CLI_ERROR_IF(freeShapes.Length() == 0, error, "STEP file contains no shapes.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);
  return true;
}

ScopedStepDocument::~ScopedStepDocument() {
  if (!handle.IsNull() && handle->CanClose() == CDM_CCS_OK) {
    XCAFApp_Application::GetApplication()->Close(handle);
  }
}

std::vector<StyledFace> CollectStyledFaces(Handle(TDocStd_Document) const& document) {
  std::vector<StyledFace> faces;
  for (XCAFPrs_DocumentExplorer explorer(document, XCAFPrs_DocumentExplorerFlags_OnlyLeafNodes);
       explorer.More();
       explorer.Next()) {
    XCAFPrs_DocumentNode const& node = explorer.Current();
    TopoDS_Shape part;
    if (!XCAFDoc_ShapeTool::GetShape(node.RefLabel, part) || part.IsNull()) {
      continue;
    }
    part.Location(node.Location, false);

    // Bind each face to its most specific style: its own, then its shell's, solid's, compound's.
    XCAFPrs_IndexedDataMapOfShapeStyle styles;
    XCAFPrs::CollectStyleSettings(node.RefLabel, TopLoc_Location(), styles);
    NCollection_DataMap<TopoDS_Shape, XCAFPrs_Style, TopTools_ShapeMapHasher> faceStyles;
    for (int type = TopAbs_FACE; type >= TopAbs_COMPOUND; --type) {
      for (XCAFPrs_IndexedDataMapOfShapeStyle::Iterator it(styles); it.More(); it.Next()) {
        if (it.Key().ShapeType() != type) {
          continue;
        }
        XCAFPrs_Style const style = Inherit(it.Value(), node.Style);
        for (TopExp_Explorer faceIt(it.Key().Located(node.Location), TopAbs_FACE); faceIt.More();
             faceIt.Next()) {
          if (!faceStyles.IsBound(faceIt.Current())) {
            faceStyles.Bind(faceIt.Current(), style);
          }
        }
      }
    }

    for (TopExp_Explorer faceIt(part, TopAbs_FACE); faceIt.More(); faceIt.Next()) {
      XCAFPrs_Style const* const style = faceStyles.Seek(faceIt.Current());
      faces.push_back(
          {TopoDS::Face(faceIt.Current()), ResolveRgb(style != nullptr ? *style : node.Style)});
    }
  }
  return faces;
}

FaceColorMap MatchFaceColors(TopoDS_Shape const& shape, std::span<StyledFace const> sources) {
  std::vector<Bnd_Box> sourceBoxes(sources.size());
  for (size_t i = 0; i < sources.size(); ++i) {
    BRepBndLib::Add(sources[i].face, sourceBoxes[i]);
  }

  FaceColorMap colors;
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(shape, TopAbs_FACE, faces);
  for (int fi = 1; fi <= faces.Extent(); ++fi) {
    TopoDS_Face const& face = TopoDS::Face(faces(fi));
    std::optional<FaceSample> const sample = SampleFaceInterior(face);
    if (!sample) {
      continue;
    }
    double const tolerance =
        std::max(kMinMatchTolerance, kMatchToleranceScale * BRep_Tool::Tolerance(face));
    TopoDS_Vertex const probe = BRepBuilderAPI_MakeVertex(sample->point);

    // Matches faces within tolerance while rejecting those adjacent and facing opposite directions
    // We do this instead of matching exactly because we can combine solids bodies
    // which break the relationship between faces (and even remove some entirely)
    double bestDistance = tolerance;
    StyledFace const* best = nullptr;
    for (size_t i = 0; i < sources.size(); ++i) {
      Bnd_Box box = sourceBoxes[i];
      box.Enlarge(tolerance);
      if (box.IsOut(sample->point)) {
        continue;
      }
      BRepExtrema_DistShapeShape const distance(probe, sources[i].face);
      if (!distance.IsDone() || distance.Value() >= bestDistance) {
        continue;
      }
      if (sample->normal.Dot(OutwardNormalNear(sources[i].face, sample->point)) < 0.0) {
        continue;
      }
      bestDistance = distance.Value();
      best = &sources[i];
    }
    if (best != nullptr && best->rgb) {
      colors.Bind(face, *best->rgb);
    }
  }
  return colors;
}

} // namespace mochi::mesh::cli::occ

bool mochi::mesh::cli::ReadStepFaceColors(
    std::string_view stepFilePath,
    std::vector<StepFaceColorInfo>& outColors,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);
  MOCHI_MESH_CLI_ERROR_IF(stepFilePath.empty(), error, "STEP file path is empty.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);

  occ::ScopedStepDocument scopedDocument;
  auto& document = scopedDocument.handle;
  if (!occ::LoadStepDocument(std::string(stepFilePath), document, error)) {
    return false;
  }

  try {
    std::map<uint32_t, StepFaceColorInfo> byColor;
    for (occ::StyledFace const& styled : occ::CollectStyledFaces(document)) {
      if (!styled.rgb) {
        continue;
      }
      GProp_GProps properties;
      BRepGProp::SurfaceProperties(styled.face, properties);
      StepFaceColorInfo& info = byColor[*styled.rgb];
      info.rgb = *styled.rgb;
      ++info.faceCount;
      info.area += properties.Mass();
    }
    outColors.clear();
    for (auto const& [rgb, info] : byColor) {
      outColors.push_back(info);
    }
    std::ranges::sort(outColors, [](auto const& a, auto const& b) { return a.area > b.area; });
  } catch (Standard_Failure const& failure) {
    char const* const message = failure.GetMessageString();
    std::fprintf(
        stderr, "[step_face_colors] OpenCascade failure: %s\n", message ? message : "(none)");
    MOCHI_MESH_CLI_ERROR_SET(error, "OpenCascade failed while reading the STEP face colors.");
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "OpenCascade threw while reading the STEP face colors.");
  }

  MOCHI_MESH_CLI_ERROR_RETURN(error, false);
  return true;
}

#else // !MOCHI_USE_OCCT

bool mochi::mesh::cli::ReadStepFaceColors(
    std::string_view /*stepFilePath*/,
    std::vector<StepFaceColorInfo>& /*outColors*/,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);
  MOCHI_MESH_CLI_ERROR_SET(
      error,
      "Reading STEP face colors is not supported on this platform (OpenCascade unavailable).");
  return false;
}

#endif // MOCHI_USE_OCCT
