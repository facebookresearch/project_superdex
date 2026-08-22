/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;

namespace Meshing
{
    /// <summary>Output file format</summary>
    public enum MeshExportFormat : uint
    {
        /// <summary>Binary glTF. Carries per-body materials and normals.</summary>
        Glb = 0,

        /// <summary>Text glTF. Same content as <see cref="Glb"/>.</summary>
        Gltf = 1,

        /// <summary>Wavefront OBJ plus a companion .mtl.</summary>
        Obj = 2,

        /// <summary>Binary STL. Geometry only, no materials, no normals.</summary>
        Stl = 3,
    }

    /// <summary>Face tessellation backend</summary>
    public enum MeshingBackend : uint
    {
        /// <summary>
        /// OpenCascade's fast BRepMesh which aims for visual fidelity.
        /// Remeshing is highly recommended for physics simulation.
        /// </summary>
        Delabella = 0,

        /// <summary>
        ///  Uses Constrained Delaunay Triangulation based on uniformly sampled points along CAD faces.
        ///  Provides highest quality for simple CAD geometry. Prioritizes uniform edge length.
        /// </summary>
        Isotropic = 1,
    }

    /// <summary>Edge sampling strategy. Mirrors StepMeshBodyParams::EdgeSampling.</summary>
    public enum EdgeSampling : uint
    {
        /// <summary>Sample count taken from arc length alone.</summary>
        Uniform = 0,

        /// <summary>
        ///  Similar to Uniform but respects angular deflection
        ///  at the cost of shorter edge lengths along curves.
        /// </summary>
        Adaptive = 1,
    }

    /// <summary>Outcome of one requested output. Mirrors VisualExportStatus.</summary>
    public enum MeshExportStatus : uint
    {
        /// <summary>File written; every face tessellated.</summary>
        Written = 0,

        /// <summary>File written, but faces that failed to mesh were skipped.</summary>
        WrittenPartial = 1,

        /// <summary>File not written.</summary>
        Failed = 2,
    }

    /// <summary>One output file to produce from the STEP.</summary>
    public struct MeshExportOutput
    {
        public MeshExportFormat Format;
        public string Path;

        public MeshExportOutput(MeshExportFormat format, string path)
        {
            Format = format;
            Path = path;
        }
    }

    /// <summary>
    /// Tessellation and output settings shared by every output in one export.
    /// Mirrors StepVisualExportParams; the defaults match it.
    /// </summary>
    public class MeshExportOptions
    {
        /// <summary>Maximum chordal deviation between a facet and the true surface [mm].</summary>
        public double LinearDeflection = 0.1;

        /// <summary>Maximum angular deviation between adjacent facet normals [rad].</summary>
        public double AngularDeflection = 0.5;

        /// <summary>
        /// Target uniform 3D edge length [mm]. When &lt;= 0 it is derived from the bounding-box
        /// diagonal of the whole file and <see cref="TargetEdgeLengthFraction"/>. Ignored by
        /// <see cref="MeshingBackend.Delabella"/>.
        /// </summary>
        public double TargetEdgeLength = 0.0;

        /// <summary>Fraction of the bounding-box diagonal used when TargetEdgeLength &lt;= 0.</summary>
        public double TargetEdgeLengthFraction = 0.02;

        public MeshingBackend Backend = MeshingBackend.Isotropic;

        public EdgeSampling EdgeSampling = EdgeSampling.Adaptive;

        /// <summary>
        /// Multiplier applied to the millimetre-normalized STEP geometry. 0.001 converts to metres.
        /// Use <see cref="ScaleFromExporterUnits"/> to convert a value from the exporter UI.
        /// </summary>
        public double Scale = 0.001;

        /// <summary>
        /// When true, faces that fail to tessellate are skipped and a partial file is still
        /// written; when false, any face failure fails the whole call.
        /// </summary>
        public bool AllowPartialFailure = true;

        /// <summary>Rename each glTF material to the RRGGBBAA hex of its base color.</summary>
        public bool RgbaMaterialNames = false;

        /// <summary>
        /// Converts a scale from the exporter's own units to <see cref="Scale"/>.
        /// </summary>
        public static double ScaleFromExporterUnits(double exporterScale) => exporterScale * 0.001;
    }

    /// <summary>
    /// STEP to render-mesh conversion, performed by the out-of-process superdex_mesh_cli helper.
    /// Colors from the CAD file and the surfaces' analytic normals are preserved.
    /// </summary>
    public static class MeshExporter
    {
        /// <summary>True when the helper executable can be located.</summary>
        public static bool IsAvailable => MeshCliRunner.FindExecutable() != null;

        /// <summary>
        /// Writes every entry of <paramref name="outputs"/> from a single load and tessellation of
        /// <paramref name="stepFilePath"/>.
        /// </summary>
        /// <param name="stepFilePath">STEP file to read.</param>
        /// <param name="outputs">Output files to produce. Must not be empty.</param>
        /// <param name="options">Shared tessellation settings.</param>
        /// <param name="shouldCancel">Polled while waiting; returning true kills the helper.</param>
        /// <returns>One status per output, in the order given.</returns>
        /// <exception cref="MeshCliException">
        /// The STEP could not be read or tessellated, or the helper could not be run. A failure
        /// affecting only one output is reported in that output's status instead.
        /// </exception>
        public static IList<MeshExportStatus> Export(
            string stepFilePath,
            IList<MeshExportOutput> outputs,
            MeshExportOptions options,
            Func<bool> shouldCancel = null)
        {
            if (string.IsNullOrEmpty(stepFilePath))
            {
                throw new ArgumentException("STEP file path is empty.", nameof(stepFilePath));
            }
            if (outputs == null || outputs.Count == 0)
            {
                throw new ArgumentException("No output files were requested.", nameof(outputs));
            }
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            byte[] payload;
            using (var writer = new MeshCliProtocol.PayloadWriter())
            {
                writer.WriteString(stepFilePath);

                // StepVisualExportParams, in declaration order. The output format is not here --
                // it travels per output.
                writer.WriteU32((uint)options.Backend);
                writer.WriteDouble(options.LinearDeflection);
                writer.WriteDouble(options.AngularDeflection);
                writer.WriteDouble(options.TargetEdgeLength);
                writer.WriteDouble(options.TargetEdgeLengthFraction);
                writer.WriteU32((uint)options.EdgeSampling);
                writer.WriteDouble(options.Scale);
                writer.WriteBool(options.AllowPartialFailure);
                writer.WriteBool(options.RgbaMaterialNames);

                writer.WriteU64((ulong)outputs.Count);
                foreach (MeshExportOutput output in outputs)
                {
                    writer.WriteU32((uint)output.Format);
                    writer.WriteString(output.Path);
                }
                payload = writer.ToArray();
            }

            byte[] response =
                MeshCliRunner.Invoke(GeometryOp.ExportStepVisual, payload, shouldCancel);

            using (var reader = new MeshCliProtocol.PayloadReader(response))
            {
                List<uint> raw = reader.ReadU32Array();
                reader.ExpectAtEnd();

                if (raw.Count != outputs.Count)
                {
                    throw new MeshCliException(
                        $"superdex_mesh_cli returned {raw.Count} status(es) for {outputs.Count} output(s).");
                }

                var statuses = new List<MeshExportStatus>(raw.Count);
                foreach (uint value in raw)
                {
                    if (value > (uint)MeshExportStatus.Failed)
                    {
                        throw new MeshCliException(
                            $"superdex_mesh_cli returned unknown export status {value}.");
                    }
                    statuses.Add((MeshExportStatus)value);
                }
                return statuses;
            }
        }
    }
}
