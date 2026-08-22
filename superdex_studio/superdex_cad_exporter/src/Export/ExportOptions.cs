/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.Serialization;

using Meshing;

namespace CADRobotExporter.Export
{
    [Serializable]
    public class ExporterMeshingOptions
    {
        public string meshTagExtension;
        public bool exportCollision;
        public bool perLinkMeshing;
        public string collisionMeshTagExtension;
        public MeshingOptions visualMeshingOptions;
        public MeshingOptions collisionMeshingOptions;

        public ExporterMeshingOptions()
        {
            visualMeshingOptions = new MeshingOptions();
            collisionMeshingOptions = new MeshingOptions();
        }
    }

    [Serializable]
    [SuppressMessage(
        "Microsoft.Usage",
        "CA2239:ProvideDeserializationMethodsForOptionalFields",
        Justification = "The optional fields are seeded from [OnDeserializing], which runs before "
            + "the data is read so values actually present still win. The [OnDeserialized] method "
            + "this rule asks for runs afterwards and would overwrite them, and could not tell an "
            + "absent field from one explicitly saved as the enum's zero value.")]
    public class MeshingOptions
    {
        public double linearDeflection;
        public double angularDeflection;
        public double scale;

        /// <summary>Face tessellation backend.</summary>
        [OptionalField]
        public MeshingBackend backend = MeshingBackend.Isotropic;

        /// <summary>Edge sampling strategy. Ignored by <see cref="MeshingBackend.Delabella"/>.</summary>
        [OptionalField]
        public EdgeSampling edgeSampling = EdgeSampling.Adaptive;

        /// <summary>
        /// Target triangle edge length [mm]. Zero means derive it from
        /// <see cref="targetEdgeLengthFraction"/> instead. Ignored by
        /// <see cref="MeshingBackend.Delabella"/>.
        /// </summary>
        [OptionalField]
        public double targetEdgeLength = 0.0;

        /// <summary>
        /// Fraction of the bounding-box diagonal used as the target edge length when
        /// <see cref="targetEdgeLength"/> is zero. Ignored by <see cref="MeshingBackend.Delabella"/>.
        /// </summary>
        [OptionalField]
        public double targetEdgeLengthFraction = 0.02;

        /// <summary>
        /// Sets the optional fields before deserialization populates whatever the data actually
        /// carries.
        /// </summary>
        [OnDeserializing]
        private void OnDeserializing(StreamingContext context)
        {
            backend = MeshingBackend.Isotropic;
            edgeSampling = EdgeSampling.Adaptive;
            targetEdgeLength = 0.0;
            targetEdgeLengthFraction = 0.02;
        }
    }
}
