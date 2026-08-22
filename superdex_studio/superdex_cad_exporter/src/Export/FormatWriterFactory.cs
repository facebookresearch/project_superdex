/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;

using CADRobotExporter.UI;

namespace CADRobotExporter.Export
{
    /// <summary>
    /// Factory for creating format-specific writers.
    /// </summary>
    public static class FormatWriterFactory
    {
        /// <summary>
        /// Creates a format writer for the specified format.
        /// </summary>
        /// <param name="format">The export format.</param>
        /// <param name="savePath">The output file path.</param>
        /// <param name="folderStructure">The folder structure for mesh directories.</param>
        /// <returns>A format writer instance.</returns>
        public static IFormatWriter Create(ExportFormat format, string savePath, FolderStructure folderStructure = FolderStructure.ROS)
        {
            switch (format)
            {
                case ExportFormat.URDF:
                    return new URDFFormatWriter(savePath);
                case ExportFormat.MJCF:
                    return new MJCFFormatWriter(savePath, folderStructure);
                case ExportFormat.SuperDexBot:
                    return new SuperDexBotFormatWriter(savePath, folderStructure);
                default:
                    throw new ArgumentException($"Unsupported format: {format}", nameof(format));
            }
        }

        /// <summary>
        /// Gets the default file extension for the specified format.
        /// </summary>
        /// <param name="format">The export format.</param>
        /// <returns>The file extension including the leading period.</returns>
        public static string GetFileExtension(ExportFormat format)
        {
            switch (format)
            {
                case ExportFormat.URDF:
                    return ".urdf";
                case ExportFormat.MJCF:
                    return ".xml";
                case ExportFormat.SuperDexBot:
                    return ".superdex_bot";
                default:
                    return ".xml";
            }
        }

        /// <summary>
        /// Gets a human-readable name for the format.
        /// </summary>
        /// <param name="format">The export format.</param>
        /// <returns>A descriptive name for the format.</returns>
        public static string GetFormatName(ExportFormat format)
        {
            switch (format)
            {
                case ExportFormat.URDF:
                    return "URDF (ROS)";
                case ExportFormat.MJCF:
                    return "MJCF (MuJoCo)";
                case ExportFormat.SuperDexBot:
                    return "SuperDex Bot";
                default:
                    return format.ToString();
            }
        }
    }
}
