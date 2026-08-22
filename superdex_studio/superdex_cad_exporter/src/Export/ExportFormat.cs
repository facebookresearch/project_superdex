/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace CADRobotExporter.Export
{
    /// <summary>
    /// Supported robot model export formats.
    /// </summary>
    public enum ExportFormat
    {
        /// <summary>
        /// Unified Robot Description Format (ROS standard)
        /// </summary>
        URDF,

        /// <summary>
        /// MuJoCo XML Format
        /// </summary>
        MJCF,

        /// <summary>
        /// SuperDex Bot (json)
        /// </summary>
        SuperDexBot
    }
}
