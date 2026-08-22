/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

using Serilog;
using Serilog.Core;
using Serilog.Events;

namespace CADRobotExporter.Utilities
{
    [ComVisible(false)]
    public static class Logger
    {
        private class FileNameEnricher : ILogEventEnricher
        {
            public void Enrich(LogEvent logEvent, ILogEventPropertyFactory propertyFactory)
            {
                var frame = new StackFrame(5, true); // Adjust skip frames as needed
                var fileName = frame.GetFileName();
                if (fileName != null)
                {
                    var shortName = System.IO.Path.GetFileName(fileName);
                    logEvent.AddPropertyIfAbsent(propertyFactory.CreateProperty("FileName", shortName));
                }
            }
        }

        private static bool Initialized = false;

        public static void Setup()
        {
            if (Initialized)
                return;

            string homeDir = Environment.ExpandEnvironmentVariables("%HOMEDRIVE%%HOMEPATH%");
#if NX
            string logPath = Path.Combine(homeDir, "NXRobotExporterLogs", "NXRobotExporter.log");
#elif SOLIDWORKS
            string logPath = Path.Combine(homeDir, "sw2urdf_logs", "sw2urdf.log");
#endif

            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Information()
                .Enrich.With(new FileNameEnricher())
                .WriteTo.File(
                    logPath,
                    rollingInterval: RollingInterval.Day,
                    fileSizeLimitBytes: 10 * 1024 * 1024, // 10MB
                    rollOnFileSizeLimit: true,
                    retainedFileCountLimit: 5,
                    outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss} {Level:u3} {FileName}:{LineNumber} - {Message:lj}{NewLine}{Exception}"
                )
                .WriteTo.Debug()
                .CreateLogger();

            Log.Information(new string('-', 80));
            Log.Information("Logging commencing for SW2URDF exporter");
            Log.Information("Commit version {CommitVersion}", Versioning.Version.GetCommitVersion());
            Log.Information("Build version {BuildVersion}", Versioning.Version.GetBuildVersion());

            Initialized = true;
        }

        public static ILogger GetLogger()
        {
            Setup();
            return Log.Logger;
        }

        public static string GetLogFolder()
        {
            return Path.Combine(
                Environment.ExpandEnvironmentVariables("%HOMEDRIVE%%HOMEPATH%"),
#if NX
                "NXRobotExporterLogs"
#elif SOLIDWORKS
                "sw2urdf_logs"
#endif
            );
        }
    }
}
