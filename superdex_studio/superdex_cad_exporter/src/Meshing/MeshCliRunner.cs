/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;

namespace Meshing
{
    /// <summary>
    /// Locates and invokes the superdex_mesh_cli helper executable.
    /// </summary>
    internal static class MeshCliRunner
    {
        /// <summary>Overrides the helper location. Same variable the C++ client honours.</summary>
        public const string PathEnvironmentVariable = "SUPERDEX_MESH_CLI_PATH";

        private const string ExecutableName = "superdex_mesh_cli.exe";

        /// <summary>How long to wait for one conversion before giving up and killing the child.</summary>
        public static TimeSpan Timeout { get; set; } = TimeSpan.FromMinutes(10);

        /// <summary>
        /// Full path to the helper, or null when it cannot be found.
        /// Checks the environment override first, then the directory holding this assembly -- which
        /// is the add-in's own install directory for both SolidWorks and NX.
        /// </summary>
        public static string FindExecutable()
        {
            string overridePath = Environment.GetEnvironmentVariable(PathEnvironmentVariable);
            if (!string.IsNullOrEmpty(overridePath) && File.Exists(overridePath))
            {
                return overridePath;
            }

            string assemblyDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            if (!string.IsNullOrEmpty(assemblyDir))
            {
                string adjacent = Path.Combine(assemblyDir, ExecutableName);
                if (File.Exists(adjacent))
                {
                    return adjacent;
                }
            }
            return null;
        }

        /// <summary>
        /// Sends one request and returns the response payload.
        /// </summary>
        /// <param name="op">Operation to dispatch.</param>
        /// <param name="payload">Encoded request payload.</param>
        /// <param name="shouldCancel">
        /// Polled while waiting; returning true kills the helper. May be null.
        /// </param>
        /// <exception cref="MeshCliException">
        /// The helper is missing, crashed, timed out, was cancelled, or reported a failure.
        /// </exception>
        public static byte[] Invoke(GeometryOp op, byte[] payload, Func<bool> shouldCancel = null)
        {
            string exePath = FindExecutable();
            if (exePath == null)
            {
                throw new MeshCliException(
                    $"{ExecutableName} was not found next to the add-in. Reinstall the exporter, or "
                        + $"set {PathEnvironmentVariable} to the helper's full path.");
            }

            byte[] request = MeshCliProtocol.EncodeRequestFrame(op, payload);

            var startInfo = new ProcessStartInfo
            {
                FileName = exePath,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                // Standard error is deliberately left inherited. The helper logs diagnostics there
                // and reroutes OpenCascade's chatter into it, and a redirected pipe that nobody
                // drains can fill and block the child. The C++ client inherits it for the same
                // reason.
                RedirectStandardError = false,
                WorkingDirectory = Path.GetDirectoryName(exePath) ?? string.Empty,
            };

            byte[] responseBytes;
            using (var process = new Process { StartInfo = startInfo })
            {
                try
                {
                    process.Start();
                }
                catch (Exception ex)
                {
                    throw new MeshCliException($"Failed to start {ExecutableName}: {ex.Message}", ex);
                }

                try
                {
                    // Write the whole request, then close stdin. The helper reads to EOF before it
                    // produces anything, so this ordering cannot deadlock -- and without the close
                    // it would wait forever.
                    using (Stream stdin = process.StandardInput.BaseStream)
                    {
                        stdin.Write(request, 0, request.Length);
                        stdin.Flush();
                    }

                    using (var buffer = new MemoryStream())
                    {
                        process.StandardOutput.BaseStream.CopyTo(buffer);
                        responseBytes = buffer.ToArray();
                    }

                    if (!WaitForExit(process, shouldCancel))
                    {
                        throw new MeshCliException(
                            shouldCancel != null && shouldCancel()
                                ? "Mesh conversion was cancelled."
                                : $"{ExecutableName} did not finish within {Timeout.TotalMinutes:0} minutes.");
                    }
                }
                catch (MeshCliException)
                {
                    KillQuietly(process);
                    throw;
                }
                catch (Exception ex)
                {
                    KillQuietly(process);
                    throw new MeshCliException(
                        $"Failed to communicate with {ExecutableName}: {ex.Message}", ex);
                }

                // The helper exits 0 even for handled errors, reporting detail in the response
                // frame, so a non-zero exit means it died before it could answer.
                if (process.ExitCode != 0)
                {
                    throw new MeshCliException(
                        $"{ExecutableName} exited with status {process.ExitCode}.");
                }
            }

            MeshCliProtocol.DecodeResponseFrame(responseBytes, out uint status, out byte[] response);
            if (status != 0)
            {
                string message = System.Text.Encoding.UTF8.GetString(response);
                throw new MeshCliException(
                    string.IsNullOrEmpty(message)
                        ? "superdex_mesh_cli reported an error processing the request."
                        : message);
            }
            return response;
        }

        private static bool WaitForExit(Process process, Func<bool> shouldCancel)
        {
            if (shouldCancel == null)
            {
                return process.WaitForExit((int)Timeout.TotalMilliseconds);
            }

            // Poll so a cancel from the progress dialog can kill a running tessellation. The
            // in-process library could not be interrupted at all.
            var elapsed = Stopwatch.StartNew();
            while (elapsed.Elapsed < Timeout)
            {
                if (process.WaitForExit(200))
                {
                    return true;
                }
                if (shouldCancel())
                {
                    return false;
                }
            }
            return false;
        }

        private static void KillQuietly(Process process)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill();
                }
            }
            catch (Exception)
            {
                // The process is already gone, or we cannot signal it; either way there is nothing
                // useful to do while unwinding.
            }
        }
    }
}
