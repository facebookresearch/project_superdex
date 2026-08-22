/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;

namespace CADRobotExporter.Import
{
    /// <summary>
    /// Represents a node in an imported tree structure.
    /// Platform-agnostic - can be used by both SolidWorks and NX.
    /// </summary>
    public class ImportedTreeNode
    {
        public string LinkName { get; set; }
        public string JointType { get; set; } = "revolute";
        public string JointName { get; set; }
        public List<ImportedTreeNode> Children { get; } = new List<ImportedTreeNode>();
        public ImportedTreeNode Parent { get; set; }
        public int Depth { get; set; }
        public bool IsRoot => Parent == null;

        public ImportedTreeNode(string linkName, string jointName = null, string jointType = "revolute")
        {
            LinkName = linkName;
            JointType = jointType ?? "revolute";
            JointName = jointName;
        }

        /// <summary>
        /// Gets the default joint name based on link name.
        /// </summary>
        public string GetDefaultJointName()
        {
            return $"joint_{LinkName}";
        }

        /// <summary>
        /// Returns true if the joint name is custom (not the default).
        /// </summary>
        public bool HasCustomJointName => !IsRoot && !string.IsNullOrEmpty(JointName) && JointName != GetDefaultJointName();
    }

    /// <summary>
    /// Imports robot tree structures from simple indented text format.
    ///
    /// Format example:
    /// <code>
    /// base_link
    ///   link_1 joint_1 [revolute]
    ///     link_2 joint_2
    ///     link_3 [prismatic]
    ///       link_4
    ///   link_5 my_joint [continuous]
    /// </code>
    ///
    /// Rules:
    /// - First line (no indentation) is the root/base link
    /// - Indentation (2 or 4 spaces, or tabs) defines hierarchy
    /// - Format: link_name [joint_name] [joint_type]
    /// - Joint name is optional, defaults to "joint_{link_name}"
    /// - Joint type in square brackets is optional, defaults to "revolute"
    /// - Valid joint types: revolute, continuous, prismatic, fixed, floating, planar
    /// - Lines starting with # are comments
    /// - Empty lines are ignored
    /// </summary>
    public static class TreeTextImporter
    {
        private static readonly string[] ValidJointTypes =
            { "revolute", "continuous", "prismatic", "fixed", "floating", "planar" };

        // Pattern: indentation, link_name, optional joint_name, optional [joint_type]
        // Examples:
        //   "  link_1" -> link_1, no joint name, no joint type
        //   "  link_1 [revolute]" -> link_1, no joint name, revolute
        //   "  link_1 joint_1" -> link_1, joint_1, no joint type
        //   "  link_1 joint_1 [revolute]" -> link_1, joint_1, revolute
        private static readonly Regex LinePattern =
            new Regex(@"^(\s*)(\S+)(?:\s+(\S+))?(?:\s*\[(\w+)\])?$", RegexOptions.Compiled);

        /// <summary>
        /// Parses tree structure from a string.
        /// </summary>
        /// <param name="text">The indented text representing the tree</param>
        /// <returns>The root node of the parsed tree</returns>
        /// <exception cref="FormatException">Thrown when the format is invalid</exception>
        public static ImportedTreeNode Parse(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new FormatException("Input text is empty.");

            var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            return ParseLines(lines);
        }

        /// <summary>
        /// Parses tree structure from a file.
        /// </summary>
        /// <param name="filePath">Path to the text file</param>
        /// <returns>The root node of the parsed tree</returns>
        public static ImportedTreeNode ParseFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"File not found: {filePath}");

            string text = File.ReadAllText(filePath);
            return Parse(text);
        }

        /// <summary>
        /// Tries to parse tree structure, returning success/failure instead of throwing.
        /// </summary>
        public static bool TryParse(string text, out ImportedTreeNode root, out string error)
        {
            root = null;
            error = null;

            try
            {
                root = Parse(text);
                return true;
            }
            catch (Exception ex)
            {
                error = ex.Message;
                return false;
            }
        }

        /// <summary>
        /// Exports a tree back to indented text format.
        /// </summary>
        /// <param name="root">Root node to export</param>
        /// <param name="indentSpaces">Number of spaces per indent level</param>
        /// <param name="includeJointNames">If true, always include joint names. If false, only include custom joint names.</param>
        public static string Export(ImportedTreeNode root, int indentSpaces = 2)
        {
            if (root == null)
                return string.Empty;

            var lines = new List<string>();
            ExportNode(root, lines, 0, indentSpaces);
            return string.Join(Environment.NewLine, lines);
        }

        private static ImportedTreeNode ParseLines(string[] lines)
        {
            ImportedTreeNode root = null;
            var nodeStack = new Stack<ImportedTreeNode>();
            var depthStack = new Stack<int>();
            int? indentUnit = null;

            int lineNumber = 0;
            foreach (var rawLine in lines)
            {
                lineNumber++;

                // Skip empty lines and comments
                string line = rawLine.TrimEnd();
                if (string.IsNullOrEmpty(line) || line.TrimStart().StartsWith("#"))
                    continue;

                var match = LinePattern.Match(line);
                if (!match.Success)
                    throw new FormatException($"Line {lineNumber}: Invalid format '{line}'");

                string indentation = match.Groups[1].Value;
                string linkName = match.Groups[2].Value;
                string secondToken = match.Groups[3].Success ? match.Groups[3].Value : null;
                string bracketToken = match.Groups[4].Success ? match.Groups[4].Value.ToLower() : null;

                // Determine joint name and joint type from tokens
                string jointName = null;
                string jointType = "revolute";

                if (bracketToken != null)
                {
                    // We have a [joint_type] at the end
                    jointType = bracketToken;
                    jointName = secondToken; // May be null
                }
                else if (secondToken != null)
                {
                    // Second token could be a joint name OR a joint type without brackets (legacy support)
                    if (IsValidJointType(secondToken))
                    {
                        jointType = secondToken.ToLower();
                    }
                    else
                    {
                        jointName = secondToken;
                    }
                }

                // Validate joint type
                if (!IsValidJointType(jointType))
                {
                    throw new FormatException(
                        $"Line {lineNumber}: Invalid joint type '{jointType}'. " +
                        $"Valid types: {string.Join(", ", ValidJointTypes)}");
                }

                // Calculate depth from indentation
                int depth = CalculateDepth(indentation, ref indentUnit, lineNumber);

                // First node must be at depth 0 (root)
                if (root == null)
                {
                    if (depth != 0)
                        throw new FormatException($"Line {lineNumber}: First line (root) must not be indented.");

                    root = new ImportedTreeNode(linkName, null, "fixed") { Depth = 0 };
                    nodeStack.Push(root);
                    depthStack.Push(0);
                    continue;
                }

                // Pop stack until we find the parent
                while (depthStack.Count > 0 && depthStack.Peek() >= depth)
                {
                    nodeStack.Pop();
                    depthStack.Pop();
                }

                if (nodeStack.Count == 0)
                    throw new FormatException($"Line {lineNumber}: Invalid indentation - no parent found.");

                // Create the new node
                var parent = nodeStack.Peek();

                // Use default joint name if not specified
                if (string.IsNullOrEmpty(jointName))
                {
                    jointName = $"joint_{linkName}";
                }

                var node = new ImportedTreeNode(linkName, jointName, jointType)
                {
                    Parent = parent,
                    Depth = depth
                };
                parent.Children.Add(node);

                // Push onto stack
                nodeStack.Push(node);
                depthStack.Push(depth);
            }

            if (root == null)
                throw new FormatException("No valid lines found in input.");

            return root;
        }

        private static int CalculateDepth(string indentation, ref int? indentUnit, int lineNumber)
        {
            if (string.IsNullOrEmpty(indentation))
                return 0;

            // Count spaces (tabs count as 4 spaces)
            int spaces = 0;
            foreach (char c in indentation)
            {
                if (c == '\t')
                    spaces += 4;
                else if (c == ' ')
                    spaces++;
            }

            if (spaces == 0)
                return 0;

            // Determine indent unit from first indented line
            if (!indentUnit.HasValue)
            {
                indentUnit = spaces;
            }

            // Calculate depth
            if (spaces % indentUnit.Value != 0)
            {
                throw new FormatException(
                    $"Line {lineNumber}: Inconsistent indentation. " +
                    $"Expected multiples of {indentUnit.Value} spaces.");
            }

            return spaces / indentUnit.Value;
        }

        private static bool IsValidJointType(string jointType)
        {
            foreach (var valid in ValidJointTypes)
            {
                if (string.Equals(valid, jointType, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        private static void ExportNode(ImportedTreeNode node, List<string> lines, int depth, int indentSpaces)
        {
            string indent = new string(' ', depth * indentSpaces);

            string line = $"{indent}{node.LinkName}";

            if (!node.IsRoot)
            {
                line += $" {node.JointName}";
                line += $" [{node.JointType}]";
            }

            lines.Add(line);

            foreach (var child in node.Children)
            {
                ExportNode(child, lines, depth + 1, indentSpaces);
            }
        }

        /// <summary>
        /// Gets a sample/template text that demonstrates the format.
        /// </summary>
        public static string GetSampleText()
        {
            return @"# Robot Tree Definition
# Lines starting with # are comments
# Indentation defines hierarchy (2 or 4 spaces, or tabs)
# Format: link_name [joint_name] [joint_type]
# Joint name is optional, defaults to ""joint_{link_name}""
# Joint type in brackets is optional, defaults to [revolute]
# Valid joint types: revolute, continuous, prismatic, fixed, floating, planar

base_link
  link_1 joint_1 [revolute]
    link_2 my_custom_joint
    link_3 [prismatic]
  link_4 [continuous]
    link_5 special_joint [fixed]
";
        }

        /// <summary>
        /// Validates text without fully parsing, returns list of errors.
        /// </summary>
        public static List<string> Validate(string text)
        {
            var errors = new List<string>();

            if (string.IsNullOrWhiteSpace(text))
            {
                errors.Add("Input text is empty.");
                return errors;
            }

            var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            int? indentUnit = null;
            bool foundRoot = false;

            int lineNumber = 0;
            foreach (var rawLine in lines)
            {
                lineNumber++;
                string line = rawLine.TrimEnd();

                if (string.IsNullOrEmpty(line) || line.TrimStart().StartsWith("#"))
                    continue;

                var match = LinePattern.Match(line);
                if (!match.Success)
                {
                    errors.Add($"Line {lineNumber}: Invalid format");
                    continue;
                }

                string indentation = match.Groups[1].Value;
                string bracketToken = match.Groups[4].Success ? match.Groups[4].Value.ToLower() : null;

                if (bracketToken != null && !IsValidJointType(bracketToken))
                {
                    errors.Add($"Line {lineNumber}: Invalid joint type '{bracketToken}'");
                }

                try
                {
                    int depth = CalculateDepth(indentation, ref indentUnit, lineNumber);
                    if (!foundRoot && depth != 0)
                    {
                        errors.Add($"Line {lineNumber}: First line (root) must not be indented");
                    }
                    foundRoot = true;
                }
                catch (FormatException ex)
                {
                    errors.Add(ex.Message);
                }
            }

            if (!foundRoot)
            {
                errors.Add("No valid lines found - at least one link is required");
            }

            return errors;
        }
    }
}
