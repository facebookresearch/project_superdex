/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.Linq;

using NXOpen;
using NXOpen.Assemblies;
using NXOpen.BlockStyler;
using NXOpen.UF;
using NXOpen.Features;

using CADRobotExporter.Utilities;

namespace CADRobotExporter.CAD.NX
{
    /// <summary>
    /// GUID Sanitizer dialog for cleaning up duplicate GUIDs created by patterning/mirroring operations.
    /// </summary>
    public class GuidSanitizer : IDisposable
    {
        private static readonly Serilog.ILogger logger = Logger.GetLogger();

        private static Session theSession = null;
        private static NXOpen.UI theUI = null;
        private string theDlxFileName;
        private NXOpen.BlockStyler.BlockDialog theDialog;
        private NXOpen.BlockStyler.Group group0;
        private NXOpen.BlockStyler.Label label02;
        private NXOpen.BlockStyler.Label label03;
        private NXOpen.BlockStyler.Group group;
        private NXOpen.BlockStyler.Label label0;
        private NXOpen.BlockStyler.ListBox listBoxKeep;
        private NXOpen.BlockStyler.Separator separator0;
        private NXOpen.BlockStyler.Label label01;
        private NXOpen.BlockStyler.ListBox listBoxRemove;

        /// <summary>
        /// Represents an object with a GUID attribute.
        /// </summary>
        private class GuidEntry
        {
            public Tag Tag { get; set; }
            public string Guid { get; set; }
            public string DisplayName { get; set; }
            public string AttributeName { get; set; }
            public NXObject Object { get; set; }
            public string ObjectType { get; set; }
        }

        private Dictionary<string, List<GuidEntry>> guidGroups = new Dictionary<string, List<GuidEntry>>();
        private Dictionary<string, GuidEntry> keepListMap = new Dictionary<string, GuidEntry>();
        private Dictionary<string, GuidEntry> removeListMap = new Dictionary<string, GuidEntry>();
        private List<DisplayableObject> highlightedObjects = new List<DisplayableObject>();
        private HashSet<Tag> scannedParts = new HashSet<Tag>();

        /// <summary>
        /// Result of a duplicate GUID check.
        /// </summary>
        public class DuplicateCheckResult
        {
            public int TotalGuidsScanned { get; set; }
            public int DuplicateGuids { get; set; }
            public int TotalDuplicateObjects { get; set; }
            public bool HasDuplicates => DuplicateGuids > 0;
        }

        /// <summary>
        /// Checks for duplicate GUIDs in the current work part without showing the UI.
        /// </summary>
        public static DuplicateCheckResult CheckForDuplicates()
        {
            var result = new DuplicateCheckResult();
            var session = Session.GetSession();
            Part workPart = session.Parts.Work;

            if (workPart == null)
            {
                return result;
            }

            var guidGroups = ScanAllGuidsStatic(session, workPart);

            result.TotalGuidsScanned = guidGroups.Count;

            foreach (var kvp in guidGroups)
            {
                if (kvp.Value.Count > 1)
                {
                    result.DuplicateGuids++;
                    result.TotalDuplicateObjects += kvp.Value.Count - 1; // -1 because one is the "original"
                }
            }

            return result;
        }

        private static Dictionary<string, List<GuidEntryData>> ScanAllGuidsStatic(Session session, Part workPart)
        {
            var guidGroups = new Dictionary<string, List<GuidEntryData>>();
            var scannedParts = new HashSet<Tag>();

            ScanPartObjectsStatic(workPart, null, guidGroups);
            scannedParts.Add(workPart.Tag);

            if (workPart.ComponentAssembly?.RootComponent != null)
            {
                ScanComponentRecursiveStatic(workPart, workPart.ComponentAssembly.RootComponent, guidGroups, scannedParts);
            }

            return guidGroups;
        }

        private class GuidEntryData
        {
            public Tag Tag { get; set; }
            public string Guid { get; set; }
            public string ObjectType { get; set; }
        }

        private static void ScanComponentRecursiveStatic(Part workPart, Component component, Dictionary<string, List<GuidEntryData>> guidGroups, HashSet<Tag> scannedParts)
        {
            if (component == null)
                return;

            // Components themselves can have duplicate GUIDs (from patterning), so always check
            try
            {
                if (component.HasUserAttribute(NXPersistentId.COMPONENT_GUID_ATTR, NXObject.AttributeType.String, -1))
                {
                    string guid = component.GetStringUserAttribute(NXPersistentId.COMPONENT_GUID_ATTR, -1);
                    if (!string.IsNullOrEmpty(guid))
                    {
                        AddGuidEntryStatic(guidGroups, guid, component.Tag, "Component");
                    }
                }
            }
            catch { }

            // Only scan each Part file once - objects inside parts (bodies, csys, axes) have GUIDs
            // stored on the prototype. If the same Part is used multiple times via different
            // component chains, we'd otherwise incorrectly flag valid GUIDs as duplicates.
            Part componentPart = component.Prototype as Part;
            if (componentPart != null && !scannedParts.Contains(componentPart.Tag))
            {
                scannedParts.Add(componentPart.Tag);
                ScanPartObjectsStatic(componentPart, component, guidGroups);
            }

            Component[] children = component.GetChildren();
            if (children != null)
            {
                foreach (Component child in children)
                {
                    ScanComponentRecursiveStatic(workPart, child, guidGroups, scannedParts);
                }
            }
        }

        private static void ScanPartObjectsStatic(Part part, Component owningComponent, Dictionary<string, List<GuidEntryData>> guidGroups)
        {
            foreach (Body body in part.Bodies)
            {
                try
                {
                    if (body.HasUserAttribute(NXPersistentId.BODY_GUID_ATTR, NXObject.AttributeType.String, -1))
                    {
                        string guid = body.GetStringUserAttribute(NXPersistentId.BODY_GUID_ATTR, -1);
                        if (!string.IsNullOrEmpty(guid))
                        {
                            AddGuidEntryStatic(guidGroups, guid, body.Tag, "Body");
                        }
                    }
                }
                catch { }
            }

            foreach (CartesianCoordinateSystem csys in part.CoordinateSystems)
            {
                try
                {
                    if (csys.HasUserAttribute(NXPersistentId.CSYS_GUID_ATTR, NXObject.AttributeType.String, -1))
                    {
                        string guid = csys.GetStringUserAttribute(NXPersistentId.CSYS_GUID_ATTR, -1);
                        if (!string.IsNullOrEmpty(guid))
                        {
                            AddGuidEntryStatic(guidGroups, guid, csys.Tag, "CSYS");
                        }
                    }
                }
                catch { }
            }

            foreach (DisplayableObject datum in part.Datums)
            {
                if (datum is DatumAxis axis)
                {
                    try
                    {
                        if (axis.HasUserAttribute(NXPersistentId.AXIS_GUID_ATTR, NXObject.AttributeType.String, -1))
                        {
                            string guid = axis.GetStringUserAttribute(NXPersistentId.AXIS_GUID_ATTR, -1);
                            if (!string.IsNullOrEmpty(guid))
                            {
                                AddGuidEntryStatic(guidGroups, guid, axis.Tag, "Axis");
                            }
                        }
                    }
                    catch { }
                }
            }

            foreach (Point point in part.Points)
            {
                try
                {
                    if (point.HasUserAttribute(NXPersistentId.POINT_GUID_ATTR, NXObject.AttributeType.String, -1))
                    {
                        string guid = point.GetStringUserAttribute(NXPersistentId.POINT_GUID_ATTR, -1);
                        if (!string.IsNullOrEmpty(guid))
                        {
                            AddGuidEntryStatic(guidGroups, guid, point.Tag, "Point");
                        }
                    }
                }
                catch { }
            }
        }

        private static void AddGuidEntryStatic(Dictionary<string, List<GuidEntryData>> guidGroups, string guid, Tag tag, string objectType)
        {
            if (!guidGroups.ContainsKey(guid))
            {
                guidGroups[guid] = new List<GuidEntryData>();
            }
            guidGroups[guid].Add(new GuidEntryData { Tag = tag, Guid = guid, ObjectType = objectType });
        }

        public GuidSanitizer()
        {
            try
            {
                theSession = Session.GetSession();
                theUI = NXOpen.UI.GetUI();
                theDlxFileName = "GuidSanitizer.dlx";
                theDialog = theUI.CreateDialog(theDlxFileName);
                theDialog.AddApplyHandler(new NXOpen.BlockStyler.BlockDialog.Apply(apply_cb));
                theDialog.AddOkHandler(new NXOpen.BlockStyler.BlockDialog.Ok(ok_cb));
                theDialog.AddUpdateHandler(new NXOpen.BlockStyler.BlockDialog.Update(update_cb));
                theDialog.AddInitializeHandler(new NXOpen.BlockStyler.BlockDialog.Initialize(initialize_cb));
                theDialog.AddDialogShownHandler(new NXOpen.BlockStyler.BlockDialog.DialogShown(dialogShown_cb));
            }
            catch (Exception ex)
            {
                logger.Error($"GuidSanitizer constructor failed: {ex}");
                throw;
            }
        }

        public static NXOpen.MenuBar.MenuBarManager.CallbackStatus LaunchGuidSanitizer(
            NXOpen.MenuBar.MenuButtonEvent buttonEvent)
        {
            GuidSanitizer theSanitizer = null;
            try
            {
                theSanitizer = new GuidSanitizer();
                theSanitizer.Launch();
            }
            catch (Exception ex)
            {
                NXOpen.UI.GetUI().NXMessageBox.Show("GUID Sanitizer", NXMessageBox.DialogType.Error, ex.ToString());
            }
            finally
            {
                theSanitizer?.Dispose();
            }
            return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Continue;
        }

        public NXOpen.BlockStyler.BlockDialog.DialogResponse Launch()
        {
            NXOpen.BlockStyler.BlockDialog.DialogResponse dialogResponse =
                NXOpen.BlockStyler.BlockDialog.DialogResponse.Invalid;
            try
            {
                dialogResponse = theDialog.Launch();
            }
            catch (Exception ex)
            {
                theUI.NXMessageBox.Show("GUID Sanitizer", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return dialogResponse;
        }

        public void Dispose()
        {
            ClearHighlights();
            if (theDialog != null)
            {
                theDialog.Dispose();
                theDialog = null;
            }
        }

        public void initialize_cb()
        {
            try
            {
                group0 = (NXOpen.BlockStyler.Group)theDialog.TopBlock.FindBlock("group0");
                label02 = (NXOpen.BlockStyler.Label)theDialog.TopBlock.FindBlock("label02");
                label03 = (NXOpen.BlockStyler.Label)theDialog.TopBlock.FindBlock("label03");
                group = (NXOpen.BlockStyler.Group)theDialog.TopBlock.FindBlock("group");
                label0 = (NXOpen.BlockStyler.Label)theDialog.TopBlock.FindBlock("label0");
                listBoxKeep = (NXOpen.BlockStyler.ListBox)theDialog.TopBlock.FindBlock("listBoxKeep");
                separator0 = (NXOpen.BlockStyler.Separator)theDialog.TopBlock.FindBlock("separator0");
                label01 = (NXOpen.BlockStyler.Label)theDialog.TopBlock.FindBlock("label01");
                listBoxRemove = (NXOpen.BlockStyler.ListBox)theDialog.TopBlock.FindBlock("listBoxRemove");

                listBoxKeep.SetDeleteHandler(new NXOpen.BlockStyler.ListBox.DeleteCallback(OnKeepDeleteCallback));
                listBoxRemove.SetAddHandler(new NXOpen.BlockStyler.ListBox.AddCallback(OnRemoveAddCallback));
            }
            catch (Exception ex)
            {
                theUI.NXMessageBox.Show("GUID Sanitizer", NXMessageBox.DialogType.Error, ex.ToString());
            }
        }

        public void dialogShown_cb()
        {
            try
            {
                ScanAllGuids();
                PopulateListsWithDefaults();
            }
            catch (Exception ex)
            {
                theUI.NXMessageBox.Show("GUID Sanitizer", NXMessageBox.DialogType.Error, ex.ToString());
            }
        }

        public int apply_cb()
        {
            int errorCode = 0;
            try
            {
                int removedCount = 0;
                foreach (var entry in removeListMap.Values)
                {
                    try
                    {
                        // Unlock might be needed if attribute was locked
                        // entry.Object.SetUserAttributeLock(entry.AttributeName, NXObject.AttributeType.String, false);

                        // Derived objects (from mirroing, etc) must have their attributes overridden instead, so we
                        // just set it to empty isntead.
                        entry.Object.SetUserAttribute(
                            entry.AttributeName,
                            -1,
                            "",
                            Update.Option.Now);
                        removedCount++;
                        logger.Information($"Deleted GUID attribute '{entry.AttributeName}' from {entry.DisplayName}");
                    }
                    catch (Exception ex)
                    {
                        logger.Warning($"Failed to delete GUID from {entry.DisplayName}: {ex.Message}");
                    }
                }

                ClearHighlights();

                if (removedCount > 0)
                {
                    theUI.NXMessageBox.Show(
                        "GUID Sanitizer",
                        NXMessageBox.DialogType.Information,
                        $"Removed {removedCount} duplicate GUID(s).");
                }
            }
            catch (Exception ex)
            {
                errorCode = 1;
                theUI.NXMessageBox.Show("GUID Sanitizer", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return errorCode;
        }

        public int update_cb(NXOpen.BlockStyler.UIBlock block)
        {
            try
            {
                if (block == listBoxKeep)
                {
                    HighlightSelectedItems(listBoxKeep, keepListMap);
                }
                else if (block == listBoxRemove)
                {
                    HighlightSelectedItems(listBoxRemove, removeListMap);
                }
            }
            catch (Exception ex)
            {
                theUI.NXMessageBox.Show("GUID Sanitizer", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return 0;
        }

        public int ok_cb()
        {
            int errorCode = 0;
            try
            {
                errorCode = apply_cb();
            }
            catch (Exception ex)
            {
                errorCode = 1;
                theUI.NXMessageBox.Show("GUID Sanitizer", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return errorCode;
        }

        /// <summary>
        /// Called when user clicks the delete button (X) on the Keep list.
        /// Moves selected items from Keep to Remove.
        /// </summary>
        public int OnKeepDeleteCallback(NXOpen.BlockStyler.ListBox listBox)
        {
            try
            {
                int[] selectedIndices = listBox.GetSelectedItems();
                string[] currentItems = listBox.GetListItems();

                foreach (int index in selectedIndices.OrderByDescending(i => i))
                {
                    if (index >= 0 && index < currentItems.Length)
                    {
                        string displayName = currentItems[index];
                        if (keepListMap.TryGetValue(displayName, out GuidEntry entry))
                        {
                            keepListMap.Remove(displayName);
                            removeListMap[displayName] = entry;
                        }
                    }
                }

                RefreshListBoxes();
            }
            catch (Exception ex)
            {
                logger.Warning($"OnKeepDeleteCallback error: {ex.Message}");
            }
            return 0;
        }

        /// <summary>
        /// Called when user clicks the add button (+) on the Remove list.
        /// Moves selected items from Remove back to Keep.
        /// </summary>
        public int OnRemoveAddCallback(NXOpen.BlockStyler.ListBox listBox)
        {
            try
            {
                int[] selectedIndices = listBox.GetSelectedItems();
                string[] currentItems = listBox.GetListItems();

                foreach (int index in selectedIndices.OrderByDescending(i => i))
                {
                    if (index >= 0 && index < currentItems.Length)
                    {
                        string displayName = currentItems[index];
                        if (removeListMap.TryGetValue(displayName, out GuidEntry entry))
                        {
                            removeListMap.Remove(displayName);
                            keepListMap[displayName] = entry;
                        }
                    }
                }

                RefreshListBoxes();
            }
            catch (Exception ex)
            {
                logger.Warning($"OnRemoveAddCallback error: {ex.Message}");
            }
            return 0;
        }

        private void RefreshListBoxes()
        {
            listBoxKeep.SetListItems(keepListMap.Keys.OrderBy(k => k).ToArray());
            listBoxRemove.SetListItems(removeListMap.Keys.OrderBy(k => k).ToArray());
        }

        private void ScanAllGuids()
        {
            guidGroups.Clear();
            scannedParts.Clear();

            Part workPart = theSession.Parts.Work;
            if (workPart == null)
            {
                logger.Warning("No work part open");
                return;
            }

            // Scan the work part's own objects (for non-assembly or single part case)
            ScanPartObjects(workPart, null);
            scannedParts.Add(workPart.Tag);

            // If assembly, scan all components recursively
            if (workPart.ComponentAssembly?.RootComponent != null)
            {
                ScanComponentRecursive(workPart, workPart.ComponentAssembly.RootComponent);
            }

            logger.Information($"ScanAllGuids: Found {guidGroups.Count} unique GUIDs");
        }

        private void ScanComponentRecursive(Part workPart, Component component)
        {
            if (component == null)
                return;

            // Components themselves can have duplicate GUIDs (from patterning), so always check
            try
            {
                if (component.HasUserAttribute(NXPersistentId.COMPONENT_GUID_ATTR, NXObject.AttributeType.String, -1))
                {
                    string guid = component.GetStringUserAttribute(NXPersistentId.COMPONENT_GUID_ATTR, -1);
                    if (!string.IsNullOrEmpty(guid))
                    {
                        AddGuidEntry(guid, component, NXPersistentId.COMPONENT_GUID_ATTR, "Component", component.Name);
                    }
                }
            }
            catch { }

            // Only scan each Part file once - objects inside parts (bodies, csys, axes) have GUIDs
            // stored on the prototype. If the same Part is used multiple times via different
            // component chains, we'd otherwise incorrectly flag valid GUIDs as duplicates.
            Part componentPart = component.Prototype as Part;
            if (componentPart != null && !scannedParts.Contains(componentPart.Tag))
            {
                scannedParts.Add(componentPart.Tag);
                ScanPartObjects(componentPart, component);
            }

            // Recurse through children
            Component[] children = component.GetChildren();
            if (children != null)
            {
                foreach (Component child in children)
                {
                    ScanComponentRecursive(workPart, child);
                }
            }
        }

        private void ScanPartObjects(Part part, Component owningComponent)
        {
            string prefix = owningComponent != null ? $"{owningComponent.Name}/" : "";

            // Scan Bodies
            foreach (Body body in part.Bodies)
            {
                try
                {
                    if (body.HasUserAttribute(NXPersistentId.BODY_GUID_ATTR, NXObject.AttributeType.String, -1))
                    {
                        string guid = body.GetStringUserAttribute(NXPersistentId.BODY_GUID_ATTR, -1);
                        if (!string.IsNullOrEmpty(guid))
                        {
                            string name = GetObjectName(body, "Body");
                            AddGuidEntry(guid, body, NXPersistentId.BODY_GUID_ATTR, "Body", prefix + name);
                        }
                    }
                }
                catch { }
            }

            // Scan Coordinate Systems
            foreach (CartesianCoordinateSystem csys in part.CoordinateSystems)
            {
                try
                {
                    if (csys.HasUserAttribute(NXPersistentId.CSYS_GUID_ATTR, NXObject.AttributeType.String, -1))
                    {
                        string guid = csys.GetStringUserAttribute(NXPersistentId.CSYS_GUID_ATTR, -1);
                        if (!string.IsNullOrEmpty(guid))
                        {
                            string name = GetObjectName(csys, "CSYS");
                            AddGuidEntry(guid, csys, NXPersistentId.CSYS_GUID_ATTR, "CSYS", prefix + name);
                        }
                    }
                }
                catch { }
            }

            // Scan Datum Axes
            foreach (DisplayableObject datum in part.Datums)
            {
                if (datum is DatumAxis axis)
                {
                    try
                    {
                        if (axis.HasUserAttribute(NXPersistentId.AXIS_GUID_ATTR, NXObject.AttributeType.String, -1))
                        {
                            string guid = axis.GetStringUserAttribute(NXPersistentId.AXIS_GUID_ATTR, -1);
                            if (!string.IsNullOrEmpty(guid))
                            {
                                string name = GetObjectName(axis, "Axis");
                                AddGuidEntry(guid, axis, NXPersistentId.AXIS_GUID_ATTR, "Axis", prefix + name);
                            }
                        }
                    }
                    catch { }
                }
            }

            // Scan Points
            foreach (Point point in part.Points)
            {
                try
                {
                    if (point.HasUserAttribute(NXPersistentId.POINT_GUID_ATTR, NXObject.AttributeType.String, -1))
                    {
                        string guid = point.GetStringUserAttribute(NXPersistentId.POINT_GUID_ATTR, -1);
                        if (!string.IsNullOrEmpty(guid))
                        {
                            string name = GetObjectName(point, "Point");
                            AddGuidEntry(guid, point, NXPersistentId.POINT_GUID_ATTR, "Point", prefix + name);
                        }
                    }
                }
                catch { }
            }
        }

        private string GetObjectName(NXObject obj, string fallbackType)
        {
            try
            {
                string name = obj.Name;
                if (!string.IsNullOrEmpty(name))
                    return name;
            }
            catch { }

            return $"{fallbackType}_{obj.Tag}";
        }

        private void AddGuidEntry(string guid, NXObject obj, string attrName, string objType, string objectName)
        {
            var entry = new GuidEntry
            {
                Tag = obj.Tag,
                Guid = guid,
                AttributeName = attrName,
                Object = obj,
                ObjectType = objType,
                DisplayName = "" // Will be set during population
            };

            if (!guidGroups.ContainsKey(guid))
            {
                guidGroups[guid] = new List<GuidEntry>();
            }
            guidGroups[guid].Add(entry);

            // Store object name temporarily
            entry.DisplayName = objectName;
        }

        private void PopulateListsWithDefaults()
        {
            keepListMap.Clear();
            removeListMap.Clear();

            foreach (var kvp in guidGroups)
            {
                string guid = kvp.Key;
                List<GuidEntry> entries = kvp.Value;

                // Only process GUIDs that appear more than once (duplicates)
                if (entries.Count <= 1)
                    continue;

                string shortGuid = guid.Length > 8 ? guid.Substring(0, 8) : guid;

                // For components, keep the LAST duplicate (most likely original)
                // For other objects (bodies, csys, axes), keep the FIRST (most likely original)
                bool isComponentGroup = entries.All(e => e.ObjectType == "Component");
                int keepIndex = isComponentGroup ? entries.Count - 1 : 0;

                for (int i = 0; i < entries.Count; i++)
                {
                    var entry = entries[i];
                    string displayName;

                    if (i == 0)
                    {
                        // First occurrence - no number suffix
                        displayName = $"{entry.DisplayName}:{shortGuid}";
                    }
                    else
                    {
                        // Subsequent occurrences get numbered
                        displayName = $"{entry.DisplayName}:{shortGuid} ({i + 1})";
                    }

                    // Ensure uniqueness
                    string uniqueDisplayName = displayName;
                    int suffix = 1;
                    while (keepListMap.ContainsKey(uniqueDisplayName) || removeListMap.ContainsKey(uniqueDisplayName))
                    {
                        uniqueDisplayName = $"{displayName}_{suffix++}";
                    }

                    entry.DisplayName = uniqueDisplayName;

                    if (i == keepIndex)
                    {
                        // This is the one to keep
                        keepListMap[uniqueDisplayName] = entry;
                    }
                    else
                    {
                        // Duplicates go to Remove list
                        removeListMap[uniqueDisplayName] = entry;
                    }
                }
            }

            RefreshListBoxes();

            logger.Information($"PopulateListsWithDefaults: Keep={keepListMap.Count}, Remove={removeListMap.Count}");
        }

        private void HighlightSelectedItems(NXOpen.BlockStyler.ListBox listBox, Dictionary<string, GuidEntry> map)
        {
            ClearHighlights();

            int[] selectedIndices = listBox.GetSelectedItems();
            string[] items = listBox.GetListItems();

            foreach (int index in selectedIndices)
            {
                if (index >= 0 && index < items.Length)
                {
                    string displayName = items[index];
                    if (map.TryGetValue(displayName, out GuidEntry entry))
                    {
                        if (entry.Object is DisplayableObject dispObj)
                        {
                            try
                            {
                                dispObj.Highlight();
                                highlightedObjects.Add(dispObj);
                            }
                            catch (Exception ex)
                            {
                                logger.Warning($"Failed to highlight {displayName}: {ex.Message}");
                            }
                        }
                    }
                }
            }

            // Refresh view to show highlights
            try
            {
                theSession.Parts.Work?.Views.Refresh();
            }
            catch { }
        }

        private void ClearHighlights()
        {
            foreach (var obj in highlightedObjects)
            {
                try
                {
                    obj.Unhighlight();
                }
                catch { }
            }
            highlightedObjects.Clear();

            try
            {
                theSession.Parts.Work?.Views.Refresh();
            }
            catch { }
        }

        public PropertyList GetBlockProperties(string blockID)
        {
            PropertyList plist = null;
            try
            {
                plist = theDialog.GetBlockProperties(blockID);
            }
            catch (Exception ex)
            {
                theUI.NXMessageBox.Show("GUID Sanitizer", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return plist;
        }
    }
}
