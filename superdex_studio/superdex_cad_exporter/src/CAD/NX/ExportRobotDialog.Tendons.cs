/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;

using NXOpen;

using CADRobotExporter.CAD.NX;
using CADRobotExporter.RobotDescription;

/// <summary>
/// NX Dialog for exporting robot models to URDF format.
/// Replicates the SolidWorks URDF Exporter experience.
/// </summary>
public partial class ExportRobotDialog
{
    // UI Blocks - Tendon Tab
    private NXOpen.BlockStyler.Tree treeControlTendons;
    private NXOpen.BlockStyler.Tree treeControlTendonRouting;
    private NXOpen.BlockStyler.StringBlock stringTendonName;
    private NXOpen.BlockStyler.Button buttonAddTendon;
    private NXOpen.BlockStyler.Button buttonAddRoutingElement;
    private NXOpen.BlockStyler.Toggle togglePointAutoAddRouting;
    private NXOpen.BlockStyler.Toggle toggleRoutingAutoSelectLink;
    private NXOpen.BlockStyler.DoubleBlock doubleTendonCoefficient;
    private NXOpen.BlockStyler.SelectObject pointSelectRouting;
    private NXOpen.BlockStyler.Enumeration enumTendonRoutingType;
    private NXOpen.BlockStyler.Enumeration enumTendonParentLink;
    private NXOpen.BlockStyler.Group groupTendons;
    private NXOpen.BlockStyler.Group groupRoutingElements;

    // Tendon state
    private List<Tendon> tendons = new List<Tendon>();
    private Tendon selectedTendon;
    private RoutingElement selectedRoutingElement;
    private int tendonCounter = 0;
    private Dictionary<NXOpen.BlockStyler.Node, Tendon> tendonNodeMap = new Dictionary<NXOpen.BlockStyler.Node, Tendon>();
    private Dictionary<NXOpen.BlockStyler.Node, RoutingElement> routingNodeMap = new Dictionary<NXOpen.BlockStyler.Node, RoutingElement>();
    private NXOpen.BlockStyler.UIBlock lastSelectedTendonBlock;

    private void InitializeTendonTreeColumns()
    {
        treeControlTendons.InsertColumn(0, "Tendon", 200);
        treeControlTendons.InsertColumn(1, "Elements", 80);

        treeControlTendonRouting.InsertColumn(0, "#", 30);
        treeControlTendonRouting.InsertColumn(1, "Type", 80);
        treeControlTendonRouting.InsertColumn(2, "Point", 60);
        treeControlTendonRouting.InsertColumn(3, "Parent Link", 120);
        treeControlTendonRouting.InsertColumn(4, "Parent Joint", 120);
        treeControlTendonRouting.InsertColumn(5, "Coefficient", 80);
    }

    private void PopulateTendonsFromList(List<Tendon> loadedTendons)
    {
        // Remove existing nodes from the tree control before repopulating
        foreach (var node in tendonNodeMap.Keys)
        {
            try { treeControlTendons.DeleteNode(node); }
            catch { }
        }

        tendons = loadedTendons;
        tendonNodeMap.Clear();
        routingNodeMap.Clear();
        selectedTendon = null;
        selectedRoutingElement = null;

        foreach (var tendon in tendons)
        {
            var node = treeControlTendons.CreateNode(tendon.Name);
            treeControlTendons.InsertNode(node, null, null, NXOpen.BlockStyler.Tree.NodeInsertOption.Last);
            node.SetColumnDisplayText(0, tendon.Name);
            node.SetColumnDisplayText(1, tendon.RoutingElements.Count.ToString());
            tendonNodeMap[node] = tendon;
        }

        SyncTendonCounter();
    }

    private void SyncTendonCounter()
    {
        tendonCounter = 0;
        foreach (var tendon in tendons)
        {
            string name = tendon.Name;
            int lastUnderscore = name.LastIndexOf('_');
            if (lastUnderscore >= 0 && int.TryParse(name.Substring(lastUnderscore + 1), out int num))
            {
                if (num > tendonCounter)
                    tendonCounter = num;
            }
        }
    }

    private void OnAddTendonClicked()
    {
        tendonCounter++;
        var tendon = new Tendon();
        tendon.Name = $"tendon_{tendonCounter}";
        tendons.Add(tendon);

        var node = treeControlTendons.CreateNode(tendon.Name);
        treeControlTendons.InsertNode(node, null, null, NXOpen.BlockStyler.Tree.NodeInsertOption.Last);
        node.SetColumnDisplayText(0, tendon.Name);
        node.SetColumnDisplayText(1, "0");
        tendonNodeMap[node] = tendon;

        treeControlTendons.SelectNode(node, true, true);
        SelectTendon(tendon);
    }

    private void SelectTendon(Tendon tendon)
    {
        selectedTendon = tendon;
        selectedRoutingElement = null;

        PopulateRoutingTree();
        PopulateParentLinkEnum(false);
        UpdateTendonTabUI();
    }

    private void PopulateRoutingTree()
    {
        // Clear existing routing tree
        routingNodeMap.Clear();
        var existingNodes = treeControlTendonRouting.GetSelectedNodes();
        // Delete all nodes by walking from root
        NXOpen.BlockStyler.Node rootRoutingNode = treeControlTendonRouting.RootNode;
        while (rootRoutingNode != null)
        {
            var next = rootRoutingNode.NextSiblingNode;
            treeControlTendonRouting.DeleteNode(rootRoutingNode);
            rootRoutingNode = next;
        }

        if (selectedTendon == null)
            return;

        for (int i = 0; i < selectedTendon.RoutingElements.Count; i++)
        {
            var element = selectedTendon.RoutingElements[i];
            AddRoutingNodeForElement(element, i);
        }
    }

    private NXOpen.BlockStyler.Node AddRoutingNodeForElement(RoutingElement element, int index)
    {
        var node = treeControlTendonRouting.CreateNode((index + 1).ToString());
        treeControlTendonRouting.InsertNode(node, null, null, NXOpen.BlockStyler.Tree.NodeInsertOption.Last);
        UpdateRoutingNodeColumns(node, element, index);
        routingNodeMap[node] = element;
        return node;
    }

    private void UpdateRoutingNodeColumns(NXOpen.BlockStyler.Node node, RoutingElement element, int index)
    {
        node.SetColumnDisplayText(0, (index + 1).ToString());
        string typeDisplay = element.Type == RoutingElement.TypeLinearJoint ? "LinearJoint" : "Waypoint";
        node.SetColumnDisplayText(1, typeDisplay);

        bool isSiteLink = IsLinkSite(element.Link);
        if (isSiteLink)
            node.SetColumnDisplayText(2, "(Site)");
        else
            node.SetColumnDisplayText(2, string.IsNullOrEmpty(element.PointKey) ? "-" : "\u2713");

        node.SetColumnDisplayText(3, string.IsNullOrEmpty(element.Link) ? "-" : element.Link);

        if (element.Type == RoutingElement.TypeLinearJoint)
        {
            string jointName = GetJointNameForLink(element.Link);
            node.SetColumnDisplayText(4, string.IsNullOrEmpty(jointName) ? "-" : jointName);
            node.SetColumnDisplayText(5, element.Coefficient.ToString("G"));
        }
        else
        {
            node.SetColumnDisplayText(4, "-");
            node.SetColumnDisplayText(5, "-");
        }
    }

    private void RenumberRoutingNodes()
    {
        NXOpen.BlockStyler.Node node = treeControlTendonRouting.RootNode;
        int index = 0;
        while (node != null)
        {
            if (routingNodeMap.TryGetValue(node, out var element))
            {
                UpdateRoutingNodeColumns(node, element, index);
                index++;
            }
            node = node.NextSiblingNode;
        }
    }

    private void UpdateTendonNodeElementCount(Tendon tendon)
    {
        foreach (var kvp in tendonNodeMap)
        {
            if (kvp.Value == tendon)
            {
                kvp.Key.SetColumnDisplayText(1, tendon.RoutingElements.Count.ToString());
                break;
            }
        }
    }

    private void OnTendonNameChanged()
    {
        if (selectedTendon == null)
            return;

        selectedTendon.Name = stringTendonName.Value;

        foreach (var kvp in tendonNodeMap)
        {
            if (kvp.Value == selectedTendon)
            {
                kvp.Key.SetColumnDisplayText(0, selectedTendon.Name);
                break;
            }
        }
    }

    private void OnStringTendonNameKeystroke(NXOpen.BlockStyler.StringBlock block, string text)
    {
    }

    private void OnAddRoutingElementClicked()
    {
        if (selectedTendon == null)
        {
            theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Warning,
                "Please select a tendon first.");
            return;
        }

        var element = new RoutingElement();
        element.Type = RoutingElement.TypeWaypoint;

        // Default parent link to first available
        string[] linkNames = GetAllLinkNames();
        if (linkNames.Length > 0)
            element.Link = StripJointSuffix(linkNames[0]);

        selectedTendon.AddRoutingElement(element);

        int index = selectedTendon.RoutingElements.Count - 1;
        var node = AddRoutingNodeForElement(element, index);

        UpdateTendonNodeElementCount(selectedTendon);

        treeControlTendonRouting.SelectNode(node, true, true);

        UpdateTendonTabUI();
    }

    private void CreateRoutingElementAndAddPoint()
    {
        if (selectedTendon == null)
        {
            theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Warning,
                "Please select a tendon first.");
            return;
        }

        var element = new RoutingElement();
        element.Type = RoutingElement.TypeWaypoint;

        // Default parent link to first available
        string[] linkNames = GetAllLinkNames();
        if (linkNames.Length > 0)
        {
            element.Link = StripJointSuffix(linkNames[0]);
        }

        try
        {
            TaggedObject[] selectedObjects = pointSelectRouting.GetSelectedObjects();
            if (selectedObjects == null || selectedObjects.Length == 0)
            {
                element.PointKey = null;
                return;
            }

            foreach (TaggedObject obj in selectedObjects)
            {
                if (obj is Point point)
                {
                    if (WaveLinker.IsFromDifferentPart(point) || !NXPersistentId.PointHasKey(point))
                    {
                        string waveName = $"Point - {selectedTendon.Name} - {selectedTendon.RoutingElements.Count + 1}";
                        var behavior = WaveLinker.ParseSelectionBehavior(enumSelectionBehavior.ValueAsString);
                        var crossPartResult = WaveLinker.HandleCrossPartSelection(
                            point, behavior,
                            p => WaveLinker.CreateWavePointLink(theSession.Parts.Work, p, waveName),
                            suppressOwnershipWarning);

                        if (crossPartResult.WarningShown)
                            ShowOwnershipWarning();

                        if (!crossPartResult.Allowed)
                        {
                            element.PointKey = null;
                            return;
                        }

                        if (crossPartResult.ResolvedObject != point)
                        {
                            point = crossPartResult.ResolvedObject;
                            pointSelectRouting.SetSelectedObjects(new TaggedObject[] { point });
                        }
                    }

                    string key = NXPersistentId.GetOrCreatePointKey(point);
                    element.PointKey = key;

                    if (toggleRoutingAutoSelectLink.Value)
                    {
                        string determinedLink = DetermineLinkForPoint(key, point);
                        if (determinedLink != null)
                        {
                            element.Link = determinedLink;
                        }
                    }
                    break;
                }
            }

        }
        catch (Exception ex)
        {
            logger.Warning($"OnRoutingPointChanged error: {ex.Message}");
        }

        selectedTendon.AddRoutingElement(element);

        int index = selectedTendon.RoutingElements.Count - 1;
        var node = AddRoutingNodeForElement(element, index);
        selectedRoutingElement = element;

        UpdateTendonNodeElementCount(selectedTendon);

        UpdateTendonTabUI();
    }

    private void UpdateTendonTabUI()
    {
        // NB: We do not support multi-select editing for now

        isUpdatingUI = true;

        CleanUpHighlighting();

        if (treeControlTendons == null || treeControlTendonRouting == null)
        {
            return;
        }

        NXOpen.BlockStyler.Node[] selectedTendonNodes = treeControlTendons.GetSelectedNodes();
        bool multiTendonSelect = selectedTendonNodes != null && selectedTendonNodes.Length > 1;

        NXOpen.BlockStyler.Node[] selectedRoutingNodes = treeControlTendonRouting.GetSelectedNodes();
        bool multiRoutingSelect = selectedRoutingNodes != null && selectedRoutingNodes.Length > 1;

        // No selected tendon or multi-select, early out
        if (selectedTendon == null || multiTendonSelect)
        {
            stringTendonName.Enable = false;
            groupRoutingElements.Enable = false;

            isUpdatingUI = false;
            return;
        }

        groupRoutingElements.Enable = true;
        stringTendonName.Enable = true;
        stringTendonName.Value = selectedTendon.Name;

        // No selected routing element or multi-select
        if (selectedRoutingElement == null || multiRoutingSelect)
        {
            enumTendonRoutingType.Enable = false;
            enumTendonParentLink.Enable = false;
            doubleTendonCoefficient.Enable = false;

            // but leave point selection on if auto-add enabled
            pointSelectRouting.Enable = togglePointAutoAddRouting.Value;
            pointSelectRouting.SetSelectedObjects(new TaggedObject[0]);

            if (lastSelectedTendonBlock != null)
            {
                lastSelectedTendonBlock.Focus();
            }

            isUpdatingUI = false;
            return;
        }

        // Routing elements
        enumTendonRoutingType.Enable = true;
        enumTendonParentLink.Enable = true;
        bool isWaypoint = selectedRoutingElement.Type == RoutingElement.TypeWaypoint;
        bool isSite = IsLinkSite(selectedRoutingElement.Link);
        pointSelectRouting.Enable = !isSite && (isWaypoint || togglePointAutoAddRouting.Value);
        doubleTendonCoefficient.Enable = !isWaypoint;

        if (selectedRoutingElement.Type == RoutingElement.TypeLinearJoint)
        {
            enumTendonRoutingType.ValueAsString = "LinearJoint";
            PopulateParentLinkEnum(true);
        }
        else
        {
            enumTendonRoutingType.ValueAsString = "Waypoint";
            PopulateParentLinkEnum(false);
        }

        if (!string.IsNullOrEmpty(selectedRoutingElement.Link))
        {
            try { SetParentLinkEnumForLink(selectedRoutingElement.Link); }
            catch { }
        }

        doubleTendonCoefficient.Value = selectedRoutingElement.Coefficient;

        pointSelectRouting.SetSelectedObjects(new TaggedObject[0]);
        if (!string.IsNullOrEmpty(selectedRoutingElement.PointKey) && !IsLinkSite(selectedRoutingElement.Link))
        {
            try
            {
                Point point = NXPersistentId.FindPointByKey(
                    theSession.Parts.Work,
                    selectedRoutingElement.PointKey);
                if (point != null)
                {
                    pointSelectRouting.SetSelectedObjects(new TaggedObject[] { point });
                }
            }
            catch { }
        }

        if (lastSelectedTendonBlock != null)
        {
            lastSelectedTendonBlock.Focus();
        }

        isUpdatingUI = false;
    }

    private void SetParentLinkEnumForLink(string linkName)
    {
        string[] members = enumTendonParentLink.GetEnumMembers();
        foreach (string member in members)
        {
            if (StripJointSuffix(member) == linkName)
            {
                enumTendonParentLink.ValueAsString = member;
                return;
            }
        }
        try { enumTendonParentLink.ValueAsString = linkName; }
        catch { }
    }

    private void OnRoutingTypeChanged()
    {
        if (selectedRoutingElement == null)
            return;

        string val = enumTendonRoutingType.ValueAsString;
        if (val == "LinearJoint")
        {
            selectedRoutingElement.Type = RoutingElement.TypeLinearJoint;
            selectedRoutingElement.PointKey = null;
            selectedRoutingElement.SetPosition(null);
            PopulateParentLinkEnum(true);
        }
        else
        {
            selectedRoutingElement.Type = RoutingElement.TypeWaypoint;
            PopulateParentLinkEnum(false);
        }

        UpdateTendonTabUI();
        UpdateRoutingColumnsForSelected();
    }

    private void OnCoefficientChanged()
    {
        if (selectedRoutingElement == null)
            return;

        selectedRoutingElement.Coefficient = doubleTendonCoefficient.Value;

        UpdateTendonTabUI();
        UpdateRoutingColumnsForSelected();
    }

    private void OnParentLinkChanged()
    {
        if (selectedRoutingElement == null)
            return;

        string rawValue = enumTendonParentLink.ValueAsString;
        string linkName = StripJointSuffix(rawValue);
        selectedRoutingElement.Link = linkName;

        if (IsLinkSite(linkName))
        {
            selectedRoutingElement.PointKey = null;
            selectedRoutingElement.SetPosition(new double[] { 0, 0, 0 });
        }

        UpdateTendonTabUI();
        UpdateRoutingColumnsForSelected();
    }

    private string StripJointSuffix(string enumValue)
    {
        if (string.IsNullOrEmpty(enumValue))
            return enumValue;
        int parenIndex = enumValue.IndexOf(" (");
        if (parenIndex > 0)
            return enumValue.Substring(0, parenIndex);
        return enumValue;
    }

    private void OnRoutingPointChanged()
    {
        if (selectedTendon == null)
        {
            return;
        }

        // Auto-add if toggle is checked
        if (togglePointAutoAddRouting.Value)
        {
            CreateRoutingElementAndAddPoint();
            return;
        }
        else if (selectedRoutingElement == null)
        {
            return;
        }

        try
        {
            TaggedObject[] selectedObjects = pointSelectRouting.GetSelectedObjects();
            if (selectedObjects == null || selectedObjects.Length == 0)
            {
                if (toggleDeselectGuardDatums.Value && selectedRoutingElement.PointKey != null)
                {
                    if (!ConfirmDeselection("Routing Point"))
                    {
                        UpdateTendonTabUI();
                        return;
                    }
                }

                selectedRoutingElement.PointKey = null;
                selectedRoutingElement.SetPosition(null);

                UpdateRoutingColumnsForSelected();
                return;
            }

            foreach (TaggedObject obj in selectedObjects)
            {
                if (obj is Point point)
                {
                    if (WaveLinker.IsFromDifferentPart(point))
                    {
                        int waypointNum = selectedTendon.RoutingElements.IndexOf(selectedRoutingElement) + 1;
                        string waveName = $"Point - {selectedTendon.Name} - {waypointNum}";
                        var behavior = WaveLinker.ParseSelectionBehavior(enumSelectionBehavior.ValueAsString);
                        var crossPartResult = WaveLinker.HandleCrossPartSelection(
                            point, behavior,
                            p => WaveLinker.CreateWavePointLink(theSession.Parts.Work, p, waveName),
                            suppressOwnershipWarning);

                        if (crossPartResult.WarningShown)
                            ShowOwnershipWarning();

                        if (!crossPartResult.Allowed)
                        {
                            selectedRoutingElement.PointKey = null;
                            selectedRoutingElement.SetPosition(null);
                            UpdateRoutingColumnsForSelected();
                            return;
                        }

                        if (crossPartResult.ResolvedObject != point)
                        {
                            point = crossPartResult.ResolvedObject;
                            pointSelectRouting.SetSelectedObjects(new TaggedObject[] { point });
                        }
                    }

                    string key = NXPersistentId.GetOrCreatePointKey(point);
                    selectedRoutingElement.PointKey = key;

                    if (toggleRoutingAutoSelectLink.Value)
                    {
                        string determinedLink = DetermineLinkForPoint(key, point);
                        if (determinedLink != null)
                        {
                            selectedRoutingElement.Link = determinedLink;
                            SetParentLinkEnumForLink(determinedLink);
                        }
                    }
                    break;
                }
            }

            UpdateRoutingColumnsForSelected();
        }
        catch (Exception ex)
        {
            logger.Warning($"OnRoutingPointChanged error: {ex.Message}");
        }
    }

    private void UpdateRoutingColumnsForSelected()
    {
        if (selectedRoutingElement == null)
            return;

        foreach (var kvp in routingNodeMap)
        {
            if (kvp.Value == selectedRoutingElement)
            {
                int index = selectedTendon.RoutingElements.IndexOf(selectedRoutingElement);
                UpdateRoutingNodeColumns(kvp.Key, selectedRoutingElement, index);
                break;
            }
        }
    }

    private void BuildComponentToLinkCache()
    {
        componentToLinkCache = new Dictionary<string, string>();
        var allNodes = treeManager.CollectAllNodes();

        foreach (var node in allNodes)
        {
            string linkName = node.LinkName;
            if (string.IsNullOrEmpty(linkName))
                continue;

            // Process in priority order: inertial, collision, visual
            AddHandlesToCache(node.InertialBodiesHandles, linkName);
            AddHandlesToCache(node.CollisionBodiesHandles, linkName);
            AddHandlesToCache(node.VisualBodiesHandles, linkName);
        }
    }

    private void AddHandlesToCache(string[] handles, string linkName)
    {
        if (handles == null)
            return;

        foreach (string handle in handles)
        {
            if (string.IsNullOrEmpty(handle))
                continue;

            string componentPath;
            if (NXPersistentId.IsComponentKey(handle))
            {
                componentPath = handle.Substring("COMPONENT:".Length);
            }
            else
            {
                (componentPath, _) = NXPersistentId.ParseCompositeKey(handle);
            }

            if (componentPath == null)
                continue;

            string leafGuid = GetLeafComponentGuid(componentPath);
            if (leafGuid != null && !componentToLinkCache.ContainsKey(leafGuid))
            {
                componentToLinkCache[leafGuid] = linkName;
            }
        }
    }

    private string DetermineLinkForPoint(string pointKey, Point point = null)
    {
        if (componentToLinkCache == null)
            BuildComponentToLinkCache();

        if (!string.IsNullOrEmpty(pointKey))
        {
            (string componentPath, _) = NXPersistentId.ParseCompositeKey(pointKey);
            if (componentPath != null)
            {
                string leafGuid = GetLeafComponentGuid(componentPath);
                if (leafGuid != null && componentToLinkCache.TryGetValue(leafGuid, out string linkName))
                    return linkName;
            }
        }

        // Fallback: trace through WAVE link feature to find source component
        if (point != null)
        {
            var sourceComponent = WaveLinker.GetSourceComponentFromWaveLink(point);
            if (sourceComponent != null)
                return DetermineLinkForComponent(sourceComponent);
        }

        return null;
    }

    private string DetermineLinkForComponent(NXOpen.Assemblies.Component component)
    {
        if (component == null)
            return null;

        if (componentToLinkCache == null)
            BuildComponentToLinkCache();

        string componentPath = NXPersistentId.GetComponentPath(component);
        string leafGuid = GetLeafComponentGuid(componentPath);
        if (leafGuid != null && componentToLinkCache.TryGetValue(leafGuid, out string linkName))
            return linkName;

        return null;
    }

    private static string GetLeafComponentGuid(string componentPath)
    {
        if (string.IsNullOrEmpty(componentPath))
            return null;
        int lastSlash = componentPath.LastIndexOf('/');
        return lastSlash >= 0 ? componentPath.Substring(lastSlash + 1) : componentPath;
    }

    private void PopulateParentLinkEnum(bool jointOnly)
    {
        string[] linkNames = jointOnly ? GetJointLinkNamesWithJoint() : GetAllLinkNames();
        if (linkNames.Length == 0)
            linkNames = new string[] { "" };

        enumTendonParentLink.SetEnumMembers(linkNames);
    }

    private string[] GetAllLinkNames()
    {
        var names = new List<string>();
        var allNodes = treeManager.CollectAllNodes();
        foreach (var node in allNodes)
        {
            if (!string.IsNullOrEmpty(node.LinkName))
            {
                if (!node.IsRootNode && !string.IsNullOrEmpty(node.JointName))
                    names.Add($"{node.LinkName} ({node.JointName})");
                else
                    names.Add(node.LinkName);
            }
        }
        return names.ToArray();
    }

    private string[] GetJointLinkNames()
    {
        var names = new List<string>();
        var allNodes = treeManager.CollectAllNodes();
        foreach (var node in allNodes)
        {
            if (node.IsRootNode)
                continue;
            string jt = node.JointType?.ToLower() ?? "";
            if (jt == "revolute" || jt == "prismatic")
            {
                if (!string.IsNullOrEmpty(node.LinkName))
                    names.Add(node.LinkName);
            }
        }
        return names.ToArray();
    }

    private string[] GetJointLinkNamesWithJoint()
    {
        var names = new List<string>();
        var allNodes = treeManager.CollectAllNodes();
        foreach (var node in allNodes)
        {
            if (node.IsRootNode)
                continue;
            string jt = node.JointType?.ToLower() ?? "";
            if (jt == "revolute" || jt == "prismatic")
            {
                if (!string.IsNullOrEmpty(node.LinkName))
                {
                    string jointLabel = string.IsNullOrEmpty(node.JointName)
                        ? node.LinkName
                        : $"{node.LinkName} ({node.JointName})";
                    names.Add(jointLabel);
                }
            }
        }
        return names.ToArray();
    }

    private string GetJointNameForLink(string linkName)
    {
        if (string.IsNullOrEmpty(linkName))
            return null;
        var allNodes = treeManager.CollectAllNodes();
        foreach (var node in allNodes)
        {
            if (node.LinkName == linkName)
                return node.JointName;
        }
        return null;
    }

    private bool IsLinkSite(string linkName)
    {
        if (string.IsNullOrEmpty(linkName))
            return false;
        var allNodes = treeManager.CollectAllNodes();
        foreach (var node in allNodes)
        {
            if (node.LinkName == linkName)
                return node.IsSite;
        }
        return false;
    }

    // Tendon tree callbacks
    private void OnTendonSelectCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID, bool selected)
    {
        if (!selected)
        {
            NXOpen.BlockStyler.Node[] selectedNodes = treeControlTendons.GetSelectedNodes();
            if (selectedNodes == null || selectedNodes.Length == 0)
            {
                selectedTendon = null;
                selectedRoutingElement = null;
                UpdateTendonTabUI();
            }
            return;
        }

        NXOpen.BlockStyler.Node[] selNodes = treeControlTendons.GetSelectedNodes();
        if (selNodes != null && selNodes.Length == 1)
        {
            if (tendonNodeMap.TryGetValue(selNodes[0], out Tendon tendon))
            {
                SelectTendon(tendon);
            }
        }
    }

    private void OnTendonMenuCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID)
    {
        var menu = tree.CreateMenu();
        menu.AddMenuItem(TendonMenuRemove, "Remove Tendon(s)");
        tree.SetMenu(menu);
        menu.Dispose();
    }

    private void OnTendonMenuSelectionCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int menuItemID)
    {
        if (menuItemID == TendonMenuRemove)
        {
            RemoveSelectedTendons();
        }
    }

    private void RemoveSelectedTendons()
    {
        NXOpen.BlockStyler.Node[] selectedNodes = treeControlTendons.GetSelectedNodes();
        if (selectedNodes == null || selectedNodes.Length == 0)
            return;

        foreach (var node in selectedNodes)
        {
            if (tendonNodeMap.TryGetValue(node, out Tendon tendon))
            {
                tendons.Remove(tendon);
                tendonNodeMap.Remove(node);
                try { treeControlTendons.DeleteNode(node); }
                catch { }
            }
        }

        // Clear routing if selected tendon was removed
        if (selectedTendon != null && !tendons.Contains(selectedTendon))
        {
            selectedTendon = null;
            selectedRoutingElement = null;
            stringTendonName.Value = "";
            PopulateRoutingTree();
            UpdateTendonTabUI();
        }
    }

    private NXOpen.BlockStyler.Tree.BeginLabelEditState OnTendonBeginLabelEditCallback(
        NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID)
    {
        if (columnID == 0)
            return NXOpen.BlockStyler.Tree.BeginLabelEditState.Allow;
        return NXOpen.BlockStyler.Tree.BeginLabelEditState.Disallow;
    }

    private NXOpen.BlockStyler.Tree.EndLabelEditState OnTendonEndLabelEditCallback(
        NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID, string editedText)
    {
        if (string.IsNullOrWhiteSpace(editedText))
            return NXOpen.BlockStyler.Tree.EndLabelEditState.RejectText;

        if (tendonNodeMap.TryGetValue(node, out Tendon tendon))
        {
            tendon.Name = editedText;
            node.SetColumnDisplayText(0, editedText);
            if (tendon == selectedTendon)
            {
                isUpdatingUI = true;
                stringTendonName.Value = editedText;
                isUpdatingUI = false;
            }
        }
        return NXOpen.BlockStyler.Tree.EndLabelEditState.AcceptText;
    }

    // Routing element tree callbacks
    private void OnRoutingSelectCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID, bool selected)
    {
        if (!selected)
        {
            NXOpen.BlockStyler.Node[] selectedNodes = treeControlTendonRouting.GetSelectedNodes();
            if (selectedNodes == null || selectedNodes.Length == 0)
            {
                selectedRoutingElement = null;
            }

            UpdateTendonTabUI();
            return;
        }

        NXOpen.BlockStyler.Node[] selNodes = treeControlTendonRouting.GetSelectedNodes();
        if (selNodes != null && selNodes.Length == 1)
        {
            if (routingNodeMap.TryGetValue(selNodes[0], out RoutingElement element))
            {
                selectedRoutingElement = element;
            }
        }
        if (selNodes.Length > 1)
        {
            selectedRoutingElement = null;
        }

        UpdateTendonTabUI();
    }

    private void OnRoutingMenuCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID)
    {
        var menu = tree.CreateMenu();
        menu.AddMenuItem(RoutingMenuRemove, "Remove Element(s)");
        tree.SetMenu(menu);
        menu.Dispose();
    }

    private void OnRoutingMenuSelectionCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int menuItemID)
    {
        if (menuItemID == RoutingMenuRemove)
        {
            RemoveSelectedRoutingElements();
        }
    }

    private void RemoveSelectedRoutingElements()
    {
        if (selectedTendon == null)
            return;

        NXOpen.BlockStyler.Node[] selectedNodes = treeControlTendonRouting.GetSelectedNodes();
        if (selectedNodes == null || selectedNodes.Length == 0)
            return;

        foreach (var node in selectedNodes)
        {
            if (routingNodeMap.TryGetValue(node, out RoutingElement element))
            {
                selectedTendon.RemoveRoutingElement(element);
                routingNodeMap.Remove(node);
                try { treeControlTendonRouting.DeleteNode(node); }
                catch { }
            }
        }

        if (selectedRoutingElement != null && !selectedTendon.RoutingElements.Contains(selectedRoutingElement))
        {
            selectedRoutingElement = null;
        }

        RenumberRoutingNodes();
        UpdateTendonNodeElementCount(selectedTendon);
        UpdateTendonTabUI();
    }

    private NXOpen.BlockStyler.Node.DragType TendonIsDragAllowedCallback(
        NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID)
    {
        if (node == null)
            return NXOpen.BlockStyler.Node.DragType.None;
        return NXOpen.BlockStyler.Node.DragType.All;
    }

    private NXOpen.BlockStyler.Node.DropType TendonIsDropAllowedCallback(
        NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID,
        NXOpen.BlockStyler.Node targetNode, int targetColumnID)
    {
        if (targetNode == null || node == null || node == targetNode)
            return NXOpen.BlockStyler.Node.DropType.None;

        return NXOpen.BlockStyler.Node.DropType.Before;
    }

    private bool TendonOnDropCallback(
        NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node[] nodes, int columnID,
        NXOpen.BlockStyler.Node targetNode, int targetColumnID,
        NXOpen.BlockStyler.Node.DropType dropType, int dropMenuItemId)
    {
        if (nodes == null || nodes.Length == 0 || targetNode == null)
            return false;

        if (!tendonNodeMap.TryGetValue(nodes[0], out Tendon draggedTendon))
            return false;
        if (!tendonNodeMap.TryGetValue(targetNode, out Tendon targetTendon))
            return false;

        int oldIndex = tendons.IndexOf(draggedTendon);
        int targetIndex = tendons.IndexOf(targetTendon);

        if (oldIndex < 0 || targetIndex < 0)
            return false;

        tendons.RemoveAt(oldIndex);

        int newIndex = tendons.IndexOf(targetTendon);
        if (newIndex < 0) newIndex = 0;

        tendons.Insert(newIndex, draggedTendon);

        var copiedNode = treeControlTendons.CopyNode(nodes[0]);
        tendonNodeMap[copiedNode] = draggedTendon;

        NXOpen.BlockStyler.Node previousSibling = targetNode.PreviousSiblingNode;
        if (previousSibling == null || previousSibling == nodes[0])
        {
            treeControlTendons.InsertNode(copiedNode, null, null, NXOpen.BlockStyler.Tree.NodeInsertOption.First);
        }
        else
        {
            treeControlTendons.InsertNode(copiedNode, null, previousSibling, NXOpen.BlockStyler.Tree.NodeInsertOption.Last);
        }

        tendonNodeMap.Remove(nodes[0]);
        treeControlTendons.DeleteNode(nodes[0]);

        treeControlTendons.SelectNode(copiedNode, true, true);
        UpdateTendonNodeElementCount(draggedTendon);

        return true;
    }

    private NXOpen.BlockStyler.Node.DragType RoutingIsDragAllowedCallback(
        NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID)
    {
        if (node == null)
            return NXOpen.BlockStyler.Node.DragType.None;
        return NXOpen.BlockStyler.Node.DragType.All;
    }

    private NXOpen.BlockStyler.Node.DropType RoutingIsDropAllowedCallback(
        NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID,
        NXOpen.BlockStyler.Node targetNode, int targetColumnID)
    {
        if (targetNode == null || node == null || node == targetNode)
            return NXOpen.BlockStyler.Node.DropType.None;

        return NXOpen.BlockStyler.Node.DropType.Before;
    }

    private bool RoutingOnDropCallback(
        NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node[] nodes, int columnID,
        NXOpen.BlockStyler.Node targetNode, int targetColumnID,
        NXOpen.BlockStyler.Node.DropType dropType, int dropMenuItemId)
    {
        if (selectedTendon == null || nodes == null || nodes.Length == 0 || targetNode == null)
            return false;

        if (!routingNodeMap.TryGetValue(nodes[0], out RoutingElement draggedElement))
            return false;
        if (!routingNodeMap.TryGetValue(targetNode, out RoutingElement targetElement))
            return false;

        // Reorder in data model: insert before the target
        int oldIndex = selectedTendon.RoutingElements.IndexOf(draggedElement);
        int targetIndex = selectedTendon.RoutingElements.IndexOf(targetElement);

        if (oldIndex < 0 || targetIndex < 0)
            return false;

        selectedTendon.RoutingElements.RemoveAt(oldIndex);

        int newIndex = selectedTendon.RoutingElements.IndexOf(targetElement);
        if (newIndex < 0) newIndex = 0;

        selectedTendon.RoutingElements.Insert(newIndex, draggedElement);

        // Move tree node: copy, insert at new position, delete old
        var copiedNode = treeControlTendonRouting.CopyNode(nodes[0]);
        routingNodeMap[copiedNode] = draggedElement;

        // Insert before target: find target's previous sibling
        NXOpen.BlockStyler.Node previousSibling = targetNode.PreviousSiblingNode;
        if (previousSibling == null || previousSibling == nodes[0])
        {
            treeControlTendonRouting.InsertNode(copiedNode, null, null, NXOpen.BlockStyler.Tree.NodeInsertOption.First);
        }
        else
        {
            treeControlTendonRouting.InsertNode(copiedNode, null, previousSibling, NXOpen.BlockStyler.Tree.NodeInsertOption.Last);
        }

        routingNodeMap.Remove(nodes[0]);
        treeControlTendonRouting.DeleteNode(nodes[0]);

        RenumberRoutingNodes();
        treeControlTendonRouting.SelectNode(copiedNode, true, true);

        return true;
    }
}
