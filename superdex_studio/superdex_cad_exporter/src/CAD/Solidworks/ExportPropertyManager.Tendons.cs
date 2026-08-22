/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if SOLIDWORKS

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

using CADRobotExporter.RobotDescription;
using CADRobotExporter.UI;

namespace CADRobotExporter.CAD
{
    public sealed partial class ExportPropertyManager
    {
        #region Tendon fields

        private List<Tendon> tendons = new List<Tendon>();
        private Tendon selectedTendon;
        private RoutingElement selectedRoutingElement;
        private int tendonCounter = 0;

        private ListView tendonListView;
        private ListView routingListView;

        private bool isUpdatingTendonUI = false;

        private PropertyManagerPageGroup PMGroupTendons;
        private PropertyManagerPageGroup PMGroupRouting;
        private PropertyManagerPageTextbox PMTextBoxTendonName;
        private PropertyManagerPageSelectionbox PMSelectionBoxPoint;
        private PropertyManagerPageNumberbox PMNumberBoxCoefficient;
        private PropertyManagerPageCombobox PMComboElementType;
        private PropertyManagerPageCombobox PMComboParentLink;
        private PropertyManagerPageCheckbox PMCheckboxAutoAddWaypoint;
        private PropertyManagerPageCheckbox PMCheckboxAutoLink;
        private PropertyManagerPageWindowFromHandle PMWindowTendonList;
        private PropertyManagerPageWindowFromHandle PMWindowRoutingTable;
        private PropertyManagerPageButton PMButtonAddTendon;
        private PropertyManagerPageButton PMButtonRemoveTendon;
        private PropertyManagerPageButton PMButtonAddRouting;
        private PropertyManagerPageButton PMButtonRemoveRouting;
        private PropertyManagerPageLabel PMLabelTendonName;
        private PropertyManagerPageLabel PMLabelPointSelection;
        private PropertyManagerPageLabel PMLabelCoefficient;
        private PropertyManagerPageLabel PMLabelElementType;
        private PropertyManagerPageLabel PMLabelTendonParentLink;

        #endregion

        #region Tendon UI Setup

        private void SetupTendonTab(PropertyManagerPageTab tab)
        {
            string caption;
            string tip;
            long options;
            int controlType;
            int alignment;
            string[] icons;

            // Tendons group
            caption = "Tendons";
            options = (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Visible +
                (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Expanded;
            PMGroupTendons = (PropertyManagerPageGroup)tab.AddGroupBox(GroupTendonsID, caption, (int)options);

            // Embedded ListView for tendons
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_WindowFromHandle;
            caption = "Tendon List";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMWindowTendonList = PMGroupTendons.AddControl2(WindowTendonListID,
                (short)swPropertyManagerPageControlType_e.swControlType_WindowFromHandle,
                caption, (short)alignment, (int)options, "");
            PMWindowTendonList.Height = 120;

            tendonListView = new ListView
            {
                View = System.Windows.Forms.View.Details,
                FullRowSelect = true,
                MultiSelect = true,
                Height = 120,
                Visible = true,
                Font = SystemFonts.MenuFont,
                HideSelection = false
            };
            tendonListView.Columns.Add("Tendon", 160);
            tendonListView.Columns.Add("Elements", 120);
            tendonListView.SelectedIndexChanged += TendonListView_SelectedIndexChanged;
            PMWindowTendonList.SetWindowHandlex64(tendonListView.Handle.ToInt64());

            // Tendon name label + textbox
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Tendon Name";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelTendonName = PMGroupTendons.AddControl2(LabelTendonNameID, (short)controlType, caption,
                (short)alignment, (int)options, "");

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Textbox;
            caption = "";
            tip = "Enter the name of the tendon";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMTextBoxTendonName = (PropertyManagerPageTextbox)PMGroupTendons.AddControl2(
                TextBoxTendonNameID, (short)controlType, caption, (short)alignment, (int)options, tip);

            // Add Tendon button
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Button;
            caption = "Add Tendon";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Add a new tendon";
            PMButtonAddTendon = PMGroupTendons.AddControl2(ButtonAddTendonID, (short)controlType, caption,
                (short)alignment, (int)options, tip);

            // Remove Tendon button
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Button;
            caption = "Remove Tendon";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Remove selected tendon";
            PMButtonRemoveTendon = PMGroupTendons.AddControl2(ButtonRemoveTendonID, (short)controlType, caption,
                (short)alignment, (int)options, tip);

            // Routing Elements group
            caption = "Routing Elements";
            options = (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Visible +
                (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Expanded;
            PMGroupRouting = (PropertyManagerPageGroup)tab.AddGroupBox(GroupRoutingID, caption, (int)options);

            // Add Routing Element button
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Button;
            caption = "Add Routing Element";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Add a new routing element to the selected tendon";
            PMButtonAddRouting = PMGroupRouting.AddControl2(ButtonAddRoutingID, (short)controlType, caption,
                (short)alignment, (int)options, tip);

            // Remove Routing Element button
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Button;
            caption = "Remove Element";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Remove selected routing element";
            PMButtonRemoveRouting = PMGroupRouting.AddControl2(ButtonRemoveRoutingID, (short)controlType, caption,
                (short)alignment, (int)options, tip);

            // Auto-add waypoint checkbox
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            caption = "Automatically add new Waypoints when selecting Points";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "When checked, selecting a point will automatically create a new routing element";
            PMCheckboxAutoAddWaypoint = (PropertyManagerPageCheckbox)PMGroupRouting.AddControl2(
                CheckboxAutoAddWaypointID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMCheckboxAutoAddWaypoint.Checked = true;

            // Auto-determine link checkbox
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            caption = "Automatically determine Link belonging to Point";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "When checked, the parent link will be determined from the component owning the point";
            PMCheckboxAutoLink = (PropertyManagerPageCheckbox)PMGroupRouting.AddControl2(
                CheckboxAutoLinkID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMCheckboxAutoLink.Checked = true;

            // Point selection label + selection box
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Point Selection";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelPointSelection = PMGroupRouting.AddControl2(LabelPointSelectionID, (short)controlType, caption,
                (short)alignment, (int)options, "");

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Selectionbox;
            caption = "Waypoint Point";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Select a reference point or sketch point for the waypoint position";
            PMSelectionBoxPoint = (PropertyManagerPageSelectionbox)PMGroupRouting.AddControl2(
                SelectionBoxPointID, (short)controlType, caption, (short)alignment, (int)options, tip);

            swSelectType_e[] pointFilters = new swSelectType_e[]
            {
                swSelectType_e.swSelDATUMPOINTS,
            };
            object pointFilterObj = pointFilters;
            PMSelectionBoxPoint.AllowSelectInMultipleBoxes = false;
            PMSelectionBoxPoint.SingleEntityOnly = true;
            PMSelectionBoxPoint.AllowMultipleSelectOfSameEntity = false;
            PMSelectionBoxPoint.Mark = 64;
            PMSelectionBoxPoint.SetSelectionFilters(pointFilterObj);
            PMSelectionBoxPoint.SetSelectionColor(true, (int)swUserPreferenceIntegerValue_e.swSystemColorsSelectedItem6);
            icons = CreateTemporaryIcons("point");
            ((PropertyManagerPageControl)PMSelectionBoxPoint).SetPictureLabelByName(icons[1], "");

            // Coefficient label + number box
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Coefficient";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelCoefficient = PMGroupRouting.AddControl2(LabelCoefficientID, (short)controlType, caption,
                (short)alignment, (int)options, "");

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Numberbox;
            caption = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            tip = "Linear joint coefficient";
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMNumberBoxCoefficient = (PropertyManagerPageNumberbox)PMGroupRouting.AddControl2(
                NumberBoxCoefficientID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMNumberBoxCoefficient.SetRange2(
                (int)swNumberboxUnitType_e.swNumberBox_UnitlessDouble, -1000, 1000, true, 0.01, 0.1, 0.001);
            PMNumberBoxCoefficient.Value = 0;

            // Element Type label + combobox
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Element Type";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelElementType = PMGroupRouting.AddControl2(LabelElementTypeID, (short)controlType, caption,
                (short)alignment, (int)options, "");

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Combobox;
            caption = "";
            tip = "Select the routing element type";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMComboElementType = (PropertyManagerPageCombobox)PMGroupRouting.AddControl2(
                ComboElementTypeID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMComboElementType.AddItems(new string[] { "Waypoint", "Linear Joint" });
            PMComboElementType.CurrentSelection = 0;
            PMComboElementType.Height = 35;

            // Parent Link label + combobox
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Parent Link";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelTendonParentLink = PMGroupRouting.AddControl2(LabelTendonParentLinkID, (short)controlType, caption,
                (short)alignment, (int)options, "");

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Combobox;
            caption = "";
            tip = "Select the parent link for this routing element";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMComboParentLink = (PropertyManagerPageCombobox)PMGroupRouting.AddControl2(
                ComboParentLinkID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMComboParentLink.Height = 35;

            // Embedded ListView for routing elements
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_WindowFromHandle;
            caption = "Routing Table";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMWindowRoutingTable = PMGroupRouting.AddControl2(WindowRoutingTableID,
                (short)swPropertyManagerPageControlType_e.swControlType_WindowFromHandle,
                caption, (short)alignment, (int)options, "");
            PMWindowRoutingTable.Height = 140;

            routingListView = new ListView
            {
                View = System.Windows.Forms.View.Details,
                FullRowSelect = true,
                MultiSelect = true,
                Height = 140,
                Visible = true,
                Font = SystemFonts.MenuFont,
                HideSelection = false
            };
            routingListView.Columns.Add("#", 30);
            routingListView.Columns.Add("Type", 120);
            routingListView.Columns.Add("Point", 60);
            routingListView.Columns.Add("Parent Link", 180);
            routingListView.Columns.Add("Coefficient", 110);
            routingListView.SelectedIndexChanged += RoutingListView_SelectedIndexChanged;
            PMWindowRoutingTable.SetWindowHandlex64(routingListView.Handle.ToInt64());

            UpdateTendonUIState();
        }

        #endregion

        #region Tendon Operations

        private void OnAddTendonClicked()
        {
            isUpdatingTendonUI = true;
            try
            {
                tendonCounter++;
                var tendon = new Tendon();
                tendon.Name = "tendon_" + tendonCounter;
                tendons.Add(tendon);

                var item = new ListViewItem(new[] { tendon.Name, "0" });
                item.Tag = tendon;
                tendonListView.Items.Add(item);
                tendonListView.SelectedItems.Clear();
                item.Selected = true;

                selectedTendon = tendon;
                selectedRoutingElement = null;
                PMTextBoxTendonName.Text = tendon.Name;
                PopulateRoutingGrid();
            }
            finally
            {
                isUpdatingTendonUI = false;
            }
        }

        private void OnRemoveTendonClicked()
        {
            if (tendonListView.SelectedItems.Count == 0) return;

            isUpdatingTendonUI = true;
            try
            {
                tendonListView.BeginUpdate();
                foreach (ListViewItem item in new List<ListViewItem>(tendonListView.SelectedItems.Cast<ListViewItem>()))
                {
                    var tendon = item.Tag as Tendon;
                    if (tendon != null)
                        tendons.Remove(tendon);
                    tendonListView.Items.Remove(item);
                }
                tendonListView.EndUpdate();

                selectedTendon = null;
                selectedRoutingElement = null;
                routingListView.BeginUpdate();
                routingListView.Items.Clear();
                routingListView.EndUpdate();
                PMTextBoxTendonName.Text = "";
            }
            finally
            {
                isUpdatingTendonUI = false;
            }
        }

        private void SelectTendon(Tendon tendon)
        {
            isUpdatingTendonUI = true;
            try
            {
                selectedTendon = tendon;
                selectedRoutingElement = null;
                PMTextBoxTendonName.Text = tendon.Name;
                PopulateRoutingGrid();
                UpdateTendonUIState();
            }
            finally
            {
                isUpdatingTendonUI = false;
            }
        }

        private void OnAddRoutingElementClicked()
        {
            if (selectedTendon == null) return;

            isUpdatingTendonUI = true;
            try
            {
                EnsureParentLinkComboPopulated();

                var element = new RoutingElement();
                element.Type = RoutingElement.TypeWaypoint;

                // Default link to first available
                List<string> linkNames = GetAllLinkNames();
                if (linkNames.Count > 0)
                {
                    element.Link = linkNames[0];
                }

                selectedTendon.AddRoutingElement(element);
                AddRoutingRow(element, selectedTendon.RoutingElements.Count);
                UpdateTendonElementCount();

                // Select the new element and update PM controls
                selectedRoutingElement = element;
                PMComboElementType.CurrentSelection = 0;
                PMNumberBoxCoefficient.Value = element.Coefficient;

                int linkIdx = cachedLinkNames.IndexOf(element.Link);
                if (linkIdx >= 0)
                    PMComboParentLink.CurrentSelection = (short)linkIdx;

                // Select the row in the ListView
                if (routingListView.Items.Count > 0)
                {
                    routingListView.SelectedItems.Clear();
                    routingListView.Items[routingListView.Items.Count - 1].Selected = true;
                }
            }
            finally
            {
                isUpdatingTendonUI = false;
            }
        }

        private void OnRemoveRoutingElementClicked()
        {
            if (selectedTendon == null || routingListView.SelectedItems.Count == 0) return;

            isUpdatingTendonUI = true;
            try
            {
                foreach (ListViewItem item in new List<ListViewItem>(routingListView.SelectedItems.Cast<ListViewItem>()))
                {
                    var element = item.Tag as RoutingElement;
                    if (element != null)
                        selectedTendon.RemoveRoutingElement(element);
                }
                selectedRoutingElement = null;
                PopulateRoutingGrid();
                UpdateTendonElementCount();
            }
            finally
            {
                isUpdatingTendonUI = false;
            }
        }

        private void OnRoutingPointSelected()
        {
            if (isUpdatingTendonUI) return;
            if (selectedTendon == null) return;

            var selMgr = (SelectionMgr)model.ISelectionManager;
            int count = selMgr.GetSelectedObjectCount2(PMSelectionBoxPoint.Mark);
            if (count == 0) return;

            object selectedObj = selMgr.GetSelectedObject6(1, PMSelectionBoxPoint.Mark);
            if (selectedObj == null) return;

            isUpdatingTendonUI = true;
            try
            {
                // Get PID for the selected point
                string pointKey = GetPointPersistKey(selectedObj);
                double[] coords = GetPointCoordinates(selectedObj);

                if (PMCheckboxAutoAddWaypoint.Checked)
                {
                    // Auto-add mode: create new routing element
                    var element = new RoutingElement();
                    element.Type = RoutingElement.TypeWaypoint;
                    element.PointKey = pointKey;

                    if (coords != null)
                    {
                        element.SetPosition(coords);
                    }

                    // Auto-determine link
                    if (PMCheckboxAutoLink.Checked)
                    {
                        string linkName = DetermineLinkForSelection(selMgr);
                        if (!string.IsNullOrEmpty(linkName))
                            element.Link = linkName;
                    }

                    if (string.IsNullOrEmpty(element.Link))
                    {
                        List<string> linkNames = GetAllLinkNames();
                        if (linkNames.Count > 0)
                            element.Link = linkNames[0];
                    }

                    selectedTendon.AddRoutingElement(element);
                    AddRoutingRow(element, selectedTendon.RoutingElements.Count);
                    UpdateTendonElementCount();

                    // Inline the selection update to avoid cascading through SelectRoutingElement
                    selectedRoutingElement = element;
                    PMComboElementType.CurrentSelection = 0;
                    PMNumberBoxCoefficient.Value = element.Coefficient;

                    EnsureParentLinkComboPopulated();
                    int linkIdx = cachedLinkNames.IndexOf(element.Link);
                    if (linkIdx >= 0)
                        PMComboParentLink.CurrentSelection = (short)linkIdx;

                    if (routingListView.Items.Count > 0)
                    {
                        routingListView.SelectedItems.Clear();
                        routingListView.Items[routingListView.Items.Count - 1].Selected = true;
                    }
                }
                else if (selectedRoutingElement != null)
                {
                    // Update existing element
                    selectedRoutingElement.PointKey = pointKey;
                    if (coords != null)
                    {
                        selectedRoutingElement.SetPosition(coords);
                    }

                    if (PMCheckboxAutoLink.Checked)
                    {
                        string linkName = DetermineLinkForSelection(selMgr);
                        if (!string.IsNullOrEmpty(linkName))
                            selectedRoutingElement.Link = linkName;
                    }

                    PopulateRoutingGrid();
                }
            }
            finally
            {
                isUpdatingTendonUI = false;
            }
        }

        private void OnElementTypeChanged(int item)
        {
            if (isUpdatingTendonUI) return;
            if (selectedRoutingElement == null) return;

            selectedRoutingElement.Type = item == 0
                ? RoutingElement.TypeWaypoint
                : RoutingElement.TypeLinearJoint;

            if (selectedRoutingElement.Type == RoutingElement.TypeLinearJoint)
            {
                selectedRoutingElement.PointKey = null;
                selectedRoutingElement.SetPosition(new double[3]);
            }

            PopulateRoutingGrid();
            UpdateTendonUIState();
        }

        private void OnParentLinkChanged(int item)
        {
            if (isUpdatingTendonUI) return;
            if (selectedRoutingElement == null) return;

            List<string> linkNames = GetAllLinkNames();
            if (item >= 0 && item < linkNames.Count)
            {
                selectedRoutingElement.Link = linkNames[item];
                PopulateRoutingGrid();
            }
        }

        private void OnCoefficientChanged(double value)
        {
            if (isUpdatingTendonUI) return;
            if (selectedRoutingElement == null) return;
            selectedRoutingElement.Coefficient = value;
            PopulateRoutingGrid();
        }

        #endregion

        #region Tendon UI Helpers

        private void TendonListView_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (isUpdatingTendonUI) return;
            if (tendonListView.SelectedItems.Count > 0)
            {
                var tendon = (Tendon)tendonListView.SelectedItems[0].Tag;
                SelectTendon(tendon);
            }
        }

        private void RoutingListView_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (isUpdatingTendonUI) return;
            if (routingListView.SelectedItems.Count > 0)
            {
                var element = routingListView.SelectedItems[0].Tag as RoutingElement;
                if (element != null)
                    SelectRoutingElement(element);
            }
        }

        private void SelectRoutingElement(RoutingElement element)
        {
            isUpdatingTendonUI = true;
            try
            {
                selectedRoutingElement = element;

                // Update UI to reflect selected element
                PMComboElementType.CurrentSelection = (short)
                    (element.Type == RoutingElement.TypeLinearJoint ? 1 : 0);

                PMNumberBoxCoefficient.Value = element.Coefficient;

                // Set parent link combo
                EnsureParentLinkComboPopulated();
                int linkIdx = cachedLinkNames.IndexOf(element.Link);
                if (linkIdx >= 0)
                    PMComboParentLink.CurrentSelection = (short)linkIdx;

                // Restore point selection
                CommonSwOperations.DeselectByMark(model, PMSelectionBoxPoint.Mark);
                if (!string.IsNullOrEmpty(element.PointKey))
                {
                    SelectPointFromKey(element.PointKey);
                }
            }
            finally
            {
                isUpdatingTendonUI = false;
            }
        }

        private void PopulateRoutingGrid()
        {
            bool wasUpdating = isUpdatingTendonUI;
            isUpdatingTendonUI = true;
            try
            {
                routingListView.BeginUpdate();
                routingListView.Items.Clear();
                if (selectedTendon == null) return;

                for (int i = 0; i < selectedTendon.RoutingElements.Count; i++)
                {
                    AddRoutingRow(selectedTendon.RoutingElements[i], i + 1);
                }
            }
            finally
            {
                routingListView.EndUpdate();
                isUpdatingTendonUI = wasUpdating;
            }
        }

        private void AddRoutingRow(RoutingElement element, int index)
        {
            string typeDisplay = element.Type == RoutingElement.TypeLinearJoint
                ? "Linear Joint" : "Waypoint";
            string pointDisplay = !string.IsNullOrEmpty(element.PointKey) ? "\u2713" : "-";
            string linkDisplay = !string.IsNullOrEmpty(element.Link) ? element.Link : "-";
            string coefDisplay = element.Type == RoutingElement.TypeLinearJoint
                ? element.Coefficient.ToString("G") : "-";

            var item = new ListViewItem(new[]
            {
                index.ToString(), typeDisplay, pointDisplay, linkDisplay, coefDisplay
            });
            item.Tag = element;
            routingListView.Items.Add(item);
        }

        private void UpdateTendonElementCount()
        {
            if (selectedTendon == null) return;

            foreach (ListViewItem item in tendonListView.Items)
            {
                if (item.Tag == selectedTendon)
                {
                    item.SubItems[1].Text = selectedTendon.RoutingElements.Count.ToString();
                    break;
                }
            }
        }

        private void UpdateTendonUIState()
        {
            // Intentionally not toggling .Enabled on PM controls here.
            // SolidWorks PM pages inside tabs can lose their rendering when
            // controls are enabled/disabled programmatically during event handlers.
            // Instead, we validate state at action time (e.g. OnAddRoutingElementClicked
            // checks selectedTendon != null).
        }

        private List<string> cachedLinkNames = new List<string>();

        private List<string> GetAllLinkNames()
        {
            var names = new List<string>();
            if (Tree == null || Tree.Nodes.Count == 0) return names;
            CollectLinkNamesFromTree((LinkNode)Tree.Nodes[0], names);
            return names;
        }

        private void CollectLinkNamesFromTree(LinkNode node, List<string> names)
        {
            if (node == null) return;
            if (!string.IsNullOrEmpty(node.Link?.Name))
                names.Add(node.Link.Name);
            foreach (LinkNode child in node.Nodes)
                CollectLinkNamesFromTree(child, names);
        }

        private void EnsureParentLinkComboPopulated()
        {
            List<string> linkNames = GetAllLinkNames();
            if (linkNames.SequenceEqual(cachedLinkNames))
                return;

            cachedLinkNames = linkNames;
            PopulateParentLinkCombo();
        }

        private void PopulateParentLinkCombo()
        {
            List<string> linkNames = GetAllLinkNames();
            PMComboParentLink.Clear();
            if (linkNames.Count > 0)
            {
                PMComboParentLink.AddItems(linkNames.ToArray());
                PMComboParentLink.CurrentSelection = 0;
            }
        }

        public void PopulateTendonsFromList(List<Tendon> loadedTendons)
        {
            isUpdatingTendonUI = true;
            try
            {
                tendons.Clear();
                tendonListView.BeginUpdate();
                tendonListView.Items.Clear();
                routingListView.BeginUpdate();
                routingListView.Items.Clear();
                selectedTendon = null;
                selectedRoutingElement = null;

                if (loadedTendons == null) return;

                foreach (var tendon in loadedTendons)
                {
                    tendons.Add(tendon);
                    var item = new ListViewItem(new[]
                    {
                        tendon.Name,
                        tendon.RoutingElements.Count.ToString()
                    });
                    item.Tag = tendon;
                    tendonListView.Items.Add(item);
                }

                SyncTendonCounter();
                UpdateTendonUIState();
            }
            finally
            {
                tendonListView.EndUpdate();
                routingListView.EndUpdate();
                isUpdatingTendonUI = false;
            }
        }

        private void SyncTendonCounter()
        {
            tendonCounter = 0;
            foreach (var tendon in tendons)
            {
                if (tendon.Name != null && tendon.Name.StartsWith("tendon_"))
                {
                    string suffix = tendon.Name.Substring("tendon_".Length);
                    if (int.TryParse(suffix, out int num) && num > tendonCounter)
                        tendonCounter = num;
                }
            }
        }

        #endregion

        #region Point Helpers

        private void SelectPointFromKey(string pointKey)
        {
            try
            {
                byte[] pid = Convert.FromBase64String(pointKey);
                object obj = model.Extension.GetObjectByPersistReference3(pid, out int errors);
                if (errors != 0 || obj == null) return;

                SelectionMgr manager = (SelectionMgr)model.ISelectionManager;
                SelectData data = manager.CreateSelectData();
                data.Mark = PMSelectionBoxPoint.Mark;

                if (obj is Feature feature)
                {
                    feature.Select2(false, data.Mark);
                    return;
                }
            }
            catch
            {
            }
        }

        private string GetPointPersistKey(object pointEntity)
        {
            if (pointEntity == null) return null;
            try
            {
                byte[] pid = (byte[])model.Extension.GetPersistReference3(pointEntity);
                if (pid == null) return null;
                return Convert.ToBase64String(pid);
            }
            catch
            {
                return null;
            }
        }

        private double[] GetPointCoordinates(object pointEntity)
        {
            try
            {
                if (pointEntity is ISketchPoint skPt)
                {
                    // Sketch points are in sketch space; need to account for sketch transform
                    ISketch ownerSketch = skPt.GetSketch();
                    double x = skPt.X;
                    double y = skPt.Y;
                    double z = skPt.Z;

                    if (ownerSketch != null)
                    {
                        MathTransform sketchTransform = ownerSketch.ModelToSketchTransform;
                        if (sketchTransform != null)
                        {
                            MathTransform inverse = sketchTransform.Inverse();
                            IMathUtility mathUtil = (IMathUtility)swApp.GetMathUtility();
                            IMathPoint pt = (IMathPoint)mathUtil.CreatePoint(new double[] { x, y, z });
                            pt = (IMathPoint)pt.MultiplyTransform(inverse);
                            double[] arr = (double[])pt.ArrayData;
                            return arr;
                        }
                    }
                    return new double[] { x, y, z };
                }
                else if (pointEntity is IRefPoint refPt)
                {
                    IMathPoint pt = refPt.GetRefPoint();
                    if (pt != null)
                    {
                        double[] arr = (double[])pt.ArrayData;
                        return arr;
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Warning("Failed to get point coordinates: " + ex.Message);
            }
            return null;
        }

        private string DetermineLinkForSelection(SelectionMgr selMgr)
        {
            try
            {
                Component2 comp = (Component2)selMgr.GetSelectedObjectsComponent4(1, PMSelectionBoxPoint.Mark);
                if (comp == null) return null;

                // Walk kinematic tree to find which link owns this component
                return FindLinkForComponent(comp, (LinkNode)Tree.Nodes[0]);
            }
            catch
            {
                return null;
            }
        }

        private string FindLinkForComponent(Component2 targetComp, LinkNode node)
        {
            if (node == null) return null;

            // Check if this link's components contain the target
            if (node.Link != null)
            {
                var allComponents = new List<Component2>();
                if (node.Link.SWVisualComponents != null) allComponents.AddRange(node.Link.SWVisualComponents);
                if (node.Link.SWCollisionComponents != null) allComponents.AddRange(node.Link.SWCollisionComponents);
                if (node.Link.SWInertialComponents != null) allComponents.AddRange(node.Link.SWInertialComponents);

                foreach (var comp in allComponents)
                {
                    if (comp != null && comp.Name2 == targetComp.Name2)
                        return node.Link.Name;
                }
            }

            foreach (LinkNode child in node.Nodes)
            {
                string result = FindLinkForComponent(targetComp, child);
                if (result != null) return result;
            }
            return null;
        }

        #endregion

        #region Tendon Name Changed

        private void OnTendonNameChanged()
        {
            if (isUpdatingTendonUI) return;
            if (selectedTendon == null) return;
            string newName = PMTextBoxTendonName.Text;
            if (string.IsNullOrEmpty(newName)) return;

            selectedTendon.Name = newName;

            tendonListView.BeginUpdate();
            foreach (ListViewItem item in tendonListView.Items)
            {
                if (item.Tag == selectedTendon)
                {
                    item.SubItems[0].Text = newName;
                    break;
                }
            }
            tendonListView.EndUpdate();
        }

        #endregion
    }
}

#endif
