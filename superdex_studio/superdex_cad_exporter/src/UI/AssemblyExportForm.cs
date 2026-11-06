/*
Copyright (c) 2015 Stephen Brawner
Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms.Integration;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;

using CADRobotExporter.Export;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.RobotExport;
using CADRobotExporter.CAD;
using CADRobotExporter.Utilities;
using CADRobotExporter.UI.ViewModels;
using CADRobotExporter.UI.Views;

using SaveFileDialog = System.Windows.Forms.SaveFileDialog;

// Legacy WinForms MessageBox
using WindowState = System.Windows.WindowState;
using MessageBox = System.Windows.Forms.MessageBox;
using MessageBoxButtons = System.Windows.Forms.MessageBoxButtons;
using MessageBoxIcon = System.Windows.Forms.MessageBoxIcon;
using MessageBoxOptions = System.Windows.Forms.MessageBoxOptions;
using MessageBoxDefaultButton = System.Windows.Forms.MessageBoxDefaultButton;


namespace CADRobotExporter.UI
{
    public partial class AssemblyExportForm : System.Windows.Window
    {
        public static bool FormIsOpen = false;
        public static AssemblyExportForm Instance { get; private set; }

        private static readonly Serilog.ILogger logger = Logger.GetLogger();

        public ExportHelper Exporter;
        public bool AutoUpdatingForm;

        private LinkTreeItem previouslySelectedNode;
        private readonly LinkNode BaseNode;

        // The robot-wide meshing options, used by every link that is not marked as overriding
        private Export.MeshingOptions globalVisualMeshing = new Export.MeshingOptions();
        private Export.MeshingOptions globalCollisionMeshing = new Export.MeshingOptions();

        // The link whose override the meshing columns are showing, or null when they are showing the
        // robot-wide options.
        private Link meshingEditorLink;

        private int _currentTabIndex;
        private bool _navigating;
        private bool _isClosing;
        private bool _syncingMeshFormats;
        private bool _syncingMeshingOverride;

        private readonly MeshFormatSelection visualMeshFormats = new MeshFormatSelection();
        private readonly MeshFormatSelection collisionMeshFormats = new MeshFormatSelection();

        private readonly KinematicsSummaryView kinematicsSummaryView;
        private readonly TendonsSummaryView tendonsSummaryView;
        private readonly JointPropertiesView jointPropertiesView;
        private readonly InertialPropertiesView inertialPropertiesView;
        private readonly VisualPropertiesView visualPropertiesView;
        private readonly MeshExportView meshExportView;

        // Solidworks
        private CADBridge CadBridge { get; set; }

        public AssemblyExportForm(LinkNode node, ExportHelper exporter)
        {
            // Tab, Enter and mnemonics are swallowed without this.
            ElementHost.EnableModelessKeyboardInterop(this);
            Dispatcher.UnhandledException += DispatcherUnhandledException;
            AppDomain.CurrentDomain.UnhandledException +=
                new UnhandledExceptionEventHandler(UnhandledException);
            InitializeComponent();

            AutoUpdatingForm = false;

            kinematicsSummaryView = new KinematicsSummaryView();
            kinematicsSummaryView.UseDegreesChanged += (s, ev) => PopulateSummaryGrid();
            kinematicsSummaryView.RowActivated += OnKinematicsRowActivated;

            jointPropertiesView = new JointPropertiesView();
            jointPropertiesView.SetJointTypes(new[] { "revolute", "continuous", "prismatic", "fixed" });
            jointPropertiesView.UseDegreesChanged += UseDegreesCheckboxChanged;
            jointPropertiesView.LimitsChanged += UpdateJointLimits;

            inertialPropertiesView = new InertialPropertiesView();
            inertialPropertiesView.RecalculateRequested += RecalculateInertialProperties;

            visualPropertiesView = new VisualPropertiesView();

            meshExportView = new MeshExportView();
            meshExportView.SetFolderStructures(Enum.GetNames(typeof(FolderStructure)));
            meshExportView.MeshingOverrideChanged += MeshingOverrideCheckboxChanged;
            meshExportView.CollisionMeshingChanged += CollisionCheckboxChanged;
            meshExportView.ResetToDefaultsRequested += ResetMeshingToDefaults;
            meshExportView.ExportLinkMeshRequested += ButtonExportLinkMeshClick;
            meshExportView.HelpMeChooseRequested += LinkLabelHelpMeChoose_LinkClicked;
            meshExportView.MesherSettingChanged += MesherSettingChanged;
            meshExportView.VisualFormatToggled += MeshExportCheckChanged;
            meshExportView.CollisionFormatToggled += MeshExportCollisionCheckChanged;

            tendonsSummaryView = new TendonsSummaryView();
            tendonsSummaryView.ShowSelectedTendonChanged += (s, ev) => ApplyTendonVisualization();
            tendonsSummaryView.ShowAllTendonsChanged += (s, ev) => ApplyShowAllTendons();
            tendonsSummaryView.TendonRowSelected += OnTendonRowSelected;

            StackPanel linkPanel = new StackPanel { Margin = new Thickness(8) };
            linkPanel.Children.Add(inertialPropertiesView);
            GroupBox meshExportGroup = new GroupBox
            {
                Header = "Visual and Collision Mesh Export Properties",
                Padding = new Thickness(6),
                Margin = new Thickness(0, 8, 0, 0),
            };
            StackPanel meshExportBody = new StackPanel();
            meshExportView.Margin = new Thickness(0, 8, 0, 0);
            meshExportBody.Children.Add(visualPropertiesView);
            meshExportBody.Children.Add(meshExportView);
            meshExportGroup.Content = meshExportBody;
            linkPanel.Children.Add(meshExportGroup);

            PopulateTabs(
                new ScrollViewer
                {
                    Content = jointPropertiesView,
                    VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                    HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
                },
                new ScrollViewer
                {
                    Content = linkPanel,
                    VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                    HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
                },
                kinematicsSummaryView,
                tendonsSummaryView);

            BaseNode = CreateLinkNodeFromLink(node.Link);
            Exporter = exporter;
            CadBridge = Exporter.CadBridge;

            LoadExporterConfigurationToForm(CadBridge.GetExporterConfiguration());

            treeViewJointTree.SelectedItemChanged += TreeViewJointtreeAfterSelect;
            treeViewLinkProperties.SelectedItemChanged += TreeViewLinkPropertiesAfterSelect;
            buttonPrevious.Click += ButtonPreviousClick;
            buttonNext.Click += ButtonNextClick;
            buttonClose.Click += ButtonCancelClick;
            buttonLinksFinish.Click += ButtonLinksFinishClick;
            buttonLinksExportUrdfOnly.Click += ButtonLinksExportUrdfOnlyClick;
            tabControl.SelectionChanged += TabControlSelected;
            Closing += OnCloseForm;
            Closed += OnFormClosed;
            Loaded += AssemblyExportFormLoad;
            checkBoxShowJointVisualization.Checked += ShowJointVizCheckChanged;
            checkBoxShowJointVisualization.Unchecked += ShowJointVizCheckChanged;
            checkBoxShowLinkVisualization.Checked += ShowLinkVizCheckChanged;
            checkBoxShowLinkVisualization.Unchecked += ShowLinkVizCheckChanged;
            trackBarJointGizmoSize.ValueChanged += UpdateJointGizmoSize;
            checkBoxShowJointHighlights.Checked += ShowJointHighlightsCheckChanged;
            checkBoxShowJointHighlights.Unchecked += ShowJointHighlightsCheckChanged;
            checkBoxLinkHighlights.Checked += ShowLinkHighlightsCheckChanged;
            checkBoxLinkHighlights.Unchecked += ShowLinkHighlightsCheckChanged;
            trackBarLinkGizmoSize.ValueChanged += UpdateLinkGizmoSize;

            SetCADSpecificCheckboxProperties();

            PopulatePresetComboBox();
            visualPropertiesView.PresetSelectionChanged += ComboBoxExporterConfigurationPreset_SelectedIndexChanged;

            _currentTabIndex = 0;
            _navigating = false;

            FormIsOpen = true;
            Instance = this;

            UseDegreesCheckboxChanged(null, null);

            this.Title = "SuperDex CAD Exporter " + Versioning.VersionString.Get() + " - Export";
        }

        private void LinkLabelHelpMeChoose_LinkClicked(object sender, EventArgs e)
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = "https://facebookresearch.github.io/project_superdex/studio/docs/cad_exporter/#mesh-export-guide",
                UseShellExecute = true
            });
        }

        private void SetCADSpecificCheckboxProperties()
        {
            string cad = "";
            string objSuffix = null;
            bool cadObjAndGlbSupported = true;
#if NX
            cad = "NX";
            objSuffix = " \u26A0\uFE0F";
#endif
#if SOLIDWORKS
            cad = "Solidworks";
            cadObjAndGlbSupported = false;
#endif
            meshExportView.ApplyHostProfile(cad, cadObjAndGlbSupported, objSuffix);

#if NX
            meshExportView.SetRowToolTip(
                MeshFormat.objCAD,
                "NX OBJ export does not produce smooth normals for visual models. " +
                "Use GLB or SuperDex options. MuJoCo also does not support multi-body OBJs, the meshes may need post-processing.",
                "NX OBJ export does not produce smooth normals. " +
                "Use GLB or SuperDex options.");
#endif
        }

        private void SanitizeCADSpecificCheckboxes()
        {
#if SOLIDWORKS
            meshExportView.UncheckRow(MeshFormat.objCAD);
            meshExportView.UncheckRow(MeshFormat.glbCAD);
#endif
            meshExportView.ApplyEnableState();
        }

        private static bool CanOverrideMeshing(Link link)
        {
            return link != null && !link.isFixedFrame && !link.isSite;
        }

        // Commits the meshing columns to whichever tier they are currently showing.
        private void CommitMeshingColumns()
        {
            if (meshingEditorLink != null)
            {
                meshExportView.Visual.ReadInto(meshingEditorLink.visualMeshingOptions);
                meshExportView.Collision.ReadInto(meshingEditorLink.collisionMeshingOptions);
            }
            else
            {
                meshExportView.Visual.ReadInto(globalVisualMeshing);
                meshExportView.Collision.ReadInto(globalCollisionMeshing);
            }
        }

        // Points the meshing columns at <paramref name="link"/>'s override if it has one, and at the
        // robot-wide options otherwise.
        private void ShowMeshingColumnsFor(Link link)
        {
            bool canOverride = CanOverrideMeshing(link);
            bool overridden = canOverride && link.meshingOverride;

            meshingEditorLink = overridden ? link : null;

            // The check box raises its change event on a programmatic set too, and this runs on every
            // link selection, so without the guard simply navigating the tree would re-run the seed
            // and discard paths.
            _syncingMeshingOverride = true;
            try
            {
                meshExportView.MeshingOverrideEnabled = canOverride;
                meshExportView.MeshingOverride = overridden;
            }
            finally
            {
                _syncingMeshingOverride = false;
            }

            WriteMeshingColumns(
                overridden ? link.visualMeshingOptions : globalVisualMeshing,
                overridden ? link.collisionMeshingOptions : globalCollisionMeshing);

            RefreshMeshingOverrideBold();
        }

        // Bolds only the values an override on screen actually changes from the robot-wide ones.
        private void RefreshMeshingOverrideBold()
        {
            bool overridden = meshingEditorLink != null;
            meshExportView.BoldValuesDifferentFrom(
                overridden ? globalVisualMeshing : null,
                overridden ? globalCollisionMeshing : null);
        }

        private void ShowMeshingColumnsForSelectedLink()
        {
            LinkTreeItem node = treeViewLinkProperties.SelectedItem as LinkTreeItem;
            ShowMeshingColumnsFor(node?.Link);
        }

        private void MeshingOverrideCheckboxChanged(object sender, EventArgs e)
        {
            if (_syncingMeshingOverride)
            {
                return;
            }

            LinkTreeItem node = treeViewLinkProperties.SelectedItem as LinkTreeItem;
            if (node == null)
            {
                return;
            }

            CommitMeshingColumns();

            Link link = node.Link;
            if (meshExportView.MeshingOverride)
            {
                link.visualMeshingOptions = globalVisualMeshing.Clone();
                link.collisionMeshingOptions = globalCollisionMeshing.Clone();
                link.meshingOverride = true;
            }
            else
            {
                link.meshingOverride = false;
            }

            ShowMeshingColumnsFor(link);
            node.RefreshMeshingOverride();
        }

        /// <summary>
        /// Re-renders the selected joint in the newly chosen angle unit. Pending edits are committed
        /// first using the previous unit so nothing typed is lost, then read back in the new one.
        /// </summary>
        private void UseDegreesCheckboxChanged(object sender, EventArgs e)
        {
            AutoUpdatingForm = true;

            LinkTreeItem node = treeViewJointTree.SelectedItem as LinkTreeItem;
            if (node != null && node.Link.Joint != null)
            {
                bool previouslyDegrees = !jointPropertiesView.UseDegrees;
                jointPropertiesView.Read().SaveTo(node.Link.Joint, previouslyDegrees);

                JointPropertiesViewModel properties = new JointPropertiesViewModel();
                properties.LoadFrom(node.Link.Joint, jointPropertiesView.UseDegrees);
                jointPropertiesView.Show(properties);
            }

            AutoUpdatingForm = false;
        }

        private static double RadiansToDegrees(double radians) => JointAngles.RadiansToDegrees(radians);
        private static double DegreesToRadians(double degrees) => JointAngles.DegreesToRadians(degrees);

        private void ResetMeshingToDefaults(object sender, EventArgs e)
        {
            ExporterConfiguration defaults = new ExporterConfiguration();
            WriteMeshingColumns(defaults.visualMeshingOptions, defaults.collisionMeshingOptions);
            CommitMeshingColumns();
        }

        private void CollisionCheckboxChanged(object sender, EventArgs e)
        {
            meshExportView.ApplyEnableState();
        }

        private void ShowJointVizCheckChanged(object sender, RoutedEventArgs e)
        {
            CadBridge.IsShowingJointGizmo = checkBoxShowJointVisualization.IsChecked == true;
            CadBridge.TriggerGraphicsRedraw();
        }

        private void ShowLinkVizCheckChanged(object sender, RoutedEventArgs e)
        {
            CadBridge.IsShowingInertialGizmo = checkBoxShowLinkVisualization.IsChecked == true;
            CadBridge.TriggerGraphicsRedraw();
        }

        private void ApplyTendonVisualization()
        {
            CadBridge.IsShowingTendonVisualization = tendonsSummaryView.ShowSelectedTendon;
            CadBridge.TriggerGraphicsRedraw();
        }

        private void ApplyShowAllTendons()
        {
            bool showAll = tendonsSummaryView.ShowAllTendons;
            CadBridge.IsShowingAllTendons = showAll;
            if (showAll)
            {
                CadBridge.AllTendons = Exporter.Robot.Tendons;
                if (CadBridge.TendonLinkCsysMap == null)
                {
                    var map = new Dictionary<string, string>();
                    CollectLinkCsysNames(BaseNode.Link, map);
                    CadBridge.TendonLinkCsysMap = map;
                }
            }
            CadBridge.ShowHideVisualizations(showAll || tendonsSummaryView.ShowSelectedTendon);
            CadBridge.TriggerGraphicsRedraw();
        }

        private void UpdateJointGizmoSize(object sender, System.Windows.RoutedPropertyChangedEventArgs<double> e)
        {
            CadBridge.JointGizmoScale = GizmoScaleFor(trackBarJointGizmoSize);
            CadBridge.TriggerGraphicsRedraw();
        }

        private void UpdateLinkGizmoSize(object sender, System.Windows.RoutedPropertyChangedEventArgs<double> e)
        {
            CadBridge.LinkGizmoScale = GizmoScaleFor(trackBarLinkGizmoSize);
            CadBridge.TriggerGraphicsRedraw();
        }

        private static double GizmoScaleFor(Slider slider)
        {
            double normalized = slider.Value / slider.Maximum;
            return Math.Pow(8.0, (normalized - 0.5) / 0.5);
        }

        private void ShowJointHighlightsCheckChanged(object sender, RoutedEventArgs e)
        {
            if (checkBoxShowJointHighlights.IsChecked != true)
            {
                CadBridge.UnselectAll();
            }
            else
            {
                LinkTreeItem node = treeViewJointTree.SelectedItem as LinkTreeItem;
                if (node != null)
                {
                    CadBridge.SelectJointComponents(node.Link);
                }
            }
        }

        private void ShowLinkHighlightsCheckChanged(object sender, RoutedEventArgs e)
        {
            if (checkBoxLinkHighlights.IsChecked != true)
            {
                CadBridge.UnselectAll();
            }
            else
            {
                LinkTreeItem node = treeViewLinkProperties.SelectedItem as LinkTreeItem;
                if (node != null)
                {
                    CadBridge.SelectLinkComponents(node.Link);
                }
            }
        }

        private void UpdateJointLimits(object sender, EventArgs e)
        {
            if (AutoUpdatingForm)
            {
                return;
            }

            LinkTreeItem node = treeViewJointTree.SelectedItem as LinkTreeItem;
            if (node != null && node.Link.Joint != null)
            {
                JointPropertiesViewModel current = jointPropertiesView.Read();
                bool useDegrees = jointPropertiesView.UseDegrees;
                double upper = Convert.ToDouble(current.LimitUpper);
                double lower = Convert.ToDouble(current.LimitLower);

                if (useDegrees && node.Link.Joint.Type == "revolute")
                {
                    upper = DegreesToRadians(upper);
                    lower = DegreesToRadians(lower);
                }

                node.Link.Joint.Limit.Upper = upper;
                node.Link.Joint.Limit.Lower = lower;
            }

            CadBridge.TriggerGraphicsRedraw();
        }

        private void RecalculateInertialProperties(object sender, EventArgs e)
        {
            LinkTreeItem node = treeViewLinkProperties.SelectedItem as LinkTreeItem;
            if (node != null && node.Link != null)
            {
                Exporter.ComputeInertialProperties(node.Link);
                FillLinkPropertyBoxes(node.Link);
            }
        }

        private void MeshExportCheckChanged(object sender, MeshFormatToggledEventArgs e)
        {
            ApplyMeshFormatChange(visualMeshFormats, isCollision: false, toggled: e);
        }

        private void MeshExportCollisionCheckChanged(object sender, MeshFormatToggledEventArgs e)
        {
            ApplyMeshFormatChange(collisionMeshFormats, isCollision: true, toggled: e);
        }

        private void ApplyMeshFormatChange(
            MeshFormatSelection selection, bool isCollision, MeshFormatToggledEventArgs toggled)
        {
            // Writing the boxes back below re-raises the toggle events; ignore those echoes so the
            // rules are applied exactly once per user action.
            if (_syncingMeshFormats)
            {
                return;
            }

            selection.Reset(
                isCollision ? meshExportView.CheckedCollisionFormats : meshExportView.CheckedVisualFormats,
                isCollision ? meshExportView.CollisionExtension : meshExportView.VisualExtension);
            selection.Toggle(toggled.Format, toggled.IsChecked);

            _syncingMeshFormats = true;
            try
            {
                if (isCollision)
                {
                    meshExportView.ShowCollisionFormats(selection.SelectedFormats());
                    meshExportView.CollisionExtension = selection.FileExtension;
                }
                else
                {
                    meshExportView.ShowVisualFormats(selection.SelectedFormats());
                    meshExportView.VisualExtension = selection.FileExtension;
                }
            }
            finally
            {
                _syncingMeshFormats = false;
            }

            // Outside the guard: the writes above re-raise the toggle events
            meshExportView.ApplyEnableState();
        }
        private void TabControlSelected(object sender, SelectionChangedEventArgs e)
        {
            // Only a change of the tab itself is a navigation.
            if (!ReferenceEquals(e.OriginalSource, tabControl))
                return;
            if (_navigating)
                return;
            NavigateToTab(tabControl.SelectedIndex);
        }

        private void NavigateToTab(int targetTab)
        {
            if (targetTab == _currentTabIndex)
                return;

            if (!OnExitTab(_currentTabIndex))
                return;

            _currentTabIndex = targetTab;
            OnEnterTab(_currentTabIndex);

            _navigating = true;
            tabControl.SelectedIndex = _currentTabIndex;
            _navigating = false;

            Activate();
            CadBridge.TriggerGraphicsRedraw();
        }

        private bool OnExitTab(int tabIndex)
        {
            CadBridge.UnselectAll();

            switch (tabIndex)
            {
                case 0: return OnExitJointTab();
                case 1: OnExitLinkTab(); return true;
                case 2: return true;
                case 3:
                    CadBridge.IsShowingTendonVisualization = false;
                    CadBridge.IsShowingAllTendons = false;
                    CadBridge.CurrentTendonShown = null;
                    CadBridge.ShowHideVisualizations(false);
                    CadBridge.TriggerGraphicsRedraw();
                    return true;
                default: return true;
            }
        }

        private void OnEnterTab(int tabIndex)
        {
            switch (tabIndex)
            {
                case 0: OnEnterJointTab(); break;
                case 1: OnEnterLinkTab(); break;
                case 2: OnEnterSummaryTab(); break;
                case 3: OnEnterTendonsSummaryTab(); break;
            }
        }

        private bool OnExitJointTab()
        {
            if (!(previouslySelectedNode == null || previouslySelectedNode.Link.Joint == null))
            {
                SaveJointDataFromPropertyBoxes(previouslySelectedNode.Link.Joint);
            }
            previouslySelectedNode = null;

            string errors = CheckJointsForErrors();
            if (!string.IsNullOrWhiteSpace(errors))
            {
                string message = "The following joints are missing required fields, please " +
                    "address them before continuing\r\n\r\n" + errors;
                MessageBox.Show(message, "SuperDex CAD Exporter - Joint Errors");
                return false;
            }

            CadBridge.IsShowingJointGizmo = false;
            CadBridge.ShowHideVisualizations(false);
            CadBridge.TriggerGraphicsRedraw();

            return true;
        }

        private void OnEnterJointTab()
        {
            treeViewLinkProperties.ItemsSource = null;
            FillJointTree();
            buttonPrevious.IsEnabled = false;
            buttonNext.IsEnabled = true;
            buttonLinksExportUrdfOnly.IsEnabled = false;
            buttonLinksFinish.IsEnabled = false;

            CadBridge.ShowHideVisualizations(checkBoxShowJointVisualization.IsChecked == true);
        }

        private void OnExitLinkTab()
        {
            LinkTreeItem node = treeViewLinkProperties.SelectedItem as LinkTreeItem;
            if (node != null)
            {
                SaveLinkDataFromPropertyBoxes(node.Link);
            }

            // Commit any pending meshing edit, then put the columns back on the robot-wide options so
            // they still agree with meshingEditorLink.
            CommitMeshingColumns();
            ShowMeshingColumnsFor(null);

            previouslySelectedNode = null;

            CadBridge.IsShowingInertialGizmo = false;
            CadBridge.CurrentLinkNodeShown = null;
            CadBridge.ShowHideVisualizations(false);
            CadBridge.TriggerGraphicsRedraw();
        }

        private void OnEnterLinkTab()
        {
            FillLinkTree();
            buttonPrevious.IsEnabled = true;
            buttonNext.IsEnabled = true;
            buttonLinksExportUrdfOnly.IsEnabled = true;
            buttonLinksFinish.IsEnabled = true;

            CadBridge.IsShowingInertialGizmo = checkBoxShowLinkVisualization.IsChecked == true;
            CadBridge.ShowHideVisualizations(true);
        }

        private void OnEnterSummaryTab()
        {
            buttonPrevious.IsEnabled = true;
            buttonNext.IsEnabled = true;
            PopulateSummaryGrid();
        }

        private void OnEnterTendonsSummaryTab()
        {
            buttonPrevious.IsEnabled = true;
            buttonNext.IsEnabled = false;

            // Build link CSYS map for tendon visualization
            var map = new Dictionary<string, string>();
            CollectLinkCsysNames(BaseNode.Link, map);
            CadBridge.TendonLinkCsysMap = map;
            CadBridge.IsShowingTendonVisualization = false;
            CadBridge.CurrentTendonShown = null;

            CadBridge.ShowHideVisualizations(true);

            PopulateTendonsSummaryGrid();
        }

        private void ButtonNextClick(object sender, EventArgs e)
        {
            if (_currentTabIndex < 3)
                NavigateToTab(_currentTabIndex + 1);
        }

        private void ButtonPreviousClick(object sender, EventArgs e)
        {
            if (_currentTabIndex > 0)
                NavigateToTab(_currentTabIndex - 1);
        }

        private void PopulateSummaryGrid()
        {
            kinematicsSummaryView.ShowRows(
                KinematicsSummary.Build(BaseNode.Link, kinematicsSummaryView.UseDegrees, GetLinkBodyCounts),
                kinematicsSummaryView.UseDegrees);
        }

        private static LinkBodyCounts GetLinkBodyCounts(Link link)
        {
            LinkBodyCounts counts = new LinkBodyCounts();
#if SOLIDWORKS
            if (link.SWInertialComponents != null) counts.Inertial = link.SWInertialComponents.Count;
            if (link.SWCollisionComponents != null) counts.Collision = link.SWCollisionComponents.Count;
            if (link.SWVisualComponents != null) counts.Visual = link.SWVisualComponents.Count;
#endif
#if NX
            if (link.NXInertialBodiesHandles != null) counts.Inertial = link.NXInertialBodiesHandles.Count;
            if (link.NXCollisionBodiesHandles != null) counts.Collision = link.NXCollisionBodiesHandles.Count;
            if (link.NXVisualBodiesHandles != null) counts.Visual = link.NXVisualBodiesHandles.Count;
#endif
            return counts;
        }

        private void PopulateTendonsSummaryGrid()
        {
            tendonsSummaryView.ShowRows(TendonSummary.Build(Exporter.Robot?.Tendons));
        }

        private void OnTendonRowSelected(object sender, TendonRowSelectedEventArgs e)
        {
            if (_isClosing || string.IsNullOrEmpty(e.TendonName) || Exporter.Robot?.Tendons == null)
            {
                return;
            }

            Tendon tendon = null;
            foreach (var t in Exporter.Robot.Tendons)
            {
                if (t.Name == e.TendonName)
                {
                    tendon = t;
                    break;
                }
            }

            if (tendon == null)
            {
                return;
            }

            CadBridge.CurrentTendonShown = tendon;
            CadBridge.TendonHighlightElementIndex = e.ElementIndex;
            CadBridge.IsShowingTendonVisualization = tendonsSummaryView.ShowSelectedTendon;
            CadBridge.TriggerGraphicsRedraw();
        }

        private void CollectLinkCsysNames(Link link, Dictionary<string, string> map)
        {
            if (link == null)
                return;

            if (!string.IsNullOrEmpty(link.Name) && !string.IsNullOrEmpty(link.Joint?.CoordinateSystemName))
                map[link.Name] = link.Joint.CoordinateSystemName;

            if (link.Children != null)
            {
                foreach (var child in link.Children)
                    CollectLinkCsysNames(child, map);
            }
        }

        private void OnKinematicsRowActivated(object sender, KinematicsRowActivatedEventArgs e)
        {
            // The double-click is still mid-flight: the DataGrid finishes the gesture by setting
            // the current cell and restoring focus to it on the dispatcher. That focus lands inside
            // the summary tab and drags the tab selection back, so the jump has to run after it.
            // Background priority is below the Input and Loaded work the grid queues.
            Dispatcher.BeginInvoke(
                new Action(() => JumpToKinematicsTarget(e)),
                System.Windows.Threading.DispatcherPriority.Background);
        }

        private void JumpToKinematicsTarget(KinematicsRowActivatedEventArgs e)
        {
            if (_isClosing)
            {
                return;
            }

            if (e.IsJointColumn)
            {
                NavigateToTab(0);
                SelectByLinkName(treeViewJointTree, e.LinkName);
            }
            else
            {
                NavigateToTab(1);
                SelectByLinkName(treeViewLinkProperties, e.LinkName);
            }
        }

        /// <summary>
        /// Selection is driven through the view model rather than the control, because
        /// TreeView.SelectedItem is read only.
        /// </summary>
        private static void SelectByLinkName(TreeView tree, string linkName)
        {
            if (tree.ItemsSource == null)
            {
                return;
            }

            foreach (LinkTreeItem root in tree.ItemsSource.OfType<LinkTreeItem>())
            {
                LinkTreeItem found = root.FindByLinkName(linkName);
                if (found == null)
                {
                    continue;
                }

                found.ExpandAncestors();
                found.IsSelected = true;
                return;
            }
        }

        private void DispatcherUnhandledException(
            object sender, System.Windows.Threading.DispatcherUnhandledExceptionEventArgs e)
        {
            logger.Error("Exception encountered in Assembly export form", e.Exception);
            MessageBox.Show("There was a problem with the export form: \n\"" +
                e.Exception.Message + "\"\nContact your maintainer with the log file found at " +
                Logger.GetLogFolder());

            // The host runs this add-in in-process with no WPF Application of its own.
            // An exception allowed to keep unwinding from here takes the CAD application down.
            e.Handled = true;
        }

        private void UnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            Exception ex = (Exception)e.ExceptionObject;
            logger.Error("Unhandled exception in Assembly Export form", ex);
            MessageBox.Show("There was a problem with the export form: \n\"" +
                ex.Message + "\"\nContact your maintainer with the log file found at " +
                Logger.GetLogFolder());
        }

        private void AssemblyExportFormLoad(object sender, RoutedEventArgs e)
        {
            FillJointTree();
            CadBridge.ShowHideVisualizations(checkBoxShowJointVisualization.IsChecked == true);
        }

        private string CheckJointsForErrors()
        {
            return RobotValidation.JointsMissingRequiredFields(BaseNode.Link);
        }

        private void ButtonCancelClick(object sender, EventArgs e)
        {
            if (TryTearDown())
            {
                Close();
            }
        }

        /// <summary>
        /// Runs the save prompt and tears down everything this form put into the CAD document.
        /// </summary>
        /// <returns>
        /// False if the user asked to cancel out of the save prompt, meaning the form should stay
        /// open. True once teardown has run, or if it had already run.
        /// </returns>
        private bool TryTearDown()
        {
            if (_isClosing)
            {
                return true;
            }

            if (CadBridge == null)
            {
                _isClosing = true;
                return true;
            }

            if (CadBridge.IsDocumentValid)
            {
                try
                {
                    if (SaveOnClose() == CAD.ConfigSaveResult.Cancelled)
                    {
                        return false;
                    }
                }
                catch (Exception ex)
                {
                    logger.Error("Exception caught while saving the configuration on close", ex);
                }
            }
            else
            {
                logger.Warning("Document is no longer open; closing without saving the configuration");
            }

            try
            {
                CadBridge.IsShowingJointGizmo = false;
                CadBridge.IsShowingTendonVisualization = false;
                CadBridge.IsShowingInertialGizmo = false;
                CadBridge.ShowHideVisualizations(false);
                Exporter.CleanUpTemporaryFeatures();
                CadBridge.UnselectAll();
            }
            catch (Exception ex)
            {
                // Best effort. Letting this escape would leave the form unclosable and, in an
                // in-process add-in, can take the host down.
                logger.Error("Exception caught while clearing visualizations on close", ex);
            }

            _isClosing = true;
            return true;
        }

        /// <summary>
        /// Saves the edits from the tab the user is currently on, prompting whether to keep them.
        /// </summary>
        private CAD.ConfigSaveResult SaveOnClose()
        {
            switch (_currentTabIndex)
            {
                case 0:
                    return SaveJointTabOnClose();
                default:
                    return SaveLinkTabOnClose();
            }
        }

        private void OnCloseForm(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (!TryTearDown())
            {
                e.Cancel = true;
            }
        }

        private void OnFormClosed(object sender, EventArgs e)
        {
            // Cleared here rather than while closing, because a close can still be cancelled from
            // the save prompt. If these leaked, CheckExportFormOpen would report a form open for
            // the rest of the session and try to activate a dead window.
            FormIsOpen = false;
            Instance = null;

            // Both were subscribed in the constructor. The AppDomain handler is a static root, so
            // leaving it attached keeps every form instance alive for the life of the host and
            // lets a stale form handle a later exception.
            Dispatcher.UnhandledException -= DispatcherUnhandledException;
            AppDomain.CurrentDomain.UnhandledException -= UnhandledException;

            // When this owned modeless form is destroyed, Windows transfers activation
            // away from the host and it ends up behind another window. Re-assert the
            // host CAD window as foreground so it stays focused.
            CadBridge?.RestoreHostForeground();
        }

        private CAD.ConfigSaveResult SaveJointTabOnClose()
        {
            if (previouslySelectedNode != null)
            {
                SaveJointDataFromPropertyBoxes(previouslySelectedNode.Link.Joint);
            }
            return SaveConfigurationFromTree(BaseNode, true, allowCancel: true);
        }

        private CAD.ConfigSaveResult SaveLinkTabOnClose()
        {
            if (previouslySelectedNode != null)
            {
                SaveLinkDataFromPropertyBoxes(previouslySelectedNode.Link);
                previouslySelectedNode.RefreshDisplayName();
            }
            return SaveConfigurationFromTree(BaseNode, true, allowCancel: true);
        }

        private void ButtonLinksFinishClick(object sender, EventArgs e)
        {
            FinishExport(true);
        }

        private void ButtonLinksExportUrdfOnlyClick(object sender, EventArgs e)
        {
            FinishExport(false);
        }

        private void ButtonExportLinkMeshClick(object sender, EventArgs e)
        {
            LinkTreeItem selectedNode = treeViewLinkProperties.SelectedItem as LinkTreeItem;
            if (selectedNode == null)
            {
                MessageBox.Show("Please select a link in the tree first.", "Export Link Mesh",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button1, MessageBoxOptions.DefaultDesktopOnly);
                return;
            }

            Link link = selectedNode.Link;

            List<MeshFormat> visualFormats = BuildVisualFormatList();
            List<MeshFormat> collisionFormats = BuildCollisionFormatList();

            if (visualFormats.Count == 0 && (!meshExportView.CollisionMeshing || collisionFormats.Count == 0))
            {
                MessageBox.Show("At least one mesh export format must be selected.", "Export Link Mesh",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button1, MessageBoxOptions.DefaultDesktopOnly);
                return;
            }

            string initialFolder = lastLinkMeshFolder;
            if (string.IsNullOrEmpty(initialFolder) || !Directory.Exists(initialFolder))
            {
                initialFolder = CadBridge.GetLatestExportLocation();
            }

            string outputFolder = FolderPicker.Show(
                new WindowInteropHelper(this).Handle,
                "Select output folder for link meshes",
                initialFolder);
            if (string.IsNullOrEmpty(outputFolder))
            {
                return;
            }
            lastLinkMeshFolder = outputFolder;

            ExporterMeshingOptions meshingOptions = BuildMeshingOptionsFromForm().ResolveFor(link);
            ExporterConfiguration config = SaveFormToExporterConfiguration();
            RobotPackage package = new RobotPackage("", "", config.folderStructure);

            Func<string, bool> confirmOverwrite = (filePath) =>
            {
                string fileName = Path.GetFileName(filePath);
                System.Windows.Forms.DialogResult result = MessageBox.Show(
                    $"File \"{fileName}\" already exists. Replace?",
                    "Export Link Mesh",
                    MessageBoxButtons.YesNo,
                    MessageBoxIcon.Question,
                    MessageBoxDefaultButton.Button1,
                    MessageBoxOptions.DefaultDesktopOnly);
                return result == System.Windows.Forms.DialogResult.Yes;
            };

            List<string> writtenPaths;
            this.Hide();
            try
            {
                writtenPaths = Exporter.ExportSingleLinkMeshes(
                    link,
                    package,
                    outputFolder,
                    visualFormats,
                    meshExportView.CollisionMeshing ? collisionFormats : new List<MeshFormat>(),
                    meshingOptions,
                    confirmOverwrite);
            }
            finally
            {
                this.Show();
            }

            if (writtenPaths.Count == 0)
            {
                MessageBox.Show($"No mesh files were written for \"{link.Name}\".", "Export Link Mesh",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button1, MessageBoxOptions.DefaultDesktopOnly);
                return;
            }

            ExportCompleteDialog.ShowMeshExportComplete(writtenPaths, owner: this);
        }

        private static string lastLinkMeshFolder;

        private List<MeshFormat> BuildVisualFormatList()
        {
            return meshExportView.CheckedVisualFormats;
        }

        private List<MeshFormat> BuildCollisionFormatList()
        {
            return meshExportView.CheckedCollisionFormats;
        }

        private ExporterMeshingOptions BuildMeshingOptionsFromForm()
        {
            CommitMeshingColumns();

            ExporterMeshingOptions meshingOptions = new ExporterMeshingOptions();
            // The robot-wide options
            meshingOptions.visualMeshingOptions = globalVisualMeshing.Clone();
            meshingOptions.collisionMeshingOptions = globalCollisionMeshing.Clone();
            meshingOptions.meshTagExtension = meshExportView.VisualExtension;
            meshingOptions.collisionMeshTagExtension = meshExportView.CollisionExtension;
            meshingOptions.exportCollision = meshExportView.CollisionMeshing;
            return meshingOptions;
        }

        private void WriteMeshingColumns(Export.MeshingOptions visual, Export.MeshingOptions collision)
        {
            meshExportView.Visual.Write(visual);
            meshExportView.Collision.Write(collision);
            meshExportView.ApplyEnableState();
        }

        private void MesherSettingChanged(object sender, EventArgs e)
        {
            meshExportView.ApplyEnableState();
            RefreshMeshingOverrideBold();
        }

        /// <summary>
        /// True when the same mesh format is requested from both the SuperDex and the CAD host
        /// exporter, which would have the two writes race for the same output file.
        /// </summary>
        private bool HasDuplicateExportMethod(bool collision)
        {
            Func<MeshFormat, bool> isChecked = collision
                ? (Func<MeshFormat, bool>)meshExportView.IsCollisionFormatChecked
                : meshExportView.IsVisualFormatChecked;

            return (isChecked(MeshFormat.stlOpenCascade) && isChecked(MeshFormat.stlSolidworks))
                || (isChecked(MeshFormat.glbOpenCascade) && isChecked(MeshFormat.glbCAD))
                || (isChecked(MeshFormat.objOpenCascade) && isChecked(MeshFormat.objCAD));
        }

        private void FinishExport(bool exportLinkMesh)
        {
            if (exportLinkMesh && meshExportView.CheckedVisualFormats.Count == 0)
            {
                MessageBox.Show("At least one Visual Link Export Format must be chosen!");
                return;
            }

            if (exportLinkMesh && meshExportView.CollisionMeshing
                && meshExportView.CheckedCollisionFormats.Count == 0)
            {
                MessageBox.Show("At least one Collision Link Export Format must be chosen when collision meshing is enabled!");
                return;
            }

            if (exportLinkMesh && HasDuplicateExportMethod(collision: false))
            {
                MessageBox.Show("Only one method of export per format is supported at a time (visual)!");
                return;
            }

            if (exportLinkMesh && meshExportView.CollisionMeshing
                && HasDuplicateExportMethod(collision: true))
            {
                MessageBox.Show("Only one method of export per format is supported at a time (collision)!");
                return;
            }

            ExporterMeshingOptions meshingOptions = BuildMeshingOptionsFromForm();

            logger.Information("Completing Robot export");
            SaveConfigurationFromTree(BaseNode, false);

            // Saving selected node
            LinkTreeItem node = treeViewLinkProperties.SelectedItem as LinkTreeItem;
            if (node != null)
            {
                SaveLinkDataFromPropertyBoxes(node.Link);
            }

            Exporter.Robot = CreateRobotFromTreeView();
            Exporter.PackageName = Exporter.Robot.Name;

            // The UI should prevent these sorts of errors, but just in case
            string errors = CheckLinksForErrors(Exporter.Robot.BaseLink);
            if (!string.IsNullOrWhiteSpace(errors))
            {
                logger.Information("Link errors encountered:\n " + errors);

                string message = "The following links contained errors in either their link or joint " +
                    "properties. Please address before continuing\r\n\r\n" + errors;
                MessageBox.Show(message, "SuperDex CAD Exporter - Link Errors");
                return;
            }

            string linksWithInvalidInertials = CheckLinksForValidInertials(Exporter.Robot.BaseLink);
            if (!string.IsNullOrEmpty(linksWithInvalidInertials))
            {
                logger.Information("Links without valid inertials encountered:\r\n" + linksWithInvalidInertials);

                string message = "The following links have possibly missing inertial properties. You may need to recalculate them." +
                "\r\n\r\nDo you want to proceed anyways?\r\n\r\n" + linksWithInvalidInertials;
                System.Windows.Forms.DialogResult result =
                    MessageBox.Show(message, "Links with invalid inertials detected", MessageBoxButtons.YesNo);

                if (result == System.Windows.Forms.DialogResult.No)
                {
                    logger.Information("Export canceled for user to review warnings");
                    return;
                }
            }

            string fileName = Path.GetFileNameWithoutExtension(Exporter.PackageName);

            bool retrievedLastSavePath = true;

            string initialDir = ShellMruDirectory.Get();
            if (string.IsNullOrEmpty(initialDir))
            {
                initialDir = CadBridge.GetLatestExportLocation();
            }

            if (string.IsNullOrEmpty(initialDir))
            {
                retrievedLastSavePath = false;
            }
            else
            {
                string candidate = Path.Combine(initialDir, fileName);
                if (Directory.Exists(candidate))
                {
                    int suffix = 1;
                    while (Directory.Exists($"{candidate} ({suffix})"))
                        suffix++;
                    fileName = $"{fileName} ({suffix})";
                }
            }

            SaveFileDialog saveFileDialog1 = new SaveFileDialog
            {
                InitialDirectory = initialDir ?? "",
                RestoreDirectory = !retrievedLastSavePath, // let Windows restore if we could't retrieve it ourselves
                FileName = fileName
            };

            bool saveResult = System.Windows.Forms.DialogResult.OK == saveFileDialog1.ShowDialog();
            saveFileDialog1.Dispose();
            if (saveResult)
            {
                string parentFolder = Path.GetDirectoryName(saveFileDialog1.FileName);
                string baseName = Path.GetFileName(saveFileDialog1.FileName);

                Exporter.SavePath = parentFolder;
                Exporter.PackageName = baseName;
                CadBridge.SetLatestExportLocation(Exporter.SavePath);

                string targetFolder = Path.Combine(parentFolder, baseName);
                logger.Information("Saving Robot package to " + targetFolder);

                List<MeshFormat> exportFormats = BuildVisualFormatList();
                List<MeshFormat> collisionExportFormats = BuildCollisionFormatList();

                this.WindowState = WindowState.Minimized;

                ExporterConfiguration config = SaveFormToExporterConfiguration();
                Exporter.ExportRobot(exportLinkMesh, exportFormats, collisionExportFormats, meshingOptions, config);

                string urdfPath = Exporter.URDFFileName;
                string mjcfPath = Exporter.MJCFFileName;
                string superdexBotPath = Exporter.SuperDexBotFileName;
                string superdexPrefabPath = Exporter.SuperDexPrefabFileName;

                ExportCompleteDialog dialog = new ExportCompleteDialog(
                    urdfPath, mjcfPath, superdexBotPath, superdexPrefabPath);
                dialog.Owner = this;
                if (dialog.ShowDialog() == true)
                {
                    Process.Start("explorer.exe", $"/select,\"{targetFolder}\"");
                    this.WindowState = WindowState.Minimized;
                }
                else
                {
                    this.WindowState = WindowState.Normal;
                    Focus();
                }
            }
        }

        private static string CheckLinksForErrors(Link baseLink)
        {
            return RobotValidation.LinksMissingRequiredFields(baseLink);
        }

        private static string CheckLinksForValidInertials(Link baseLink)
        {
            return RobotValidation.LinksWithInvalidInertials(baseLink);
        }

        private void TreeViewLinkPropertiesAfterSelect(
            object sender,
            RoutedPropertyChangedEventArgs<object> e)
        {
            LinkTreeItem node = e.NewValue as LinkTreeItem;
            if (node == null)
            {
                return;
            }

            if (previouslySelectedNode != null)
            {
                SaveLinkDataFromPropertyBoxes(previouslySelectedNode.Link);
            }

            // Committed against the link the columns were actually showing, which is tracked
            // separately from previouslySelectedNode because the joint tree shares that field.
            CommitMeshingColumns();

            if (checkBoxLinkHighlights.IsChecked == true)
            {
                CadBridge.SelectLinkComponents(node.Link);
            }

            FillLinkPropertyBoxes(node.Link);
            treeViewLinkProperties.Focus();
            previouslySelectedNode = node;

            CadBridge.CurrentLinkNodeShown = node.Link;
            CadBridge.TriggerGraphicsRedraw();
        }

        /// <summary>
        /// Validates text entry for numerical text boxes to limit improper input. It's not perfect
        /// because you can still copy and paste bad input into the fields, but that's addressed
        /// elsewhere
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void TreeViewJointtreeAfterSelect(
            object sender,
            RoutedPropertyChangedEventArgs<object> e)
        {
            LinkTreeItem node = e.NewValue as LinkTreeItem;
            if (node == null)
            {
                return;
            }

            if (previouslySelectedNode != null && !previouslySelectedNode.IsBaseNode)
            {
                SaveJointDataFromPropertyBoxes(previouslySelectedNode.Link.Joint);
            }

            if (checkBoxShowJointHighlights.IsChecked == true)
            {
                CadBridge.SelectJointComponents(node.Link);
            }

            FillJointPropertyBoxes(node.Link.Joint);
            previouslySelectedNode = node;

            CadBridge.IsShowingJointGizmo = true;
            CadBridge.CurrentNodeShown = node.Link;

            CadBridge.TriggerGraphicsRedraw();
        }

        public void LoadMeshingOptionsToForm(ExporterConfiguration config)
        {
            if (config == null)
            {
                config = new ExporterConfiguration(); // Use defaults
            }

            // Commit first: with an override on screen the columns hold that link's values, and
            // replacing the globals underneath them would drop the edit rather than save it.
            CommitMeshingColumns();

            globalVisualMeshing = config.visualMeshingOptions.Clone();
            globalCollisionMeshing = config.collisionMeshingOptions.Clone();

            ShowMeshingColumnsForSelectedLink();
        }

        public void LoadExporterConfigurationToForm(ExporterConfiguration config)
        {
            if (config == null)
            {
                config = new ExporterConfiguration();
            }

            LoadMeshingOptionsToForm(config);

            meshExportView.ShowVisualFormats(config.meshFormats);

            // A configuration saved before collision formats existed reuses the visual ones.
            meshExportView.ShowCollisionFormats(
                config.hasCollisionMeshFormats ? config.collisionMeshFormats : config.meshFormats);

            SanitizeCADSpecificCheckboxes();

            meshExportView.VisualExtension = config.meshTagExtension;
            meshExportView.CollisionExtension = config.collisionMeshTagExtension;
            meshExportView.CollisionMeshing = config.enableCollisionMeshing;
            meshExportView.ApplyEnableState();

            string documentTitle = CadBridge.GetDocumentTitle();

            textBoxRobotName.Text = string.IsNullOrEmpty(config.robotName) ? Path.GetFileNameWithoutExtension(documentTitle) : config.robotName;

            meshExportView.FolderStructure = config.folderStructure.ToString();
        }

        public ExporterConfiguration SaveFormToExporterConfiguration()
        {
            CommitMeshingColumns();

            ExporterConfiguration config = new ExporterConfiguration();

            config.visualMeshingOptions = globalVisualMeshing.Clone();
            config.collisionMeshingOptions = globalCollisionMeshing.Clone();

            config.meshFormats = new List<MeshFormat>();

            if (meshExportView.IsVisualFormatChecked(MeshFormat.glbOpenCascade))
                config.meshFormats.Add(MeshFormat.glbOpenCascade);
            if (meshExportView.IsVisualFormatChecked(MeshFormat.objOpenCascade))
                config.meshFormats.Add(MeshFormat.objOpenCascade);
            if (meshExportView.IsVisualFormatChecked(MeshFormat.stlOpenCascade))
                config.meshFormats.Add(MeshFormat.stlOpenCascade);
            if (meshExportView.IsVisualFormatChecked(MeshFormat.stlSolidworks))
                config.meshFormats.Add(MeshFormat.stlSolidworks);
            if (meshExportView.IsVisualFormatChecked(MeshFormat.stepSolidworks))
                config.meshFormats.Add(MeshFormat.stepSolidworks);
            if (meshExportView.IsVisualFormatChecked(MeshFormat.glbCAD))
                config.meshFormats.Add(MeshFormat.glbCAD);
            if (meshExportView.IsVisualFormatChecked(MeshFormat.objCAD))
                config.meshFormats.Add(MeshFormat.objCAD);

            config.meshTagExtension = meshExportView.VisualExtension;
            config.collisionMeshTagExtension = meshExportView.CollisionExtension;
            config.enableCollisionMeshing = meshExportView.CollisionMeshing;

            config.collisionMeshFormats = new List<MeshFormat>();

            if (meshExportView.IsCollisionFormatChecked(MeshFormat.glbOpenCascade))
                config.collisionMeshFormats.Add(MeshFormat.glbOpenCascade);
            if (meshExportView.IsCollisionFormatChecked(MeshFormat.objOpenCascade))
                config.collisionMeshFormats.Add(MeshFormat.objOpenCascade);
            if (meshExportView.IsCollisionFormatChecked(MeshFormat.stlOpenCascade))
                config.collisionMeshFormats.Add(MeshFormat.stlOpenCascade);
            if (meshExportView.IsCollisionFormatChecked(MeshFormat.stlSolidworks))
                config.collisionMeshFormats.Add(MeshFormat.stlSolidworks);
            if (meshExportView.IsCollisionFormatChecked(MeshFormat.stepSolidworks))
                config.collisionMeshFormats.Add(MeshFormat.stepSolidworks);
            if (meshExportView.IsCollisionFormatChecked(MeshFormat.glbCAD))
                config.collisionMeshFormats.Add(MeshFormat.glbCAD);
            if (meshExportView.IsCollisionFormatChecked(MeshFormat.objCAD))
                config.collisionMeshFormats.Add(MeshFormat.objCAD);

            config.hasCollisionMeshFormats = true;

            config.robotName = textBoxRobotName.Text;

            FolderStructure parsedFolderStructure;
            config.folderStructure =
                Enum.TryParse(meshExportView.FolderStructure, out parsedFolderStructure)
                    ? parsedFolderStructure
                    : default(FolderStructure);

            return config;
        }

        private void ComboBoxExporterConfigurationPreset_SelectedIndexChanged(object sender, EventArgs e)
        {
            var selected = visualPropertiesView.SelectedPreset;
            // Ignore separator
            if (selected is string str)
            {
                if (str == SavePresetOption)
                {
                    SaveCurrentAsPreset();
                    // Reset selection to previous or default
                    visualPropertiesView.SelectedPresetIndex = 0;
                }
                return;
            }
            if (selected is ExporterConfigurationPreset preset)
            {
                ExporterConfiguration newConfig = preset.Configuration;
                newConfig.robotName = textBoxRobotName.Text;
                LoadExporterConfigurationToForm(newConfig);
            }
        }

        private void SaveCurrentAsPreset()
        {
            // Get current config from form
            ExporterConfiguration config = SaveFormToExporterConfiguration();

            InputDialog dialog = new InputDialog("Save Preset", "Enter preset name:");
            dialog.Owner = this;
            if (dialog.ShowDialog() != true || string.IsNullOrWhiteSpace(dialog.InputText))
            {
                return;
            }

            string presetName = PresetStore.SaveUserPreset(dialog.InputText.Trim(), config);

            // Refresh the comboBox
            PopulatePresetComboBox();
            MessageBox.Show($"Preset '{presetName}' saved.\n\nYou can share your presets from  {PresetStore.GetPresetsDirectory()}", "Success",
                MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private const string SavePresetOption = "(Save Preset...)";

        private void PopulatePresetComboBox()
        {
            visualPropertiesView.ClearPresets();
            visualPropertiesView.AddPreset("");
            // Add predefined presets
            foreach (var preset in ExporterConfigurationPresets.Predefined)
            {
                visualPropertiesView.AddPreset(preset);
            }
            // Add user presets
            var userPresets = PresetStore.LoadUserPresets();
            if (userPresets.Count > 0)
            {
                visualPropertiesView.AddPreset("─── User Presets ───"); // Separator
                foreach (var preset in userPresets)
                {
                    visualPropertiesView.AddPreset(preset);
                }
            }
            // Add save option at the end
            visualPropertiesView.AddPreset(SavePresetOption);
            // Select default
            visualPropertiesView.SelectedPresetIndex = 0;
        }
    }
}
