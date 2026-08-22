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

#if SOLIDWORKS

using System;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;
using SolidWorks.Interop.swpublished;

using CADRobotExporter.UI;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.Utilities;
using CADRobotExporter.RobotExport;
using CADRobotExporter.SW;

namespace CADRobotExporter.CAD
{
    [ComVisible(true)]
    [Serializable]
    public sealed partial class ExportPropertyManager : PropertyManagerPage2Handler9
    {
        #region class variables

        private static readonly Serilog.ILogger logger = Logger.GetLogger();
        public SldWorks swApp;
        public ModelDoc2 model;

        [NonSerialized]
        public ExportHelper Exporter;
        [NonSerialized]
        public LinkNode previouslySelectedNode;
        [NonSerialized]
        public List<Link> linksToVisit;
        [NonSerialized]
        public LinkNode rightClickedNode;
        private readonly ContextMenuStrip docMenu;

        //General objects required for the PropertyManager page

        private readonly PropertyManagerPage2 PMPage;
        private PropertyManagerPageTab PMTabKinematics;
        private PropertyManagerPageTab PMTabTendons;
        private PropertyManagerPageGroup PMGroup;
        private PropertyManagerPageSelectionbox PMLinkVisualComponentsSelection;
        private PropertyManagerPageSelectionbox PMLinkCollisionComponentsSelection;
        private PropertyManagerPageSelectionbox PMLinkInertialComponentsSelection;
        private PropertyManagerPageSelectionbox PMCoordinateSystemSelection;
        private PropertyManagerPageSelectionbox PMRefAxisSelection;
        private PropertyManagerPageButton PMButtonExport;
        private PropertyManagerPageButton PMButtonLoad;
        private PropertyManagerPageTextbox PMTextBoxLinkName;
        private PropertyManagerPageTextbox PMTextBoxJointName;
        private PropertyManagerPageNumberbox PMNumberboxSerialChainCount; // repurposed
        private PropertyManagerPageCheckbox PMComputeMassInertia;
        private PropertyManagerPageCheckbox PMComputeVisualCollision;
        private PropertyManagerPageCheckbox PMComputeJointKinematics;
        private PropertyManagerPageCheckbox PMComputeJointLimits;

        private PropertyManagerPageOption PMOptionRevoluteJoint;
        private PropertyManagerPageOption PMOptionPrismaticJoint;
        private PropertyManagerPageOption PMOptionFixedJoint;

        private PropertyManagerPageLabel PMLabelParentLink;
        private PropertyManagerPageLabel PMLabelParentLinkName;
        private PropertyManagerPageLabel PMLabelLinkName;
        private PropertyManagerPageLabel PMLabelJointName;
        private PropertyManagerPageLabel PMLabelAxes;
        private PropertyManagerPageLabel PMLabelCoordSys;
        private PropertyManagerPageLabel PMLabelJointType;
        private PropertyManagerPageLabel PMLabelCSVFilename;
        private PropertyManagerPageLabel PMLabelLinkVisualComponents;
        private PropertyManagerPageLabel PMLabelLinkCollisionComponents;
        private PropertyManagerPageLabel PMLabelLinkInertialComponents;
        private PropertyManagerPageLabel PMLabelSerialChainCount;
        private PropertyManagerPageLabel PMLabelKinematicTree;

        private PropertyManagerPageGroup PMGroupTreeTools;
        private PropertyManagerPageGroup PMGroupTree;
        private PropertyManagerPageGroup PMGroupTopButtons;

        private PropertyManagerPageCheckbox PMRefAxisFlipCheckbox;

        private PropertyManagerPageOption PMOptionAxisRefAxis;
        private PropertyManagerPageOption PMOptionAxisX;
        private PropertyManagerPageOption PMOptionAxisY;
        private PropertyManagerPageOption PMOptionAxisZ;
        private PropertyManagerPageOption PMOptionLinkTypeLink;
        private PropertyManagerPageOption PMOptionLinkTypeSite;

        private PropertyManagerPageBitmapButton PMButtonOkay;
        private PropertyManagerPageBitmapButton PMButtonCancel;

        private PropertyManagerPageCheckbox PMCheckboxLinkPurelyInertial;
        private PropertyManagerPageCheckbox PMCheckboxLinkPurelyVisual;

        private PropertyManagerPageCheckbox PMCheckboxShowExtraCaptions;

        private PropertyManagerPageButton PMButtonCreateSerialChain;
        private PropertyManagerPageButton PMButtonInsertParentNode;
        private PropertyManagerPageButton PMButtonInsertChildNode;
        private PropertyManagerPageButton PMButtonImportTree;
        private PropertyManagerPageButton PMButtonExportTree;

        private PropertyManagerPageCheckbox PMCheckboxAutoAdvance;
        private PropertyManagerPageCheckbox PMCheckboxAutoJointNaming;

        private PropertyManagerPageWindowFromHandle PMTree;

        public TreeView Tree
        { get; set; }

        private bool automaticallySwitched = false;
        private Dictionary<string, int> _linkCounters = new Dictionary<string, int>();

        //Each object in the page needs a unique ID

        private const int GroupID = 1;
        private const int TextBoxLinkNameID = 2;
        private const int LinkVisualComponentsSelectionID = 3;
        private const int LinkCollisionComponentsSelectionID = 4;
        private const int LinkInertialComponentsSelectionID = 5;
        private const int NumberboxSerialChainCountID = 7; // repurposed
        private const int LabelParentLinkID = 8;
        private const int LabelJointNameID = 14;
        private const int dotNetTree = 16;
        private const int ButtonExportID = 17;
        private const int LabelAxesID = 20;
        private const int LabelCoordSysID = 21;
        private const int LoadConfigurationID = 26;
        private const int ComputeMassInertiaID = 27;
        private const int ComputeVisualCollisionID = 28;
        private const int ComputeJointKinematicsID = 29;
        private const int ComputeJointLimitsID = 30;
        private const int LoadedCSVFilenameID = 31;
        private const int TextBoxJointNameID = 32;
        private const int LabelParentLinkNameID = 33;
        private const int LabelLinkNameID = 34;
        private const int LabelLinkVisualComponentsID = 35;
        private const int LabelSerialChainCountID = 36;
        private const int LabelKinematicTreeID = 37;
        private const int CoordinateSystemSelectionID = 38;
        private const int RefAxisSelectionID = 39;

        private const int OptionRevoluteJointID = 40;
        private const int OptionPrismaticJointID = 41;
        private const int OptionFixedJointID = 42;

        private const int LabelLinkCollisionComponentsID = 44;
        private const int RefAxisFlipID = 45;
        private const int LabelLinkInertialComponentsID = 46;

        private const int GroupTreeToolsID = 47;
        private const int GroupTreeID = 48;

        private const int GroupTopButtonsID = 49;
        private const int ButtonOkayID = 50;
        private const int ButtonCancelID = 51;

        private const int CheckboxLinkPurelyInertiallID = 52;
        private const int CheckboxLinkNoPurelyVisualID = 53;

        private const int CheckboxShowExtraCaptionsID = 54;

        private const int ButtonCreateSerialChainID = 55;
        private const int ButtonInsertParentNodeID = 56;
        private const int ButtonInsertChildNodeID = 57;
        private const int ButtonImportTreeID = 59;
        private const int ButtonExportTreeID = 60;

        private const int CheckboxAutoAdvanceID = 58;
        private const int CheckboxAutoJointNamingID = 125;
        private const int OptionAxisRefAxisID = 61;
        private const int OptionAxisXID = 62;
        private const int OptionAxisYID = 63;
        private const int OptionAxisZID = 64;

        // Tab IDs
        private const int TabKinematicsID = 100;
        private const int TabTendonsID = 101;

        // Tendon control IDs
        private const int GroupTendonsID = 102;
        private const int GroupRoutingID = 103;
        private const int TextBoxTendonNameID = 104;
        private const int ButtonAddTendonID = 105;
        private const int ButtonAddRoutingID = 106;
        private const int SelectionBoxPointID = 107;
        private const int NumberBoxCoefficientID = 108;
        private const int ComboElementTypeID = 109;
        private const int ComboParentLinkID = 110;
        private const int CheckboxAutoAddWaypointID = 111;
        private const int CheckboxAutoLinkID = 112;
        private const int WindowTendonListID = 113;
        private const int WindowRoutingTableID = 114;
        private const int ButtonRemoveTendonID = 115;
        private const int ButtonRemoveRoutingID = 116;
        private const int LabelTendonNameID = 117;
        private const int LabelPointSelectionID = 118;
        private const int LabelCoefficientID = 119;
        private const int LabelElementTypeID = 120;
        private const int LabelTendonParentLinkID = 121;

        private const int LabelJointTypeID = 122;
        private const int OptionLinkTypeLinkID = 123;
        private const int OptionLinkTypeSiteID = 124;

        private const string LabelLinkVisualCaption = "Link Visual Components (Optional)";
        private const string LabelLinkVisualExtraCaption = "\nUse for visual meshes, recommend selecting individual parts";

        private const string LabelLinkCollisionCaption = "Link Collision Components (Optional)";
        private const string LabelLinkCollisionExtraCaption = "\nUse for simplified geometry of complex components\n" +
            "Will be generated from visuals, then inertials if not selected";

        private const string LabelLinkInertialCaption = "Link Inertial Components (Optional)";
        private const string LabelLinkInertialExtraCaption = "\nUse for accurate inertial properties, recommned selecting whole subassemblies from Feature Tree";

        private const string CheckboxLinkPurelyInertialCaption = "Purely Inertial Link";
        private const string CheckboxLinkPurelyInertialExtraCaption = ". If no Visual components,\n" +
                "do not create visuals from Inertial components\n" +
                "do not create collision from Inertial components";

        private const string CheckboxLinkPurelyVisualCaption = "Purely Visual Link";
        private const string CheckboxLinkPurelyVisualExtraCaption = ". If no Inertial components,\n" +
                "do not calculate inertial properties from Visual components\n" +
                "do not create collision from Visual components";

        private const string LabelKinematicTreeCaption = "Kinematic Tree";
        private const string LabelKinematicTreeExtraCaption = "\nLinks can be dragged and dropped for reparenting and reordering.";

        [NonSerialized]
        private LinkNode _rootNode;

        #endregion class variables

        public void Show()
        {
            PMPage.Show2(0);
            PMCheckboxLinkPurelyInertial.Caption = CheckboxLinkPurelyInertialCaption;
            PMCheckboxLinkPurelyVisual.Caption = CheckboxLinkPurelyVisualCaption;
            Tree.AfterSelect += new TreeViewEventHandler(TreeAfterSelect);
            if (_rootNode != null)
            {
                SwitchActiveNodes(_rootNode);
            }
            automaticallySwitched = false;
        }

        public void Close(bool Okay)
        {
            try
            {
                logger.Information("Closing PropertyManger Page");
                SaveActiveNode();
                Exporter.CadBridge.SaveConfigurationFromTree(null, (LinkNode)Tree.Nodes[0], !Okay, "");

                // Save tendons alongside the kinematic tree
                if (Okay)
                {
                    var bridge = (SolidworksBridge)Exporter.CadBridge;
                    bridge.SaveTendons(tendons);
                }
            }
            catch (Exception e)
            {
                logger.Error("Exception caught on close ", e);
                MessageBox.Show("There was a problem prior to closing the property manager: \n\"" +
                    e.Message + "\"\nPlease check the log file at " + Logger.GetLogFolder());
            }

            PMPage.Close(Okay);
        }

        //The following runs when a new instance of the class is created
        public ExportPropertyManager(SldWorks swAppPtr)
        {
            swApp = swAppPtr;
            model = swApp.ActiveDoc;

            SolidworksBridge cadBridge = new SolidworksBridge(swApp, model);

            Exporter = new ExportHelper(cadBridge);
            Exporter.Robot = new Robot();
            Exporter.Robot.Name = "";

            linksToVisit = new List<Link>();
            docMenu = new ContextMenuStrip();

            string caption = null;
            string tip = null;
            int longerrors = 0;
            int controlType = 0;
            int alignment = 0;

            model.ShowConfiguration2("SuperDex CAD Exporter");

            #region Create and instantiate components of PM page

            //Set the variables for the page
            string PageTitle = "SuperDex CAD Exporter " + Versioning.VersionString.Get() + " - Setup";
            long options = (int)swPropertyManagerPageOptions_e.swPropertyManagerOptions_HandleKeystrokes
                + (int)swPropertyManagerPageOptions_e.swPropertyManagerOptions_LockedPage
                + (int)swPropertyManagerPageOptions_e.swPropertyManagerOptions_DisablePageBuildDuringHandlers;

            //Create the PropertyManager page
            PMPage = (PropertyManagerPage2)swApp.CreatePropertyManagerPage(
                PageTitle, (int)options, this, ref longerrors);

            //Make sure that the page was created properly
            if (longerrors == (int)swPropertyManagerPageStatus_e.swPropertyManagerPage_Okay)
            {
                SetupPropertyManagerPage(ref caption, ref tip, ref options,
                    ref controlType, ref alignment);
            }
            else
            {
                //If the page is not created
                logger.Error("An error occurred while attempting to create the PropertyManager Page\nError: " + longerrors);
                MessageBox.Show("There was a problem setting up the property manager: " +
                    "\nContact your maintainer with the log file found at " + Logger.GetLogFolder());
            }

            #endregion Create and instantiate components of PM page

            var selectionManager = model.ISelectionManager;
            selectionManager.SelectionColor[2] = (int)swUserPreferenceIntegerValue_e.swSystemColorsSelectedItem4;
            selectionManager.SelectionColor[4] = (int)swUserPreferenceIntegerValue_e.swSystemColorsDynamicHighlight;
        }

        private void ExceptionHandler(object sender, ThreadExceptionEventArgs e)
        {
            logger.Warning("Exception encountered in URDF configuration form\n" +
                "Contact your maintainer with the log file found at " + Logger.GetLogFolder(),
                e.Exception);
        }

        private void UnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            logger.Error("Unhandled exception in URDF configuration form\n" +
                "Contact your maintainer with the log file found at " + Logger.GetLogFolder(),
                (Exception)e.ExceptionObject);
        }

        #region Implemented Property Manager Page Handler Methods

        void IPropertyManagerPage2Handler9.AfterActivation()
        {
            //Turns the selection box blue so that selected components are added to the PMPage
            // selection box
            PMPage.SetFocus(CoordinateSystemSelectionID);
        }

        private void ExportButtonPress()
        {
            SaveActiveNode();
            Exporter.CadBridge.SaveConfigurationFromTree(null, (LinkNode)Tree.Nodes[0], false, "");

            if (!CheckAllLinksHaveCoordinateSystems((LinkNode)Tree.Nodes[0]))
            {
                return;
            }

            if (!CheckAllLinksHaveRefAxes((LinkNode)Tree.Nodes[0]))
            {
                return;
            }

            if (CheckIfNamesAreUnique((LinkNode)Tree.Nodes[0]) && CheckNodesComplete(Tree))
            {
                //It saves automatically when sending Okay as true;
                PMPage.Close(true);
                AssemblyDoc assy = (AssemblyDoc)model;

                //This call can be a real sink of processing time if the model is large.
                //Unfortunately there isn't a way around it I believe.
                int result = assy.ResolveAllLightWeightComponents(true);

                // If the user confirms to resolve the components and they are successfully
                // resolved we can continue
                if (result == (int)swComponentResolveStatus_e.swResolveOk)
                {
                    List<string> unresolvedComponents = new List<string>();
                    CheckModelDocsExist((LinkNode)Tree.Nodes[0], unresolvedComponents);
                    if (unresolvedComponents.Count > 0)
                    {
                        string componentNames = string.Join("\r\n", unresolvedComponents);
                        logger.Error("SolidWorks told us the resolve succeeded, but ModelDocs" +
                            " could not be obtained for: " + componentNames);
                        MessageBox.Show("Model Documents could not be obtained for the following" +
                            " components. Please resolve them:\r\n" + componentNames);
                        return;
                    }

                    // Builds the links and joints from the PMPage configuration
                    LinkNode BaseNode = (LinkNode)Tree.Nodes[0];
                    automaticallySwitched = true;
                    Tree.Nodes.Remove(BaseNode);

                    bool exportSuccess = Exporter.CreateRobotFromTreeView(BaseNode);
                    if (exportSuccess)
                    {
                        if (tendons.Count > 0)
                        {
                            Exporter.Robot.Tendons.Clear();
                            Exporter.Robot.Tendons.AddRange(tendons);
                            Exporter.LocalizeTendonPositions();
                        }

                        AssemblyExportForm exportForm = new AssemblyExportForm(BaseNode, Exporter);
                        exportForm.Exporter = Exporter;

                        // Own the form to the SolidWorks main window so it stays above
                        // SolidWorks (but not every application) and hides/restores with it.
                        // Use the frame handle from the SolidWorks API rather than
                        // Process.MainWindowHandle, which can resolve to the wrong top-level
                        // window and cause SolidWorks to lose focus/minimize when the form closes.
                        IntPtr swHwnd = new IntPtr(((Frame)swApp.Frame()).GetHWnd());
                        if (swHwnd != IntPtr.Zero)
                        {
                            exportForm.Show(new NativeWindowWrapper(swHwnd));
                        }
                        else
                        {
                            exportForm.Show();
                        }
                        exportForm.Focus();
                    }
                }
                else if (result == (int)swComponentResolveStatus_e.swResolveError ||
                    result == (int)swComponentResolveStatus_e.swResolveNotPerformed)
                {
                    logger.Warning("Resolving components failed. Warning user to do so on their own");
                    MessageBox.Show("Resolving components failed. In order for export to succeed, " +
                        " this tool needs all components to be resolved. Try resolving " +
                        "lightweight components manually before attempting to export again");
                }
                else if (result == (int)swComponentResolveStatus_e.swResolveAbortedByUser)
                {
                    logger.Warning("Components were not resolved by user");
                    MessageBox.Show("In order for export to succeed, this tool needs all " +
                        "components to be resolved. You can resolve them manually or try " +
                        "exporting again");
                }
            }
        }

        private void EnableControl(IPropertyManagerPageControl control, bool isEnabled = true)
        {
            control.Enabled = isEnabled;
            control.Visible = true;
        }

        private LinkNode LinkNodeFromTreeViewItem(System.Windows.Controls.TreeViewItem item)
        {
            Link itemLink = (Link)item.Tag;
            LinkNode node = new LinkNode
            {
                Link = itemLink,
                Name = itemLink.Name,
                Text = itemLink.Name
            };
            node.IsBaseNode = item.Parent.GetType() != typeof(System.Windows.Controls.TreeViewItem);
            foreach (System.Windows.Controls.TreeViewItem child in item.Items)
            {
                node.Nodes.Add(LinkNodeFromTreeViewItem(child));
            }
            return node;
        }

        private void OnButtonPress(int Id)
        {
            switch (Id)
            {
                case ButtonExportID:
                    ExportButtonPress();
                    break;
                case ButtonCancelID:
                    Close(false);
                    break;
                case ButtonOkayID:
                    Close(true);
                    break;
                case ButtonCreateSerialChainID:
                    CreateSerialChain((int)PMNumberboxSerialChainCount.Value);
                    break;
                case ButtonInsertParentNodeID:
                    InsertParentNode();
                    break;
                case ButtonInsertChildNodeID:
                    CreateChildNode();
                    break;
                case ButtonImportTreeID:
                    ImportTreeFromFile();
                    break;
                case ButtonExportTreeID:
                    ExportTreeToFile();
                    break;
                case ButtonAddTendonID:
                    OnAddTendonClicked();
                    break;
                case ButtonRemoveTendonID:
                    OnRemoveTendonClicked();
                    break;
                case ButtonAddRoutingID:
                    OnAddRoutingElementClicked();
                    break;
                case ButtonRemoveRoutingID:
                    OnRemoveRoutingElementClicked();
                    break;
                default:
                    break;
            }
        }

        void IPropertyManagerPage2Handler9.OnButtonPress(int Id)
        {
            try
            {
                OnButtonPress(Id);
            }
            catch (Exception e)
            {
                logger.Error("Exception caught handling button press " + Id, e);
                MessageBox.Show("There was a problem with the configuration property manager: \n\"" +
                    e.Message + "\"\nContact your maintainer with the log file found at " + Logger.GetLogFolder());
            }
        }

        void IPropertyManagerPage2Handler9.OnClose(int Reason)
        {
            // we've disabled the native OK/Close buttons and implemented our own
            // because OnClose is called after selection controls are already destroyed

            //This function must contain code, even if it does nothing, to prevent the
            //.NET runtime environment from doing garbage collection at the wrong time.
            int IndentSize;
            IndentSize = System.Diagnostics.Debug.IndentSize;
            System.Diagnostics.Debug.WriteLine(IndentSize);
        }

        void IPropertyManagerPage2Handler9.OnGainedFocus(int Id)
        {
        }

        bool IPropertyManagerPage2Handler9.OnHelp()
        {
            System.Diagnostics.Process.Start("https://facebookresearch.github.io/project_superdex/studio/docs/cad_exporter/");
            return true;
        }

        bool IPropertyManagerPage2Handler9.OnKeystroke(int Wparam, int Message, int Lparam, int Id)
        {
            if (Wparam == (int)Keys.Enter || Wparam == (int)Keys.Escape)
            {
                return true;
            }
            return false;
        }

        void IPropertyManagerPage2Handler9.OnLostFocus(int Id)
        {
            Debug.Print("Control box " + Id + " has lost focus");
        }

        void IPropertyManagerPage2Handler9.OnNumberboxChanged(int Id, double Value)
        {
            if (Id == NumberBoxCoefficientID)
            {
                OnCoefficientChanged(Value);
            }
        }

        void IPropertyManagerPage2Handler9.OnSelectionboxFocusChanged(int Id)
        {
            Debug.Print("The focus has moved to selection box " + Id);
        }

        void IPropertyManagerPage2Handler9.OnSelectionboxListChanged(int Id, int Count)
        {
            // Move focus to next selection box if right-mouse button pressed
            PMPage.SetCursor((int)swPropertyManagerPageCursors_e.swPropertyManagerPageCursors_Advance);

            if (Id == SelectionBoxPointID)
            {
                OnRoutingPointSelected();
            }
            else if (Id == CoordinateSystemSelectionID && PMCheckboxAutoAdvance.Checked)
            {
                if (previouslySelectedNode.IsBaseNode || previouslySelectedNode.Link.Joint.Type == "fixed")
                {
                    PMPage.SetFocus(LinkInertialComponentsSelectionID);
                }
                else
                {
                    PMPage.SetFocus(RefAxisSelectionID);
                }
            }

            if (Id == RefAxisSelectionID && PMCheckboxAutoAdvance.Checked)
            {
                PMPage.SetFocus(LinkInertialComponentsSelectionID);
            }

            if (Id == LinkVisualComponentsSelectionID)
            {
                PMCheckboxLinkPurelyInertial.Checked = false;
                ((PropertyManagerPageControl)PMCheckboxLinkPurelyInertial).Enabled = false;
            }

            if (Id == LinkInertialComponentsSelectionID)
            {
                PMCheckboxLinkPurelyVisual.Checked = false;
                ((PropertyManagerPageControl)PMCheckboxLinkPurelyVisual).Enabled = false;
            }

            if (PMLinkInertialComponentsSelection.ItemCount == 0 && PMLinkVisualComponentsSelection.ItemCount > 0)
            {
                ((PropertyManagerPageControl)PMCheckboxLinkPurelyVisual).Enabled = true;
            }

            if (PMLinkVisualComponentsSelection.ItemCount == 0 && PMLinkInertialComponentsSelection.ItemCount > 0)
            {
                ((PropertyManagerPageControl)PMCheckboxLinkPurelyInertial).Enabled = true;
            }
        }

        bool IPropertyManagerPage2Handler9.OnSubmitSelection(
            int Id, object Selection, int SelType, ref string ItemText)
        {
            // This method must return true for selections to occur
            return true;
        }

        void IPropertyManagerPage2Handler9.OnTextboxChanged(int Id, string Text)
        {
            if (Id == TextBoxLinkNameID)
            {
                LinkNode node = (LinkNode)Tree.SelectedNode;
                node.Text = PMTextBoxLinkName.Text;
                node.Name = PMTextBoxLinkName.Text;
                SyncCountersWithTree(_rootNode);

                if (PMCheckboxAutoJointNaming.Checked && !node.IsBaseNode && !node.Link.isSite)
                {
                    string jointName = LinkNameToJointName(PMTextBoxLinkName.Text);
                    node.Link.Joint.Name = jointName;
                    PMTextBoxJointName.Text = jointName;
                }
            }
            else if (Id == TextBoxTendonNameID)
            {
                OnTendonNameChanged();
            }
        }

        int IPropertyManagerPage2Handler9.OnWindowFromHandleControlCreated(int Id, bool Status)
        {
            return 0;
        }

        private static string LinkNameToJointName(string linkName)
        {
            string result = Regex.Replace(linkName, @"(?<=[_\-])link(?=[_\-]|$)|^link(?=[_\-])", m =>
            {
                if (m.Value == "LINK") return "JOINT";
                if (m.Value == "Link") return "Joint";
                return "joint";
            }, RegexOptions.IgnoreCase);

            if (result != linkName)
                return result;

            result = Regex.Replace(linkName, @"(?<=\p{Ll})Link|^Link(?=\p{Lu}|\p{Ll})|^link(?=\p{Lu})", m =>
                m.Value[0] == 'L' ? "Joint" : "joint");

            if (result != linkName)
                return result;

            return "joint_" + linkName;
        }

        #endregion Implemented Property Manager Page Handler Methods

        #region TreeView handler methods

        // Upon selection of a node, the node displayed on the PMPage is saved and the
        // selected one is then set
        private void TreeAfterSelect(object sender, TreeViewEventArgs e)
        {
            try
            {
                if (!automaticallySwitched && e.Node != null)
                {
                    SwitchActiveNodes((LinkNode)e.Node);
                    if (PMCheckboxAutoAdvance.Checked)
                    {
                        PMPage.SetFocus(CoordinateSystemSelectionID);
                    }
                }
                automaticallySwitched = false;
            }
            catch (Exception ex)
            {
                logger.Error("Exception caught on tree view AfterSelect ", ex);
                MessageBox.Show("There was a problem with the property manager: \n\"" +
                    ex.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
        }

        // Captures which node was right clicked
        private void TreeNodeMouseClick(object sender, TreeNodeMouseClickEventArgs e)
        {
            rightClickedNode = (LinkNode)e.Node;
        }

        //When a keyboard key is pressed on the tree
        private void TreeKeyDown(object sender, KeyEventArgs e)
        {
            if (rightClickedNode.IsEditing)
            {
                if (e.KeyCode == Keys.Enter)
                {
                    rightClickedNode.EndEdit(false);
                }
                else if (e.KeyCode == Keys.Escape)
                {
                    rightClickedNode.EndEdit(true);
                }
            }
        }

        // The callback for the configuration page context menu 'Add Child' option
        private void AddChildClick(object sender, EventArgs e)
        {
            try
            {
                CreateNewNodes(rightClickedNode, 1);
            }
            catch (Exception ex)
            {
                logger.Error("Exception caught on tree view add child ", ex);
                MessageBox.Show("There was a problem with the property manager: \n\"" +
                    ex.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
        }

        private void AddSiteClick(object sender, EventArgs e)
        {
            try
            {
                LinkNode node = CreateEmptyNode(rightClickedNode);
                node.Link.isSite = true;
                node.Link.Joint.Type = "fixed";
                node.Link.Name = "site_" + node.Link.Name;
                node.Name = node.Link.Name;
                node.Text = node.Link.Name;
                rightClickedNode.Nodes.Add(node);
                rightClickedNode.ExpandAll();
            }
            catch (Exception ex)
            {
                logger.Error("Exception caught on tree view add site ", ex);
                MessageBox.Show("There was a problem with the property manager: \n\"" +
                    ex.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
        }

        // The callback for the configuration page context menu 'Remove Child' option
        private void RemoveChildClick(object sender, EventArgs e)
        {
            try
            {
                LinkNode parent = (LinkNode)rightClickedNode.Parent;
                if (parent == null)
                {
                    // this is the root node, which we cannot remove
                    return;
                }
                parent.Nodes.Remove(rightClickedNode);
                SyncCountersWithTree(_rootNode);
            }
            catch (Exception ex)
            {
                logger.Error("Exception caught on tree view remove child ", ex);
                MessageBox.Show("There was a problem with the property manager: \n\"" +
                    ex.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
        }

        // The callback for the configuration page context menu 'Rename Child' option
        // This isn't really working right now, so the option was deactivated from the
        // context menu
        private void RenameChildClick(object sender, EventArgs e)
        {
            try
            {
                Tree.SelectedNode = rightClickedNode;
                Tree.LabelEdit = true;
                rightClickedNode.BeginEdit();
                PMPage.SetFocus(dotNetTree);
            }
            catch (Exception ex)
            {
                logger.Error("Exception caught on tree view rename child ", ex);
                MessageBox.Show("There was a problem with the property manager: \n\"" +
                    ex.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
        }

        private void TreeItemDrag(object sender, ItemDragEventArgs e)
        {
            try
            {
                Tree.DoDragDrop(e.Item, DragDropEffects.Move);
            }
            catch (Exception ex)
            {
                logger.Error("Exception caught on tree view Drag ", ex);
                MessageBox.Show("There was a problem with the property manager: \n\"" +
                    ex.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
        }

        private void TreeDragOver(object sender, DragEventArgs e)
        {
            try
            {
                // Retrieve the client coordinates of the mouse position.
                Point targetPoint = Tree.PointToClient(new Point(e.X, e.Y));

                // Select the node at the mouse position.
                Tree.SelectedNode = Tree.GetNodeAt(targetPoint);
                e.Effect = DragDropEffects.Move;
            }
            catch (Exception ex)
            {
                logger.Error("Exception caught on tree view Drag Over ", ex);
                MessageBox.Show("There was a problem with the property manager: \n\"" +
                    ex.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
        }

        private void TreeDragEnter(object sender, DragEventArgs e)
        {
            try
            {
                // Retrieve the client coordinates of the mouse position.
                Point targetPoint = Tree.PointToClient(new Point(e.X, e.Y));

                // Select the node at the mouse position.
                Tree.SelectedNode = Tree.GetNodeAt(targetPoint);
                e.Effect = DragDropEffects.Move;
            }
            catch (Exception ex)
            {
                logger.Error("Exception caught on tree view DragEnter ", ex);
                MessageBox.Show("There was a problem with the property manager: \n\"" +
                    ex.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
        }

        private static bool IsDescendantOf(TreeNode potentialChild, TreeNode potentialParent)
        {
            TreeNode current = potentialChild;
            while (current != null)
            {
                if (current == potentialParent)
                    return true;
                current = current.Parent;
            }
            return false;
        }

        private void DoDragDrop(DragEventArgs e)
        {
            // Retrieve the client coordinates of the drop location.
            Point point = Tree.PointToClient(new Point(e.X, e.Y));

            // Retrieve the node at the drop location.
            LinkNode targetNode = (LinkNode)Tree.GetNodeAt(point);

            LinkNode draggedNode = (LinkNode)e.Data.GetData(typeof(LinkNode));

            // Check if the move is valid, if not then we won't do anything
            if (draggedNode == null || draggedNode == targetNode || draggedNode.TreeView != Tree)
            {
                return;
            }

            // If the dragged node is a parent of the target, we don't do anything.
            if (IsDescendantOf(targetNode, draggedNode))
            {
                return;
            }

            // If the it was dropped into the box itself, but not onto an actual node
            targetNode = targetNode ?? (LinkNode)Tree.TopNode;

            draggedNode.Remove();
            targetNode.Nodes.Add(draggedNode);
            targetNode.ExpandAll();
        }

        private void TreeDragDrop(object sender, DragEventArgs e)
        {
            try
            {
                DoDragDrop(e);
            }
            catch (Exception ex)
            {
                logger.Error("Exception caught on tree view Drag Drop ", ex);
                MessageBox.Show("There was a problem with the property manager: \n\"" +
                    ex.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
        }

        #endregion TreeView handler methods

        private string[] CreateTemporaryIcons(string filename)
        {
            string[] iconLocations = new string[6];

            BitmapHandler handler = BitmapHandler.Instance;
            Assembly thisAssembly;
            thisAssembly = Assembly.GetAssembly(this.GetType());
            iconLocations[0] = handler.CreateFileFromResourceBitmap("CADRobotExporter.Icons." + filename + "_20px.png", thisAssembly);
            iconLocations[1] = handler.CreateFileFromResourceBitmap("CADRobotExporter.Icons." + filename + "_32px.png", thisAssembly);
            iconLocations[2] = handler.CreateFileFromResourceBitmap("CADRobotExporter.Icons." + filename + "_40px.png", thisAssembly);
            iconLocations[3] = handler.CreateFileFromResourceBitmap("CADRobotExporter.Icons." + filename + "_64px.png", thisAssembly);
            iconLocations[4] = handler.CreateFileFromResourceBitmap("CADRobotExporter.Icons." + filename + "_96px.png", thisAssembly);
            iconLocations[5] = handler.CreateFileFromResourceBitmap("CADRobotExporter.Icons." + filename + "_128px.png", thisAssembly);

            return iconLocations;
        }

        //A method that sets up the Property Manager Page
        private void SetupPropertyManagerPage(ref string caption, ref string tip,
            ref long options, ref int controlType, ref int alignment)
        {
            string[] icons;

            // Create tabs
            PMTabKinematics = PMPage.AddTab(
                TabKinematicsID, "Kinematics", "", 0);
            PMTabTendons = PMPage.AddTab(
                TabTendonsID, "Tendons", "", 0);

            caption = "Page Controls";
            options = (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Visible +
                (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Expanded;
            PMGroupTopButtons = (PropertyManagerPageGroup)PMTabKinematics.AddGroupBox(GroupTopButtonsID, caption, (int)options);

            // We need to add a dummy label for the bitmaps icons to appear
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            PMLabelParentLink = (PropertyManagerPageLabel)PMGroupTopButtons.AddControl2(
                9999, (short)controlType, "foo", (short)alignment, 0, "");

            caption = "OK";
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_BitmapButton;
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_LeftEdge;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMButtonOkay = (PropertyManagerPageBitmapButton)PMGroupTopButtons.AddControl2(
                ButtonOkayID, (short)controlType, caption, (short)alignment, (int)options, "Save and close");
            icons = CreateTemporaryIcons("ok");
            PMButtonOkay.SetBitmapsByName3(icons, new string[6]);
            ((PropertyManagerPageControl)PMButtonOkay).Top = 0;
            ((PropertyManagerPageControl)PMButtonOkay).Left = 0;

            caption = "Cancel";
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_BitmapButton;
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_LeftEdge;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMButtonCancel = (PropertyManagerPageBitmapButton)PMGroupTopButtons.AddControl2(
                ButtonCancelID, (short)controlType, caption, (short)alignment, (int)options, "Cancel");
            icons = CreateTemporaryIcons("cancel");
            PMButtonCancel.SetBitmapsByName3(icons, new string[6]);
            ((PropertyManagerPageControl)PMButtonCancel).Top = 0;
            ((PropertyManagerPageControl)PMButtonCancel).Left = 20;

            caption = "Show extra hints";
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_LeftEdge;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Show extra information under some captions";
            PMCheckboxShowExtraCaptions = (PropertyManagerPageCheckbox)PMGroupTopButtons.AddControl2(
                CheckboxShowExtraCaptionsID, (short)controlType, caption, (short)alignment, (int)options, tip);

            caption = "Auto advance CSYS and Axis selections";
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_LeftEdge;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Automatically advances cooridnate systems and axis selections";
            PMCheckboxAutoAdvance = (PropertyManagerPageCheckbox)PMGroupTopButtons.AddControl2(
                CheckboxAutoAdvanceID, (short)controlType, caption, (short)alignment, (int)options, tip);

            caption = "Auto joint naming from link name";
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_LeftEdge;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Simply replaces \"link\" with \"joint\" while entering the Link name. If the word \"link\" is not found, will prepend \"joint\" in front.";
            PMCheckboxAutoJointNaming = (PropertyManagerPageCheckbox)PMGroupTopButtons.AddControl2(
                CheckboxAutoJointNamingID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMCheckboxAutoJointNaming.Checked = false;

            // Begin adding the controls to the page
            // Create the group box
            caption = "Configure and Links and Joints";
            options = (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Visible +
                (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Expanded;
            PMGroup = (PropertyManagerPageGroup)PMTabKinematics.AddGroupBox(GroupID, caption, (int)options);

            // Create the parent link label (static)
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Parent Link:";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelParentLink = (PropertyManagerPageLabel)PMGroup.AddControl2(
                LabelParentLinkID, (short)controlType, caption, (short)alignment, (int)options, "");

            // Create the parent link name label, the one that is updated
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelParentLinkName = (PropertyManagerPageLabel)PMGroup.AddControl2(
                LabelParentLinkNameID, (short)controlType, caption, (short)alignment, (int)options, "");

            // Create the link name label (static)
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Link Name:";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelLinkName = (PropertyManagerPageLabel)PMGroup.AddControl2(
                LabelLinkNameID, (short)controlType, caption, (short)alignment, (int)options, "");

            // Create the link name text box
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Textbox;
            caption = "base_link";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            tip = "Enter the name of the link";
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMTextBoxLinkName = (PropertyManagerPageTextbox)PMGroup.AddControl2(
                TextBoxLinkNameID, (short)(controlType), caption, (short)alignment, (int)options, tip);

            // Create Link Type radio buttons
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Option;
            caption = "Link";
            tip = "This node is a Link (has geometry and joint)";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMOptionLinkTypeLink = (PropertyManagerPageOption)PMGroup.AddControl2(
                OptionLinkTypeLinkID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMOptionLinkTypeLink.Checked = true;
            PMOptionLinkTypeLink.Style = (int)swPropMgrPageOptionStyle_e.swPropMgrPageOptionStyle_FirstInGroup;
            ((PropertyManagerPageControl)PMOptionLinkTypeLink).Top = 100;
            ((PropertyManagerPageControl)PMOptionLinkTypeLink).Left = 19;
            ((PropertyManagerPageControl)PMOptionLinkTypeLink).Width = 1;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Option;
            caption = "Site";
            tip = "This node is a Site (frame-only, no geometry)";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMOptionLinkTypeSite = (PropertyManagerPageOption)PMGroup.AddControl2(
                OptionLinkTypeSiteID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMOptionLinkTypeSite.Checked = false;
            ((PropertyManagerPageControl)PMOptionLinkTypeSite).Top = 100;
            ((PropertyManagerPageControl)PMOptionLinkTypeSite).Left = 24;
            ((PropertyManagerPageControl)PMOptionLinkTypeLink).Width = 1;

            // Create the joint name text box label
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Joint Name:";
            tip = "Enter the name of the joint";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible;
            PMLabelJointName = (PropertyManagerPageLabel)PMGroup.AddControl2(
                LabelJointNameID, (short)controlType, caption, (short)alignment, (int)options, tip);

            //Create the joint name text box
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Textbox;
            caption = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            tip = "Enter the name of the joint";
            options = (int)swAddControlOptions_e.swControlOptions_Visible;
            PMTextBoxJointName = (PropertyManagerPageTextbox)PMGroup.AddControl2(
                TextBoxJointNameID, (short)(controlType), caption, (short)alignment, (int)options, tip);

            //Create the ref coordinate sys label
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Reference Coordinate System:";
            tip = "Select the reference coordinate system for the joint origin";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelCoordSys = (PropertyManagerPageLabel)PMGroup.AddControl2(
                LabelCoordSysID, (short)controlType, caption, (short)alignment, (int)options, tip);

            // Create coordinate systems selection
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Selectionbox;
            caption = "Coordinate system";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Select coordinate system";
            PMCoordinateSystemSelection = (PropertyManagerPageSelectionbox)PMGroup.AddControl2(
                CoordinateSystemSelectionID, (short)controlType, caption, (short)alignment, (int)options, tip);

            swSelectType_e[] csysFilters = new swSelectType_e[1];
            csysFilters[0] = swSelectType_e.swSelCOORDSYS;
            object csysfilterObj = null;
            csysfilterObj = csysFilters;

            PMCoordinateSystemSelection.AllowSelectInMultipleBoxes = true;
            PMCoordinateSystemSelection.SingleEntityOnly = true;
            PMCoordinateSystemSelection.AllowMultipleSelectOfSameEntity = false;
            PMCoordinateSystemSelection.Mark = 2;
            PMCoordinateSystemSelection.SetSelectionFilters(csysfilterObj);
            PMCoordinateSystemSelection.SetSelectionColor(true, (int)swUserPreferenceIntegerValue_e.swSystemColorsSelectedItem4);
            icons = CreateTemporaryIcons("coordsys");
            ((PropertyManagerPageControl)PMCoordinateSystemSelection).SetPictureLabelByName(icons[1], "");

            //Create the ref axis label
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Reference Axis:";
            tip = "Select the reference axis for the joint";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible;
            PMLabelAxes = (PropertyManagerPageLabel)PMGroup.AddControl2(
                LabelAxesID, (short)controlType, caption, (short)alignment, (int)options, tip);

            // Create RefAxis Selection
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Selectionbox;
            caption = "Reference Axis Name";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Select the reference axis for the joint";
            PMRefAxisSelection = (PropertyManagerPageSelectionbox)PMGroup.AddControl2(
                RefAxisSelectionID, (short)controlType, caption, (short)alignment, (int)options, tip);

            swSelectType_e[] refAxisFilters = new swSelectType_e[1];
            refAxisFilters[0] = swSelectType_e.swSelDATUMAXES;
            object refAxisFiltesObj = null;
            refAxisFiltesObj = refAxisFilters;

            PMRefAxisSelection.AllowSelectInMultipleBoxes = true;
            PMRefAxisSelection.SingleEntityOnly = true;
            PMRefAxisSelection.AllowMultipleSelectOfSameEntity = false;
            PMRefAxisSelection.Mark = 4; // NB: each selection box must have its own mark that is unique and a power of two
            PMRefAxisSelection.SetSelectionFilters(refAxisFiltesObj);
            PMRefAxisSelection.SetSelectionColor(true, (int)swUserPreferenceIntegerValue_e.swSystemColorsDynamicHighlight);
            icons = CreateTemporaryIcons("axis");
            ((PropertyManagerPageControl)PMRefAxisSelection).SetPictureLabelByName(icons[1], "");

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            caption = "Flip Axis";
            tip = "The 'Axis' label represents the arrowhead of the axis, tick this checkbox to flip the calculated axis";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMRefAxisFlipCheckbox = PMGroup.AddControl2(
                RefAxisFlipID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMRefAxisFlipCheckbox.Checked = false;

            // Create the Axis from CSYS radio buttons
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Option;
            caption = "Select Axis";
            tip = "Use a reference axis selection";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMOptionAxisRefAxis = (PropertyManagerPageOption)PMGroup.AddControl2(
                OptionAxisRefAxisID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMOptionAxisRefAxis.Checked = true;
            PMOptionAxisRefAxis.Style = (int)swPropMgrPageOptionStyle_e.swPropMgrPageOptionStyle_FirstInGroup;
            ((PropertyManagerPageControl)PMOptionAxisRefAxis).Top = 330;
            ((PropertyManagerPageControl)PMOptionAxisRefAxis).Left = 19;
            ((PropertyManagerPageControl)PMOptionAxisRefAxis).Width = 1;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Option;
            caption = "CSYS X";
            tip = "Use CSYS X Axis";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMOptionAxisX = (PropertyManagerPageOption)PMGroup.AddControl2(
                OptionAxisXID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMOptionAxisX.Checked = false;
            ((PropertyManagerPageControl)PMOptionAxisX).Top = 330;
            ((PropertyManagerPageControl)PMOptionAxisX).Left = 40;
            ((PropertyManagerPageControl)PMOptionAxisX).Width = 1;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Option;
            caption = "CSYS Y";
            tip = "Use CSYS Y Axis";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMOptionAxisY = (PropertyManagerPageOption)PMGroup.AddControl2(
                OptionAxisYID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMOptionAxisY.Checked = false;
            ((PropertyManagerPageControl)PMOptionAxisY).Top = 330;
            ((PropertyManagerPageControl)PMOptionAxisY).Left = 50;
            ((PropertyManagerPageControl)PMOptionAxisY).Width = 1;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Option;
            caption = "CSYS Z";
            tip = "Use CSYS Z Axis";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMOptionAxisZ = (PropertyManagerPageOption)PMGroup.AddControl2(
                OptionAxisZID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMOptionAxisZ.Checked = false;
            ((PropertyManagerPageControl)PMOptionAxisZ).Top = 330;
            ((PropertyManagerPageControl)PMOptionAxisZ).Left = 60;
            ((PropertyManagerPageControl)PMOptionAxisZ).Width = 1;

            //Create the joint type label
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Joint Type:";
            tip = "Select the joint type";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible;
            PMLabelJointType = (PropertyManagerPageLabel)PMGroup.AddControl2(
                LabelJointTypeID, (short)controlType, caption, (short)alignment, (int)options, tip);

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Option;
            caption = "Revolute";
            tip = "Revolute Joint";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMOptionRevoluteJoint = (PropertyManagerPageOption)PMGroup.AddControl2(
                OptionRevoluteJointID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMOptionRevoluteJoint.Checked = true;
            PMOptionRevoluteJoint.Style = (int)swPropMgrPageOptionStyle_e.swPropMgrPageOptionStyle_FirstInGroup;
            ((PropertyManagerPageControl)PMOptionRevoluteJoint).Top = 400;
            ((PropertyManagerPageControl)PMOptionRevoluteJoint).Left = 19;
            ((PropertyManagerPageControl)PMOptionRevoluteJoint).Width = 1;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Option;
            caption = "Prismatic";
            tip = "Prismatic Joint";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMOptionPrismaticJoint = (PropertyManagerPageOption)PMGroup.AddControl2(
                OptionPrismaticJointID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMOptionPrismaticJoint.Checked = false;
            ((PropertyManagerPageControl)PMOptionPrismaticJoint).Top = 400;
            ((PropertyManagerPageControl)PMOptionPrismaticJoint).Left = 40;
            ((PropertyManagerPageControl)PMOptionPrismaticJoint).Width = 1;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Option;
            caption = "Fixed";
            tip = "Fixed Joint";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMOptionFixedJoint = (PropertyManagerPageOption)PMGroup.AddControl2(
                OptionFixedJointID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMOptionFixedJoint.Checked = false;
            ((PropertyManagerPageControl)PMOptionFixedJoint).Top = 400;
            ((PropertyManagerPageControl)PMOptionFixedJoint).Left = 80;
            ((PropertyManagerPageControl)PMOptionFixedJoint).Width = 1;

            swSelectType_e[] filters = new swSelectType_e[1];
            filters = new swSelectType_e[1];
            object filterObj = null;

            //Create the selection box label
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = LabelLinkInertialCaption;
            tip = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelLinkInertialComponents = (PropertyManagerPageLabel)PMGroup.AddControl2(
                LabelLinkInertialComponentsID, (short)controlType, caption, (short)alignment, (int)options, tip);

            //Create selection box
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Selectionbox;
            caption = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Use for accurate inertial properties, recommned selecting whole subassemblies from Feature Tree";
            PMLinkInertialComponentsSelection = (PropertyManagerPageSelectionbox)PMGroup.AddControl2(
                LinkInertialComponentsSelectionID, (short)controlType, caption, (short)alignment, (int)options, tip);

            filters[0] = swSelectType_e.swSelCOMPONENTS;
            filterObj = null;
            filterObj = filters;

            PMLinkInertialComponentsSelection.AllowSelectInMultipleBoxes = true;
            PMLinkInertialComponentsSelection.SingleEntityOnly = false;
            PMLinkInertialComponentsSelection.AllowMultipleSelectOfSameEntity = false;
            PMLinkInertialComponentsSelection.Height = 32;
            PMLinkInertialComponentsSelection.Mark = 16;
            PMLinkInertialComponentsSelection.EnableSelectIdenticalComponents = true;
            PMLinkInertialComponentsSelection.SetSelectionFilters(filterObj);
            PMLinkInertialComponentsSelection.Style = /*(int)swPropMgrPageSelectionBoxStyle_e.swPropMgrPageSelectionBoxStyle_MultipleItemSelect +*/
                (int)swPropMgrPageSelectionBoxStyle_e.swPropMgrPageSelectionBoxStyle_HScroll;
            PMLinkInertialComponentsSelection.SetSelectionColor(true, (int)swUserPreferenceIntegerValue_e.swSystemColorsSelectedItem2);
            icons = CreateTemporaryIcons("inertial");
            ((PropertyManagerPageControl)PMLinkInertialComponentsSelection).SetPictureLabelByName(icons[1], "");

            //Create the selection box label
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = LabelLinkCollisionCaption;
            tip = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelLinkCollisionComponents = (PropertyManagerPageLabel)PMGroup.AddControl2(
                LabelLinkCollisionComponentsID, (short)controlType, caption, (short)alignment, (int)options, tip);

            //Create selection box
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Selectionbox;
            caption = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Use for simplified geometry of complex components\n" +
                "Will be generated from visuals, then inertials if not selected";
            PMLinkCollisionComponentsSelection = (PropertyManagerPageSelectionbox)PMGroup.AddControl2(
                LinkCollisionComponentsSelectionID, (short)controlType, caption, (short)alignment, (int)options, tip);

            filters[0] = swSelectType_e.swSelCOMPONENTS;
            filterObj = null;
            filterObj = filters;

            PMLinkCollisionComponentsSelection.AllowSelectInMultipleBoxes = true;
            PMLinkCollisionComponentsSelection.SingleEntityOnly = false;
            PMLinkCollisionComponentsSelection.AllowMultipleSelectOfSameEntity = false;
            PMLinkCollisionComponentsSelection.Height = 32;
            PMLinkCollisionComponentsSelection.Mark = 8;
            PMLinkCollisionComponentsSelection.EnableSelectIdenticalComponents = true;
            PMLinkCollisionComponentsSelection.SetSelectionFilters(filterObj);
            PMLinkCollisionComponentsSelection.Style = /*(int)swPropMgrPageSelectionBoxStyle_e.swPropMgrPageSelectionBoxStyle_MultipleItemSelect +*/
                (int)swPropMgrPageSelectionBoxStyle_e.swPropMgrPageSelectionBoxStyle_HScroll;
            PMLinkCollisionComponentsSelection.SetSelectionColor(true, (int)swUserPreferenceIntegerValue_e.swSystemColorsSelectedItem3);
            icons = CreateTemporaryIcons("collision");
            ((PropertyManagerPageControl)PMLinkCollisionComponentsSelection).SetPictureLabelByName(icons[1], "");

            //Create the selection box label
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = LabelLinkVisualCaption;
            tip = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelLinkVisualComponents = (PropertyManagerPageLabel)PMGroup.AddControl2(
                LabelLinkVisualComponentsID, (short)controlType, caption, (short)alignment, (int)options, tip);

            //Create selection box
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Selectionbox;
            caption = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Select components that are visually rendered, if none are selected, inertial components will be used.";
            PMLinkVisualComponentsSelection = (PropertyManagerPageSelectionbox)PMGroup.AddControl2(
                LinkVisualComponentsSelectionID, (short)controlType, caption, (short)alignment, (int)options, tip);

            filters[0] = swSelectType_e.swSelCOMPONENTS;
            filterObj = filters;

            PMLinkVisualComponentsSelection.AllowSelectInMultipleBoxes = true;
            PMLinkVisualComponentsSelection.SingleEntityOnly = false;
            PMLinkVisualComponentsSelection.AllowMultipleSelectOfSameEntity = false;
            PMLinkVisualComponentsSelection.Height = 50;
            PMLinkVisualComponentsSelection.Mark = 1;
            PMLinkVisualComponentsSelection.EnableSelectIdenticalComponents = true;
            PMLinkVisualComponentsSelection.SetSelectionFilters(filterObj);
            PMLinkVisualComponentsSelection.Style = /*(int)swPropMgrPageSelectionBoxStyle_e.swPropMgrPageSelectionBoxStyle_MultipleItemSelect +*/
                (int)swPropMgrPageSelectionBoxStyle_e.swPropMgrPageSelectionBoxStyle_HScroll;
            icons = CreateTemporaryIcons("visual");
            ((PropertyManagerPageControl)PMLinkVisualComponentsSelection).SetPictureLabelByName(icons[1], "");

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            caption = CheckboxLinkPurelyInertialCaption + CheckboxLinkPurelyInertialExtraCaption;
            tip = "By default, if either Inertial or Visual components are selected, the link will have visual meshes. This will disable visual meshes for this link.";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMCheckboxLinkPurelyInertial = PMGroup.AddControl2(
                CheckboxLinkPurelyInertiallID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMCheckboxLinkPurelyInertial.Checked = false;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            caption = CheckboxLinkPurelyVisualCaption + CheckboxLinkPurelyVisualExtraCaption;
            tip = "By default, if either Inertial or Visual components are selected, the link will have inertial properties. This will disable inertial properties for this link.";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible + (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMCheckboxLinkPurelyVisual = PMGroup.AddControl2(
                CheckboxLinkNoPurelyVisualID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMCheckboxLinkPurelyVisual.Checked = false;

            // Create the group box
            caption = "Tree tools";
            options = (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Visible;
            PMGroupTreeTools = (PropertyManagerPageGroup)PMTabKinematics.AddGroupBox(GroupTreeToolsID, caption, (int)options);

            //Create the number box label
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Number of links in serial chain";
            tip = "Enter the number of links to be created for the serial chain.";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelSerialChainCount = (PropertyManagerPageLabel)PMGroupTreeTools.AddControl2(
                LabelSerialChainCountID, (short)controlType, caption, (short)alignment, (int)options, tip);

            //Create the number box
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Numberbox;
            caption = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            tip = "Enter the number of links to be created for the serial chain.";
            options = (int)swAddControlOptions_e.swControlOptions_Enabled +
                (int)swAddControlOptions_e.swControlOptions_Visible;
            PMNumberboxSerialChainCount = PMGroupTreeTools.AddControl2(
                NumberboxSerialChainCountID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMNumberboxSerialChainCount.SetRange2(
                (int)swNumberboxUnitType_e.swNumberBox_UnitlessInteger, 1, int.MaxValue, true, 1, 1, 1);
            PMNumberboxSerialChainCount.Value = 1;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Button;
            caption = "Create Serial Chain";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Create a serial chain of number of specified links at the currently selected link";
            PMButtonCreateSerialChain = PMGroupTreeTools.AddControl2(
                ButtonCreateSerialChainID, (short)controlType, caption, (short)alignment, (int)options, tip);

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Button;
            caption = "Insert Parent Link";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Inserts a parent link between the currently selected link and its parent";
            PMButtonInsertParentNode = PMGroupTreeTools.AddControl2(
                ButtonInsertParentNodeID, (short)controlType, caption, (short)alignment, (int)options, tip);

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Button;
            caption = "Insert Child Link";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Inserts a single child link to the currently selected link";
            PMButtonInsertChildNode = PMGroupTreeTools.AddControl2(
                ButtonInsertChildNodeID, (short)controlType, caption, (short)alignment, (int)options, tip);

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Button;
            caption = "Import Tree...";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Import tree structure from a text file";
            PMButtonImportTree = PMGroupTreeTools.AddControl2(
                ButtonImportTreeID, (short)controlType, caption, (short)alignment, (int)options, tip);

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Button;
            caption = "Export Tree...";
            alignment = 0;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            tip = "Export tree structure to a text file";
            PMButtonExportTree = PMGroupTreeTools.AddControl2(
                ButtonExportTreeID, (short)controlType, caption, (short)alignment, (int)options, tip);

            // Create the group box
            caption = "Export";
            options = (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Visible + (int)swAddGroupBoxOptions_e.swGroupBoxOptions_Expanded;
            PMGroupTree = (PropertyManagerPageGroup)PMTabKinematics.AddGroupBox(GroupTreeID, caption, (int)options);

            // Load Configuration button
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Button;
            caption = "Load Configuration...";
            tip = "Import values from a CSV file";
            alignment = 0;// (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_DoubleIndent;
            options = 0; // disabled
                //(int)swAddControlOptions_e.swControlOptions_Visible +
                //(int)swAddControlOptions_e.swControlOptions_Enabled;
            PMButtonLoad = PMGroupTree.AddControl2(
                LoadConfigurationID, (short)controlType, caption, (short)alignment, (int)options, tip);
            (PMButtonLoad as IPropertyManagerPageControl).Width = 200;

            // Loaded CSV Filename label
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = "Imported File:";
            tip = "";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = 0;
            PMLabelCSVFilename = PMGroupTree.AddControl2(
                LoadedCSVFilenameID, (short)controlType, caption, (short)alignment, (int)options, tip);

            // Create Check Boxes to select whether to recompute values
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            caption = "Compute Mass and Inertia:";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            tip = "External values have been loaded. Check this box to recompute the Mass and Inertia values";
            options = 0;
            PMComputeMassInertia = PMGroupTree.AddControl2(
                ComputeMassInertiaID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMComputeMassInertia.Checked = true;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            caption = "Compute Visual and Collision:";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            tip = "External values have been loaded. Check this box to recompute the visual and collision values";
            options = 0;
            PMComputeVisualCollision = PMGroupTree.AddControl2(
                ComputeVisualCollisionID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMComputeVisualCollision.Checked = true;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            caption = "Compute Joint Kinematics:";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            tip = "External values have been loaded. Check this box to recompute the joint kinematics";
            options = 0;
            PMComputeJointKinematics = PMGroupTree.AddControl2(
                ComputeJointKinematicsID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMComputeJointKinematics.Checked = true;

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Checkbox;
            caption = "Compute Joint Limits:";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            tip = "External values have been loaded. Check this box to recompute the joint limits";
            options = 0;
            PMComputeJointLimits = PMGroupTree.AddControl2(
                ComputeJointLimitsID, (short)controlType, caption, (short)alignment, (int)options, tip);
            PMComputeJointLimits.Checked = true;

            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMButtonExport = PMGroupTree.AddControl2(ButtonExportID,
                (short)swPropertyManagerPageControlType_e.swControlType_Button,
                "Preview and Export...", 0, (int)options, "Preview the robot configuration for export");
            (PMButtonExport as IPropertyManagerPageControl).Width = 200;

            // Tree label
            controlType = (int)swPropertyManagerPageControlType_e.swControlType_Label;
            caption = LabelKinematicTreeCaption;
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMLabelKinematicTree = (PropertyManagerPageLabel)PMGroupTree.AddControl2(
                LabelKinematicTreeID, (short)controlType, caption, (short)alignment, (int)options, "");

            controlType = (int)swPropertyManagerPageControlType_e.swControlType_WindowFromHandle;
            caption = "Link Tree";
            alignment = (int)swPropertyManagerPageControlLeftAlign_e.swControlAlign_Indent;
            options = (int)swAddControlOptions_e.swControlOptions_Visible +
                (int)swAddControlOptions_e.swControlOptions_Enabled;
            PMTree = PMGroupTree.AddControl2(dotNetTree,
                (short)swPropertyManagerPageControlType_e.swControlType_WindowFromHandle, caption, 0, (int)options, "");
            PMTree.Height = 163;
            Tree = new TreeView
            {
                Height = 163,
                Visible = true
            };

            Tree.NodeMouseClick += new TreeNodeMouseClickEventHandler(TreeNodeMouseClick);
            Tree.KeyDown += new KeyEventHandler(TreeKeyDown);
            Tree.DragDrop += new DragEventHandler(TreeDragDrop);
            Tree.DragOver += new DragEventHandler(TreeDragOver);
            Tree.DragEnter += new DragEventHandler(TreeDragEnter);
            Tree.ItemDrag += new ItemDragEventHandler(TreeItemDrag);
            Tree.AllowDrop = true;
            Tree.Font = System.Drawing.SystemFonts.MenuFont;
            PMTree.SetWindowHandlex64(Tree.Handle.ToInt64());

            ToolStripMenuItem addChild = new ToolStripMenuItem();
            ToolStripMenuItem addSite = new ToolStripMenuItem();
            ToolStripMenuItem removeChild = new ToolStripMenuItem();
            //ToolStripMenuItem renameChild = new ToolStripMenuItem();
            addChild.Text = "Add Child Link";
            addChild.Click += new EventHandler(AddChildClick);

            addSite.Text = "Add Site";
            addSite.Click += new EventHandler(AddSiteClick);

            removeChild.Text = "Remove";
            removeChild.Click += new EventHandler(RemoveChildClick);
            docMenu.Items.AddRange(new ToolStripMenuItem[] { addChild, addSite, removeChild });
            LinkNode node = CreateEmptyNode(null);
            node.ContextMenuStrip = docMenu;
            Tree.Nodes.Add(node);
            Tree.SelectedNode = Tree.Nodes[0];

            // Setup the Tendons tab
            SetupTendonTab(PMTabTendons);

            PMCoordinateSystemSelection.SetSelectionFocus();
            PMPage.SetFocus(dotNetTree);
        }

        void IPropertyManagerPage2Handler9.OnCheckboxCheck(int Id, bool Checked)
        {
            switch (Id)
            {
                case CheckboxLinkPurelyInertiallID:
                    if (Checked)
                    {
                        CommonSwOperations.DeselectByMark(model, PMLinkVisualComponentsSelection.Mark);
                    }
                    break;
                case CheckboxLinkNoPurelyVisualID:
                    if (Checked)
                    {
                        CommonSwOperations.DeselectByMark(model, PMLinkInertialComponentsSelection.Mark);
                    }
                    break;
                case CheckboxShowExtraCaptionsID:
                    PMLabelLinkVisualComponents.Caption = LabelLinkVisualCaption + (Checked ? LabelLinkVisualExtraCaption : "");
                    PMLabelLinkCollisionComponents.Caption = LabelLinkCollisionCaption + (Checked ? LabelLinkCollisionExtraCaption : "");
                    PMLabelLinkInertialComponents.Caption = LabelLinkInertialCaption + (Checked ? LabelLinkInertialExtraCaption : "");
                    PMCheckboxLinkPurelyInertial.Caption = CheckboxLinkPurelyInertialCaption + (Checked ? CheckboxLinkPurelyInertialExtraCaption : "");
                    PMCheckboxLinkPurelyVisual.Caption = CheckboxLinkPurelyVisualCaption + (Checked ? CheckboxLinkPurelyVisualExtraCaption : "");
                    PMLabelKinematicTree.Caption = LabelKinematicTreeCaption + (Checked ? LabelKinematicTreeExtraCaption : "");
                    break;
                case CheckboxAutoAddWaypointID:
                case CheckboxAutoLinkID:
                    UpdateTendonUIState();
                    break;
                default:
                    break;
            }
        }

        void IPropertyManagerPage2Handler9.OnComboboxEditChanged(int Id, string Text)
        {
        }

        void IPropertyManagerPage2Handler9.OnComboboxSelectionChanged(int Id, int Item)
        {
            if (Id == ComboElementTypeID)
            {
                OnElementTypeChanged(Item);
            }
            else if (Id == ComboParentLinkID)
            {
                OnParentLinkChanged(Item);
            }
        }

        void IPropertyManagerPage2Handler9.OnGroupCheck(int Id, bool Checked)
        {
        }

        void IPropertyManagerPage2Handler9.OnGroupExpand(int Id, bool Expanded)
        {
        }

        void IPropertyManagerPage2Handler9.OnListboxSelectionChanged(int Id, int Item)
        {
        }

        bool IPropertyManagerPage2Handler9.OnNextPage()
        {
            return true;
        }

        void IPropertyManagerPage2Handler9.OnOptionCheck(int Id)
        {
            if ((Id == OptionRevoluteJointID || Id == OptionPrismaticJointID || Id == OptionFixedJointID)
                && previouslySelectedNode != null && !previouslySelectedNode.IsBaseNode)
            {
                previouslySelectedNode.Link.Joint.Type = GetJointTypeFromOptions();
                UpdateKinematicControlStates();
            }
            else if ((Id == OptionLinkTypeLinkID || Id == OptionLinkTypeSiteID)
                && previouslySelectedNode != null && !previouslySelectedNode.IsBaseNode)
            {
                OnLinkTypeChanged(PMOptionLinkTypeSite.Checked ? 1 : 0);
            }
            else if ((Id == OptionAxisRefAxisID || Id == OptionAxisXID || Id == OptionAxisYID || Id == OptionAxisZID)
                && previouslySelectedNode != null && !previouslySelectedNode.IsBaseNode)
            {
                if (PMOptionAxisX.Checked)
                {
                    previouslySelectedNode.Link.Joint.AxisName = Joint.AxisFromCsysX;
                    previouslySelectedNode.Link.Joint.SWRefAxisFeature = null;
                }
                else if (PMOptionAxisY.Checked)
                {
                    previouslySelectedNode.Link.Joint.AxisName = Joint.AxisFromCsysY;
                    previouslySelectedNode.Link.Joint.SWRefAxisFeature = null;
                }
                else if (PMOptionAxisZ.Checked)
                {
                    previouslySelectedNode.Link.Joint.AxisName = Joint.AxisFromCsysZ;
                    previouslySelectedNode.Link.Joint.SWRefAxisFeature = null;
                }
                else
                {
                    previouslySelectedNode.Link.Joint.AxisName = "";
                }
                UpdateKinematicControlStates();
            }
        }

        void IPropertyManagerPage2Handler9.OnPopupMenuItem(int Id)
        {
        }

        void IPropertyManagerPage2Handler9.OnPopupMenuItemUpdate(int Id, ref int retval)
        {
        }

        bool IPropertyManagerPage2Handler9.OnPreview()
        {
            return true;
        }

        bool IPropertyManagerPage2Handler9.OnPreviousPage()
        {
            return true;
        }

        void IPropertyManagerPage2Handler9.OnRedo()
        {
        }

        void IPropertyManagerPage2Handler9.OnSelectionboxCalloutCreated(int Id)
        {
        }

        void IPropertyManagerPage2Handler9.OnSelectionboxCalloutDestroyed(int Id)
        {
        }

        void IPropertyManagerPage2Handler9.OnSliderPositionChanged(int Id, double Value)
        {
        }

        void IPropertyManagerPage2Handler9.OnSliderTrackingCompleted(int Id, double Value)
        {
        }

        bool IPropertyManagerPage2Handler9.OnTabClicked(int Id)
        {
            return true;
        }

        void IPropertyManagerPage2Handler9.OnUndo()
        {
        }

        void IPropertyManagerPage2Handler9.OnWhatsNew()
        {
        }

        void IPropertyManagerPage2Handler9.OnListboxRMBUp(int Id, int PosX, int PosY)
        {
        }

        void IPropertyManagerPage2Handler9.OnNumberBoxTrackingCompleted(int Id, double Value)
        {
        }

        void IPropertyManagerPage2Handler9.AfterClose()
        {
            //This function must contain code, even if it does nothing, to prevent the
            //.NET runtime environment from doing garbage collection at the wrong time.
            int IndentSize;
            IndentSize = System.Diagnostics.Debug.IndentSize;
            System.Diagnostics.Debug.WriteLine(IndentSize);
        }

        int IPropertyManagerPage2Handler9.OnActiveXControlCreated(int Id, bool Status)
        {
            return 0;
        }

    }
}

#endif
