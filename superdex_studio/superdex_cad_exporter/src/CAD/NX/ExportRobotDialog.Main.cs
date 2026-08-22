/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using NXOpen;
using NXOpen.Assemblies;
using NXOpen.BlockStyler;
using NXOpen.Features;
using NXOpen.UF;
using NXRobotExporter.CAD.NX;
using CADRobotExporter.CAD.NX;
using CADRobotExporter.Import;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.RobotExport;
using CADRobotExporter.UI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using Joint = CADRobotExporter.RobotDescription.Joint;
using SelectObject = NXOpen.BlockStyler.SelectObject;

/// <summary>
/// NX Dialog for exporting robot models to URDF format.
/// Replicates the SolidWorks URDF Exporter experience.
/// </summary>
public partial class ExportRobotDialog
{
    // NX Session and UI references
    private static Session theSession = null;
    private static UI theUI = null;
    private static UFSession ufSession = null;
    private string theDlxFileName;
    public static BlockDialog theDialog;

    // UI Blocks - Link Properties Group
    private NXOpen.BlockStyler.Group groupLinkProperties;
    private NXOpen.BlockStyler.Label label0;
    private NXOpen.BlockStyler.Label labelParentLink;
    private NXOpen.BlockStyler.StringBlock stringLinkName;
    private NXOpen.BlockStyler.StringBlock stringJointName;
    private NXOpen.BlockStyler.Enumeration enumLinkType;

    // UI Blocks - Options
    private NXOpen.BlockStyler.Group groupOptions;
    private NXOpen.BlockStyler.Toggle toggleAutoAdvanceCsys;
    private NXOpen.BlockStyler.Toggle toggleAutoAdvanceBody;// Block type: Toggle
    // TODO: actually implement
    private NXOpen.BlockStyler.Toggle toggleKeepSelectionsHighlighted;// Block type: Toggle
    private NXOpen.BlockStyler.Toggle toggleAutoJointNaming;
    private NXOpen.BlockStyler.Enumeration enumSelectBodyComponents;// Block type: Enumeration

    // UI Blocks - Joint Configuration Group
    private NXOpen.BlockStyler.Group groupJointProperties;
    private NXOpen.BlockStyler.SelectObject selectionCsys;// Block type: Selection
    private NXOpen.BlockStyler.SelectObject selectionAxis;// Block type: Selection
    private NXOpen.BlockStyler.Enumeration enumAxisFromCsys; // No (use selection), X, Y, Z
    private NXOpen.BlockStyler.Toggle toggleFlipAxis;// Block type: Toggle
    private NXOpen.BlockStyler.Enumeration enumJointType;

    // UI Blocks - Body Selection Group
    private NXOpen.BlockStyler.Group groupLinkBodies;
    private NXOpen.BlockStyler.SelectObject bodySelectInertial;
    private NXOpen.BlockStyler.SelectObject bodySelectCollision;
    private NXOpen.BlockStyler.SelectObject bodySelectVisual;
    private NXOpen.BlockStyler.Toggle togglePureInertial;// Block type: Toggle
    private NXOpen.BlockStyler.Toggle togglePureVisual;// Block type: Toggle

    // UI Blocks - Tree Tools
    private NXOpen.BlockStyler.Group groupTreeTools;// Block type: Group
    private NXOpen.BlockStyler.IntegerBlock integerNumLinksInSerialChain;// Block type: Integer
    private NXOpen.BlockStyler.Button buttonCreateSerialChain;// Block type: Button
    private NXOpen.BlockStyler.Button buttonInsertParentLink;// Block type: Button
    private NXOpen.BlockStyler.Button buttonInsertChildLink;// Block type: Button
    private NXOpen.BlockStyler.Button buttonImportTree;// Block type: Button
    private NXOpen.BlockStyler.Button buttonExportTree;// Block type: Button

    // UI Blocks - Export and Tree
    private NXOpen.BlockStyler.Button buttonExport;
    private NXOpen.BlockStyler.Button buttonSave;// Block type: Button
    private NXOpen.BlockStyler.Separator separator0;
    private NXOpen.BlockStyler.Label label01;
    private NXOpen.BlockStyler.Tree treeControlKinematicTree;

    // UI Blocks - Quick Settings
    private NXOpen.BlockStyler.Toggle toggleShowThroughCSYS;
    private NXOpen.BlockStyler.Toggle toggleShowThroughPoints;
    private NXOpen.BlockStyler.Toggle toggleShowThroughCurves;
    private NXOpen.BlockStyler.Toggle toggleDeselectGuardBodies;
    private NXOpen.BlockStyler.Toggle toggleDeselectGuardDatums;
    private NXOpen.BlockStyler.Enumeration enumSelectionBehavior;

    // Tree Manager and Robot Model
    private NXTreeManager treeManager;
    private NXBridge nxBridge;
    private NXOpen.Features.CustomFeature editedFeature;
    private Dictionary<string, string> componentToLinkCache;

    // UI flags and helpers
    private bool isUpdatingUI = false;
    private bool isBulkPopulating = false;
    private UIBlock lastSelectedKinematicBlock;
    private bool suppressOwnershipWarning = false;
    private bool suppressBodyOwnershipWarning = false;

    // Help system
    private const string HelpMapFileName = "RobotExporterHelp.map";
    private const string HelpContextMain = "robotexporter:MainDialog";
    private bool helpContextPushed = false;

    // Context menu item IDs
    // Right-click Menu IDs
    private const int MenuAddChild = 1;
    private const int MenuAddSibling = 2;
    private const int MenuInsertParent = 3;
    private const int MenuRemoveNode = 4;
    private const int MenuMoveUp = 5;
    private const int MenuMoveDown = 6;
    private const int MenuAddNewRoot = 7;
    private const int MenuConvertToRoot = 8;
    private const int MenuAddSite = 9;

    // Tendon tree context menu IDs
    private const int TendonMenuRemove = 101;
    private const int RoutingMenuRemove = 201;

    public enum LinkSelectionMode
    {
        Both,
        Bodies,
        Components,
    }

    private static readonly Serilog.ILogger logger = CADRobotExporter.Utilities.Logger.GetLogger();

    public ExportRobotDialog()
    {
        try
        {
            theSession = Session.GetSession();
            theUI = UI.GetUI();
            ufSession = UFSession.GetUFSession();
            theDlxFileName = "ExportRobotDialog.dlx";
            theDialog = theUI.CreateDialog(theDlxFileName);
            theDialog.AddUpdateHandler(new NXOpen.BlockStyler.BlockDialog.Update(UpdateCallback));
            theDialog.AddInitializeHandler(new NXOpen.BlockStyler.BlockDialog.Initialize(InitializeCallback));
            theDialog.AddDialogShownHandler(new NXOpen.BlockStyler.BlockDialog.DialogShown(DialogShownCallback));
            theDialog.AddCloseHandler(new NXOpen.BlockStyler.BlockDialog.Close(CloseCallback));
        }
        catch (Exception ex)
        {
            throw new Exception("Failed to initialize ExportRobotDialog: " + ex.Message, ex);
        }
    }

    public static void Main()
    {
        // This will work if the plugin lives in /application rather than /startup
        // and will allow us to right-click on the feature and hit Edit to bring up the dialog
        // however the plugin has to live in /startup to register custom features and actions,
        // so here we are.

        /*
        // Check if we're editing an existing custom feature
        theSession = NXOpen.Session.GetSession();
        NXOpen.Features.CustomFeatureClassManager mgr = theSession.CustomFeatureClassManager;
        var feature = mgr.GetEditedCustomFeature();
        if (feature == null)
        {
            return;
        }
        else
        {
            ExportRobotDialog theExportRobotDialog = null;
            try
            {
                theExportRobotDialog = new ExportRobotDialog();
                theExportRobotDialog.Launch();
            }
            catch (Exception ex)
            {
                theUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
            }
            finally
            {
                if (theExportRobotDialog != null)
                {
                    theExportRobotDialog.Dispose();
                    theExportRobotDialog = null;
                }
            }
        }
        //*/
    }

    // This entry point activates the application at NX startup
    // Will work when complete path of the dll is provided to Environment Variable
    // USER_STARTUP or USER_DEFAULT
    // Will also work if dll is at folder named "startup" under any folder listed in the
    // text file pointed to by the environment variable UGII_CUSTOM_DIRECTORY_FILE.
    public static int Startup()
    {
        try
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
        }
        catch
        {
            // These should only be called once per session, ignore exceptions.
        }

        int retValue = 0;
        try
        {
            theSession = NXOpen.Session.GetSession();
            theUI = UI.GetUI();

            logger.Information("Starting up NX Robot Exporter");

            // Register our custom feature class if needed
            // The CustomFeatureConfiguration.xml handles the registration,
            // but we can verify it's loaded here
            NXOpen.Features.CustomFeatureClassManager manager = theSession.CustomFeatureClassManager;
            try
            {
                NXOpen.Features.CustomFeatureClass cfClass = manager.GetClassFromName(NXConfigurationSerialization.CustomFeatureClassName);
                // Class is registered
            }
            catch (NXOpen.NXException ex)
            {
                theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                    "Could not find Custom Feature required for the Exporter. \n" +
                    "Exception thrown: \n" +
                    ex.Message);
            }

            logger.Information("Registering callbacks");

            theUI.MenuBarManager.RegisterApplication("ROBOT_EXPORTER",
            new NXOpen.MenuBar.MenuBarManager.InitializeMenuApplication(DummyCallback),
            new NXOpen.MenuBar.MenuBarManager.EnterMenuApplication(DummyCallback),
            new NXOpen.MenuBar.MenuBarManager.ExitMenuApplication(DummyCallback), true, true, true);
            theUI.MenuBarManager.AddMenuAction("ROBOT_EXPORTER__CreateNewRobot", new NXOpen.MenuBar.MenuBarManager.ActionCallback(CreateNewRobot));
            theUI.MenuBarManager.AddMenuAction("ROBOT_EXPORTER__EditRobot", new NXOpen.MenuBar.MenuBarManager.ActionCallback(EditRobot));
            theUI.MenuBarManager.AddMenuAction("ROBOT_EXPORTER__BackupRobot", new NXOpen.MenuBar.MenuBarManager.ActionCallback(BackupRobot));
            theUI.MenuBarManager.AddMenuAction("ROBOT_EXPORTER__ImportRobot", new NXOpen.MenuBar.MenuBarManager.ActionCallback(ImportRobot));
            theUI.MenuBarManager.AddMenuAction("ROBOT_EXPORTER__ImportURDF", new NXOpen.MenuBar.MenuBarManager.ActionCallback(ImportURDF));
            theUI.MenuBarManager.AddMenuAction("ROBOT_EXPORTER__DuplicateRobot", new NXOpen.MenuBar.MenuBarManager.ActionCallback(DuplicateRobot));
            theUI.MenuBarManager.AddMenuAction("ROBOT_EXPORTER__LaunchGuidSanitizer", new NXOpen.MenuBar.MenuBarManager.ActionCallback(LaunchGuidSanitizer));
            theUI.MenuBarManager.AddMenuAction("ROBOT_EXPORTER__LaunchGuidDebugger", new NXOpen.MenuBar.MenuBarManager.ActionCallback(GuidDebugger.LaunchGuidDebugger));
            theUI.MenuBarManager.AddMenuAction("ROBOT_EXPORTER__LaunchSuperDexMeshExporter", new NXOpen.MenuBar.MenuBarManager.ActionCallback(SuperDexMeshExporterDialog.LaunchSuperDexMeshExporter));
        }
        catch (NXOpen.NXException ex)
        {
            theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                "Startup() had an exception: \n" +
                ex.Message);
        }
        return retValue;
    }

    public static NXOpen.MenuBar.MenuBarManager.CallbackStatus CreateNewRobot(NXOpen.MenuBar.MenuButtonEvent buttonEvent)
    {
        if (Session.GetSession().Parts.Work == null)
        {
            return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Cancel;
        }

        var result = GuidSanitizer.CheckForDuplicates();

        if (result.HasDuplicates)
        {
            var answer = theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Question,
                $"Detected {result.DuplicateGuids} duplicate GUIDs in features in this assembly. " +
                $"It is highly recommended to use the GUID Sanitizer to remove duplicates first. " +
                $"Would you like to launch the tool now?");

            if (answer == 1)
            {
                LaunchGuidSanitizer(buttonEvent);
                return 0;
            }
        }

        ExportRobotDialog theExportRobotDialog = new ExportRobotDialog();
        theExportRobotDialog.editedFeature = null;
        theDialog.Launch();

        return 0;
    }

    public static NXOpen.MenuBar.MenuBarManager.CallbackStatus EditRobot(NXOpen.MenuBar.MenuButtonEvent buttonEvent)
    {
        if (Session.GetSession().Parts.Work == null)
        {
            return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Cancel;
        }

        var result = GuidSanitizer.CheckForDuplicates();

        if (result.HasDuplicates)
        {
            var answer = theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Question,
                $"Detected {result.DuplicateGuids} duplicate GUIDs in features in this assembly. " +
                $"It is highly recommended to use the GUID Sanitizer to remove duplicates first. " +
                $"Would you like to launch the tool now?");

            if (answer == 1)
            {
                LaunchGuidSanitizer(buttonEvent);
                return 0;
            }
        }

        ExportRobotDialog theExportRobotDialog = new ExportRobotDialog();

        int numSelected = theUI.SelectionManager.GetNumSelectedObjects();
        if (numSelected == 1)
        {
            TaggedObject selectedObject = theUI.SelectionManager.GetSelectedTaggedObject(0);
            if (selectedObject is CustomFeature)
            {
                theExportRobotDialog.editedFeature = selectedObject as CustomFeature;
                theDialog.Launch();
            }
        }
        else if (numSelected == 0)
        {
            theUI.NXMessageBox.Show("SuperDex CAD Exporter",
                NXMessageBox.DialogType.Error,
                "Nothing selected. Please select a Robot Configuration in the Part Navigator.");
        }
        else
        {
            theUI.NXMessageBox.Show("SuperDex CAD Exporter",
                NXMessageBox.DialogType.Error,
                "More than one feature selected. Please select a single Robot Configuration from the Part Navigator.");
        }

        return 0;
    }

    public static NXOpen.MenuBar.MenuBarManager.CallbackStatus LaunchGuidSanitizer(
        NXOpen.MenuBar.MenuButtonEvent buttonEvent)
    {
        if (Session.GetSession().Parts.Work == null)
        {
            return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Cancel;
        }

        GuidSanitizer theSanitizer = null;
        try
        {
            theSanitizer = new GuidSanitizer();
            theSanitizer.Launch();
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("GUID Sanitizer", NXMessageBox.DialogType.Error, ex.ToString());
        }
        finally
        {
            theSanitizer?.Dispose();
        }
        return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Continue;
    }

    public static int DummyCallback()
    {
        return 0;
    }

    public static NXOpen.MenuBar.MenuBarManager.CallbackStatus BackupRobot(NXOpen.MenuBar.MenuButtonEvent buttonEvent)
    {
        if (Session.GetSession().Parts.Work == null)
        {
            return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Cancel;
        }

        try
        {
            // Get the selected custom feature
            int numSelected = theUI.SelectionManager.GetNumSelectedObjects();
            if (numSelected != 1)
            {
                theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                    "Please select a single Robot Configuration from the Part Navigator.");
                return 0;
            }

            TaggedObject selectedObject = theUI.SelectionManager.GetSelectedTaggedObject(0);
            if (!(selectedObject is CustomFeature feature))
            {
                theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                    "Selected object is not a Robot Configuration.");
                return 0;
            }

            // Load raw string data from the feature
            if (!NXConfigurationSerialization.GetRawStringDataFromFeature(
                feature,
                out string robotName,
                out string urdfConfiguration,
                out string exporterConfiguration,
                out string tendonData))
            {
                theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                    "Failed to load configuration data from the selected feature.");
                return 0;
            }

            if (string.IsNullOrEmpty(robotName))
            {
                robotName = "robot";
            }

            // Prompt user to select save folder
            using (var folderDialog = new FolderBrowserDialog())
            {
                folderDialog.Description = "Select folder to save configuration files";

                if (folderDialog.ShowDialog() != DialogResult.OK)
                    return 0;

                string basePath = folderDialog.SelectedPath;
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

                string xmlPath = System.IO.Path.Combine(basePath, $"{robotName}.{timestamp}.urdfConfiguration.xml");
                System.IO.File.WriteAllText(xmlPath, urdfConfiguration);

                string jsonPath = System.IO.Path.Combine(basePath, $"{robotName}.{timestamp}.exporterConfiguration.json");
                System.IO.File.WriteAllText(jsonPath, exporterConfiguration);

                if (!string.IsNullOrEmpty(tendonData))
                {
                    string tendonPath = System.IO.Path.Combine(basePath, $"{robotName}.{timestamp}.tendons.xml");
                    System.IO.File.WriteAllText(tendonPath, tendonData);
                }

                theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Information,
                    $"Configuration backed up to:\n{basePath}");
            }
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                $"Error backing up configuration:\n{ex.Message}");
        }

        return 0;
    }

    public static NXOpen.MenuBar.MenuBarManager.CallbackStatus ImportRobot(NXOpen.MenuBar.MenuButtonEvent buttonEvent)
    {
        if (Session.GetSession().Parts.Work == null)
        {
            return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Cancel;
        }

        try
        {
            using (var openDialog = new OpenFileDialog())
            {
                openDialog.Title = "Select Robot Configuration XML";
                openDialog.Filter = "XML files (*.xml)|*.xml";

                if (openDialog.ShowDialog() != DialogResult.OK)
                    return 0;

                string xmlPath = openDialog.FileName;
                string directory = System.IO.Path.GetDirectoryName(xmlPath);
                string fileName = System.IO.Path.GetFileName(xmlPath);

                if (!fileName.EndsWith(".urdfConfiguration.xml"))
                {
                    theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                        "Invalid file. Expected *.urdfConfiguration.xml");
                    return 0;
                }

                // Extract robotName (and timestamp if present)
                // e.g., "myRobot.20250118_143022.urdfConfiguration.xml" -> "myRobot.20250118_143022"
                string baseName = fileName.Replace(".urdfConfiguration.xml", "");
                string jsonPath = System.IO.Path.Combine(directory, $"{baseName}.exporterConfiguration.json");

                string urdfConfiguration = System.IO.File.ReadAllText(xmlPath);
                string exporterConfiguration = "";
                string tendonData = "";

                if (System.IO.File.Exists(jsonPath))
                {
                    exporterConfiguration = System.IO.File.ReadAllText(jsonPath);
                }
                else
                {
                    theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Warning,
                        $"Companion file not found: {baseName}.exporterConfiguration.json\n\nProceeding without exporter configuration.");
                }

                string tendonPath = System.IO.Path.Combine(directory, $"{baseName}.tendons.xml");
                if (System.IO.File.Exists(tendonPath))
                {
                    tendonData = System.IO.File.ReadAllText(tendonPath);
                }

                // Extract just the robot name (strip timestamp if present)
                string robotName = baseName.Split('.')[0];

                // Create a new custom feature with the imported configuration
                NXConfigurationSerialization.CreateNewConfigurationFromRawData(
                    theSession.Parts.Work,
                    robotName,
                    urdfConfiguration,
                    exporterConfiguration,
                    tendonData);

                theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Information,
                    $"Configuration '{robotName}' imported successfully.");
            }
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                $"Error importing configuration:\n{ex.Message}");
        }

        return 0;
    }

    /// <summary>
    /// Menu action to import a URDF file into NX.
    /// Creates coordinate systems at joint origins and optionally a Robot Configuration feature.
    /// </summary>
    public static NXOpen.MenuBar.MenuBarManager.CallbackStatus ImportURDF(NXOpen.MenuBar.MenuButtonEvent buttonEvent)
    {
        if (Session.GetSession().Parts.Work == null)
        {
            return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Cancel;
        }

        URDFImporterDialog importer = null;
        try
        {
            importer = new URDFImporterDialog();
            importer.Launch();
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("URDF Import", NXMessageBox.DialogType.Error, ex.ToString());
        }
        finally
        {
            importer?.Dispose();
        }

        return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Continue;
    }

    public static NXOpen.MenuBar.MenuBarManager.CallbackStatus DuplicateRobot(NXOpen.MenuBar.MenuButtonEvent buttonEvent)
    {
        if (Session.GetSession().Parts.Work == null)
        {
            return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Cancel;
        }

        try
        {
            // Get the selected custom feature
            int numSelected = theUI.SelectionManager.GetNumSelectedObjects();
            if (numSelected != 1)
            {
                theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                    "Please select a single Robot Configuration from the Part Navigator.");
                return 0;
            }

            TaggedObject selectedObject = theUI.SelectionManager.GetSelectedTaggedObject(0);
            if (!(selectedObject is CustomFeature feature))
            {
                theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                    "Selected object is not a Robot Configuration.");
                return 0;
            }

            // Load raw string data from the feature
            if (!NXConfigurationSerialization.GetRawStringDataFromFeature(
                feature,
                out string robotName,
                out string urdfConfiguration,
                out string exporterConfiguration,
                out string tendonData))
            {
                theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                    "Failed to load configuration data from the selected feature.");
                return 0;
            }

            if (string.IsNullOrEmpty(robotName))
            {
                robotName = "robot";
            }

            // Create a new custom feature with the duplicated configuration
            string newName = robotName;
            NXConfigurationSerialization.CreateNewConfigurationFromRawData(
                theSession.Parts.Work,
                newName,
                urdfConfiguration,
                exporterConfiguration,
                tendonData);

            theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Information,
                $"Configuration duplicated as '{newName}'.");
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Error,
                $"Error duplicating configuration:\n{ex.Message}");
        }

        return 0;
    }

    public static int GetUnloadOption(string arg)
    {
        return System.Convert.ToInt32(Session.LibraryUnloadOption.AtTermination);
    }

    public static void UnloadLibrary(string arg)
    {
    }

    /// <summary>
    /// Launches the dialog in the appropriate mode (Create or Edit).
    /// When editing an existing custom feature, loads the saved configuration.
    /// </summary>
    public NXOpen.BlockStyler.BlockDialog.DialogResponse Launch()
    {
        NXOpen.BlockStyler.BlockDialog.DialogResponse dialogResponse = NXOpen.BlockStyler.BlockDialog.DialogResponse.Invalid;
        try
        {
            // Check if we're editing an existing custom feature
            NXOpen.Features.CustomFeatureClassManager mgr = theSession.CustomFeatureClassManager;
            editedFeature = mgr.GetEditedCustomFeature();

            // Determine the dialog mode
            NXOpen.BlockStyler.BlockDialog.DialogMode mode =
                editedFeature != null
                    ? NXOpen.BlockStyler.BlockDialog.DialogMode.Edit
                    : NXOpen.BlockStyler.BlockDialog.DialogMode.Create;

            dialogResponse = theDialog.LaunchInDialogMode(mode);
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
        }
        return dialogResponse;
    }

    public void Dispose()
    {
        if (theDialog != null)
        {
            theDialog.Dispose();
            theDialog = null;
        }
    }

    public void InitializeCallback()
    {
        try
        {
            Part workPart = theSession.Parts.Work;

            // Get references to all UI blocks
            groupLinkProperties = (NXOpen.BlockStyler.Group)theDialog.TopBlock.FindBlock("groupLinkProperties");
            label0 = (NXOpen.BlockStyler.Label)theDialog.TopBlock.FindBlock("label0");
            labelParentLink = (NXOpen.BlockStyler.Label)theDialog.TopBlock.FindBlock("labelParentLink");
            stringLinkName = (NXOpen.BlockStyler.StringBlock)theDialog.TopBlock.FindBlock("stringLinkName");
            stringJointName = (NXOpen.BlockStyler.StringBlock)theDialog.TopBlock.FindBlock("stringJointName");
            enumLinkType = (NXOpen.BlockStyler.Enumeration)theDialog.TopBlock.FindBlock("enumLinkType");
            groupOptions = (NXOpen.BlockStyler.Group)theDialog.TopBlock.FindBlock("groupOptions");
            toggleAutoAdvanceCsys = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleAutoAdvanceCsys");
            toggleAutoAdvanceBody = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleAutoAdvanceBody");
            toggleAutoJointNaming = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleAutoJointNaming");
            toggleKeepSelectionsHighlighted = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleKeepSelectionsHighlighted");
            enumSelectBodyComponents = (NXOpen.BlockStyler.Enumeration)theDialog.TopBlock.FindBlock("enumSelectBodyComponents");
            groupJointProperties = (NXOpen.BlockStyler.Group)theDialog.TopBlock.FindBlock("groupJointProperties");
            selectionCsys = (NXOpen.BlockStyler.SelectObject)theDialog.TopBlock.FindBlock("selectionCsys");
            selectionAxis = (NXOpen.BlockStyler.SelectObject)theDialog.TopBlock.FindBlock("selectionAxis");
            enumAxisFromCsys = (NXOpen.BlockStyler.Enumeration)theDialog.TopBlock.FindBlock("enumAxisFromCsys");
            toggleFlipAxis = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleFlipAxis");
            enumJointType = (NXOpen.BlockStyler.Enumeration)theDialog.TopBlock.FindBlock("enumJointType");
            groupLinkBodies = (NXOpen.BlockStyler.Group)theDialog.TopBlock.FindBlock("groupLinkBodies");
            bodySelectInertial = (NXOpen.BlockStyler.SelectObject)theDialog.TopBlock.FindBlock("bodySelectInertial");
            bodySelectCollision = (NXOpen.BlockStyler.SelectObject)theDialog.TopBlock.FindBlock("bodySelectCollision");
            bodySelectVisual = (NXOpen.BlockStyler.SelectObject)theDialog.TopBlock.FindBlock("bodySelectVisual");
            togglePureInertial = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("togglePureInertial");
            togglePureVisual = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("togglePureVisual");
            integerNumLinksInSerialChain = (NXOpen.BlockStyler.IntegerBlock)theDialog.TopBlock.FindBlock("integerNumLinksInSerialChain");
            buttonCreateSerialChain = (NXOpen.BlockStyler.Button)theDialog.TopBlock.FindBlock("buttonCreateSerialChain");
            buttonInsertParentLink = (NXOpen.BlockStyler.Button)theDialog.TopBlock.FindBlock("buttonInsertParentLink");
            buttonInsertChildLink = (NXOpen.BlockStyler.Button)theDialog.TopBlock.FindBlock("buttonInsertChildLink");
            buttonImportTree = (NXOpen.BlockStyler.Button)theDialog.TopBlock.FindBlock("buttonImportTree");
            buttonExportTree = (NXOpen.BlockStyler.Button)theDialog.TopBlock.FindBlock("buttonExportTree");
            buttonExport = (NXOpen.BlockStyler.Button)theDialog.TopBlock.FindBlock("buttonExport");
            buttonSave = (NXOpen.BlockStyler.Button)theDialog.TopBlock.FindBlock("buttonSave");
            separator0 = (NXOpen.BlockStyler.Separator)theDialog.TopBlock.FindBlock("separator0");
            label01 = (NXOpen.BlockStyler.Label)theDialog.TopBlock.FindBlock("label01");
            treeControlKinematicTree = (NXOpen.BlockStyler.Tree)theDialog.TopBlock.FindBlock("treeControlKinematicTree");
            groupTreeTools = (NXOpen.BlockStyler.Group)theDialog.TopBlock.FindBlock("groupTreeTools");

            // Tendon UI blocks
            treeControlTendons = (NXOpen.BlockStyler.Tree)theDialog.TopBlock.FindBlock("treeControlTendons");
            treeControlTendonRouting = (NXOpen.BlockStyler.Tree)theDialog.TopBlock.FindBlock("treeControlTendonRouting");
            stringTendonName = (NXOpen.BlockStyler.StringBlock)theDialog.TopBlock.FindBlock("stringTendonName");
            buttonAddTendon = (NXOpen.BlockStyler.Button)theDialog.TopBlock.FindBlock("buttonAddTendon");
            buttonAddRoutingElement = (NXOpen.BlockStyler.Button)theDialog.TopBlock.FindBlock("buttonAddRoutingElement");
            togglePointAutoAddRouting = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("togglePointAutoAddRouting");
            toggleRoutingAutoSelectLink = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleRoutingAutoSelectLink");
            pointSelectRouting = (NXOpen.BlockStyler.SelectObject)theDialog.TopBlock.FindBlock("pointSelectRouting");
            doubleTendonCoefficient = (NXOpen.BlockStyler.DoubleBlock)theDialog.TopBlock.FindBlock("doubleTendonCoefficient");
            enumTendonRoutingType = (NXOpen.BlockStyler.Enumeration)theDialog.TopBlock.FindBlock("enumTendonRoutingType");
            enumTendonParentLink = (NXOpen.BlockStyler.Enumeration)theDialog.TopBlock.FindBlock("enumTendonParentLink");
            groupTendons = (NXOpen.BlockStyler.Group)theDialog.TopBlock.FindBlock("groupTendons");
            groupRoutingElements = (NXOpen.BlockStyler.Group)theDialog.TopBlock.FindBlock("groupRoutingElements");

            // Quick settings UI blocks
            toggleShowThroughCSYS = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleShowThroughCSYS");
            toggleShowThroughCurves = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleShowThroughCurves");
            toggleShowThroughPoints = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleShowThroughPoints");
            toggleDeselectGuardBodies = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleDeselectGuardBodies");
            toggleDeselectGuardDatums = (NXOpen.BlockStyler.Toggle)theDialog.TopBlock.FindBlock("toggleDeselectGuardDatums");
            enumSelectionBehavior = (NXOpen.BlockStyler.Enumeration)theDialog.TopBlock.FindBlock("enumSelectionBehavior");

            // Register body selection callbacks
            theDialog.AddFocusNotifyHandler(new BlockDialog.FocusNotify(OnFocusNotify));

            // Register tree callbacks
            treeControlKinematicTree.SetOnSelectHandler(new NXOpen.BlockStyler.Tree.OnSelectCallback(OnSelectCallback));
            treeControlKinematicTree.SetOnMenuHandler(new NXOpen.BlockStyler.Tree.OnMenuCallback(OnMenuCallback));
            treeControlKinematicTree.SetOnMenuSelectionHandler(new NXOpen.BlockStyler.Tree.OnMenuSelectionCallback(OnMenuSelectionCallback));
            treeControlKinematicTree.SetIsDragAllowedHandler(new NXOpen.BlockStyler.Tree.IsDragAllowedCallback(IsDragAllowedCallback));
            treeControlKinematicTree.SetIsDropAllowedHandler(new NXOpen.BlockStyler.Tree.IsDropAllowedCallback(IsDropAllowedCallback));
            treeControlKinematicTree.SetOnDropHandler(new NXOpen.BlockStyler.Tree.OnDropCallback(OnDropCallback));
            treeControlKinematicTree.SetOnBeginLabelEditHandler(new NXOpen.BlockStyler.Tree.OnBeginLabelEditCallback(OnBeginLabelEditCallback));
            treeControlKinematicTree.SetOnEndLabelEditHandler(new NXOpen.BlockStyler.Tree.OnEndLabelEditCallback(OnEndLabelEditCallback));

            // Register tendon tree callbacks
            treeControlTendons.SetOnSelectHandler(new NXOpen.BlockStyler.Tree.OnSelectCallback(OnTendonSelectCallback));
            treeControlTendons.SetOnMenuHandler(new NXOpen.BlockStyler.Tree.OnMenuCallback(OnTendonMenuCallback));
            treeControlTendons.SetOnMenuSelectionHandler(new NXOpen.BlockStyler.Tree.OnMenuSelectionCallback(OnTendonMenuSelectionCallback));
            treeControlTendons.SetOnBeginLabelEditHandler(new NXOpen.BlockStyler.Tree.OnBeginLabelEditCallback(OnTendonBeginLabelEditCallback));
            treeControlTendons.SetOnEndLabelEditHandler(new NXOpen.BlockStyler.Tree.OnEndLabelEditCallback(OnTendonEndLabelEditCallback));
            treeControlTendons.SetIsDragAllowedHandler(new NXOpen.BlockStyler.Tree.IsDragAllowedCallback(TendonIsDragAllowedCallback));
            treeControlTendons.SetIsDropAllowedHandler(new NXOpen.BlockStyler.Tree.IsDropAllowedCallback(TendonIsDropAllowedCallback));
            treeControlTendons.SetOnDropHandler(new NXOpen.BlockStyler.Tree.OnDropCallback(TendonOnDropCallback));

            // Register routing element tree callbacks
            treeControlTendonRouting.SetOnSelectHandler(new NXOpen.BlockStyler.Tree.OnSelectCallback(OnRoutingSelectCallback));
            treeControlTendonRouting.SetOnMenuHandler(new NXOpen.BlockStyler.Tree.OnMenuCallback(OnRoutingMenuCallback));
            treeControlTendonRouting.SetOnMenuSelectionHandler(new NXOpen.BlockStyler.Tree.OnMenuSelectionCallback(OnRoutingMenuSelectionCallback));
            treeControlTendonRouting.SetIsDragAllowedHandler(new NXOpen.BlockStyler.Tree.IsDragAllowedCallback(RoutingIsDragAllowedCallback));
            treeControlTendonRouting.SetIsDropAllowedHandler(new NXOpen.BlockStyler.Tree.IsDropAllowedCallback(RoutingIsDropAllowedCallback));
            treeControlTendonRouting.SetOnDropHandler(new NXOpen.BlockStyler.Tree.OnDropCallback(RoutingOnDropCallback));

            theDialog.TopBlock.Label = "Robot Configuration Exporter " + CADRobotExporter.Versioning.VersionString.Get();

            // Initialize the tree manager
            treeManager = new NXTreeManager(treeControlKinematicTree);
            treeManager.OnNodeCreated += OnNodeCreated;

            // Initialize the NX Bridge for saving/loading configurations
            nxBridge = new NXBridge(theSession.Parts.Work);

            // Set up selectionCsys to select only actual CSYS
            Selection.MaskTriple selectionMask;
            selectionMask.Type = UFConstants.UF_coordinate_system_type;
            selectionMask.Subtype = UFConstants.UF_csys_normal_subtype;
            selectionMask.SolidBodySubtype = 0;
            selectionCsys.SetSelectionFilter(Selection.SelectionAction.ClearAndEnableSpecific, new Selection.MaskTriple[] { selectionMask });

            // selectionAxis to only use datum axis (perhaps we can do UF_axis_type, too, but unclear what that is)
            selectionMask.Type = UFConstants.UF_datum_axis_type;
            selectionMask.Subtype = 0;
            selectionMask.SolidBodySubtype = 0;
            selectionAxis.SetSelectionFilter(Selection.SelectionAction.ClearAndEnableSpecific, new Selection.MaskTriple[] { selectionMask });

            // pointSelectRouting to select only points
            selectionMask.Type = UFConstants.UF_point_type;
            selectionMask.Subtype = 0;
            selectionMask.SolidBodySubtype = 0;
            pointSelectRouting.SetSelectionFilter(Selection.SelectionAction.ClearAndEnableSpecific, new Selection.MaskTriple[] { selectionMask });

            stringLinkName.SetKeystrokeCallbackHandler(new StringBlock.KeystrokeCallbackHandler(OnStringLinkNameKeystroke));
            stringJointName.SetKeystrokeCallbackHandler(new StringBlock.KeystrokeCallbackHandler(OnStringJointNameKeystroke));
            stringTendonName.SetKeystrokeCallbackHandler(new StringBlock.KeystrokeCallbackHandler(OnStringTendonNameKeystroke));

            if (editedFeature == null)
            {
                toggleAutoAdvanceCsys.Value = true;
            }

            toggleShowThroughCSYS.Value = workPart.Preferences.ScreenVisualization.CsysShowThrough;
            toggleShowThroughPoints.Value = workPart.Preferences.ScreenVisualization.PointShowThrough;
            toggleShowThroughCurves.Value = workPart.Preferences.ScreenVisualization.CurveShowThrough;
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
        }
    }

    private void OnFocusNotify(UIBlock block, bool isFocus)
    {
        if (!isFocus)
        {
            return;
        }

        if (block == selectionCsys || block == selectionAxis ||
            block == bodySelectCollision || block == bodySelectInertial || block == bodySelectVisual)
        {
            lastSelectedKinematicBlock = block;
        }

        if (block == pointSelectRouting)
        {
            lastSelectedTendonBlock = block;
        }
    }

    public void DialogShownCallback()
    {
        try
        {
            InitializeHelpSystem();

            // Update selection mode to reflect saved enum
            OnEnumSelectBodyComponentsChanged();

            // This is required due to some bizzare NXOpen bug
            bodySelectInertial.AllowConvergentObject = true;
            bodySelectCollision.AllowConvergentObject = true;
            bodySelectVisual.AllowConvergentObject = true;

            treeManager.InitializeColumns();
            InitializeTendonTreeColumns();

            // Check if we're editing an existing configuration
            if (editedFeature != null)
            {
                LoadExistingConfiguration();
            }
            else
            {
                // Create new configuration - create the root node (base_link)
                var rootNode = treeManager.CreateRootNode("base_link");

                treeControlKinematicTree.SelectNode(rootNode.TreeNode, true, true);
                treeManager.UpdateSelectedNodes();
            }

            UpdateKinematicTabUI();
            UpdateTendonTabUI();

            // Override user defaults
            enumSelectBodyComponents.ValueAsString = "Only Components";
            enumSelectionBehavior.ValueAsString = "Ignore Read-Only";

            OnEnumSelectBodyComponentsChanged();
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show(
                "SuperDex CAD Exporter",
                NXMessageBox.DialogType.Error,
                "DialogShownCallback(), Exception thrown:" + ex.ToString());
        }
    }

    public int CloseCallback()
    {
        CleanupHelpSystem();

        if (AssemblyExportForm.Instance != null)
        {
            AssemblyExportForm.Instance.WindowState = FormWindowState.Normal;
            AssemblyExportForm.Instance.Focus();
            return 1;
        }

        var result = MessageBox.Show(
            "Would you like to save the configuration?",
            "SuperDex CAD Exporter",
            MessageBoxButtons.YesNoCancel);

        switch (result)
        {
            case DialogResult.Yes:
                SaveConfiguration();
                return 0;
            case DialogResult.No:
                return 0;
            default:
                return 1;
            case DialogResult.Cancel:
                return 1;
        }
    }

    /// <summary>
    /// Saves the current robot configuration to a custom feature in the work part.
    /// </summary>
    private void SaveConfiguration()
    {
        try
        {
            // Build the Link tree from the tree manager
            Link baseLink = BuildLinkTreeFromNodes();
            if (baseLink == null)
            {
                theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Warning,
                    "No robot configuration to save.");
                return;
            }

            // Create exporter configuration
            ExporterConfiguration config = nxBridge.GetExporterConfiguration();

            // Save using NXBridge
            nxBridge.SaveConfigurationFromLink(config, baseLink, false, null, tendons);

            theUI.NXMessageBox.Show("SuperDex CAD Exporter", NXMessageBox.DialogType.Information,
                "Saved robot configuration to feature.");
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("Save Configuration", NXMessageBox.DialogType.Error,
                $"Failed to save configuration: {ex.Message}");
        }
    }

    /// <summary>
    /// Builds a Link tree from the current tree manager nodes.
    /// </summary>
    private Link BuildLinkTreeFromNodes()
    {
        return treeManager.BuildLinkTree();
    }

    /// <summary>
    /// Loads an existing configuration from the edited custom feature.
    /// Called when the dialog is opened in Edit mode.
    /// </summary>
    private void LoadExistingConfiguration()
    {
        if (editedFeature == null)
            return;

        try
        {
            // Load the configuration from the custom feature
            ExporterConfiguration loadedConfig;
            List<Tendon> loadedTendons;
            Link baseLink = NXConfigurationSerialization.LoadConfiguration(editedFeature, out loadedConfig, out loadedTendons);
            NXLinkNode rootNode;

            if (baseLink == null)
            {
                theUI.NXMessageBox.Show("Load Configuration", NXMessageBox.DialogType.Warning,
                    "Could not load existing configuration. Starting with empty tree.");

                // Create a new root node
                rootNode = treeManager.CreateRootNode("base_link");
                treeControlKinematicTree.SelectNode(rootNode.TreeNode, true, true);
                treeManager.UpdateSelectedNodes();
                return;
            }

            // Update the NXBridge with loaded configuration
            nxBridge.SetExporterConfiguration(loadedConfig);
            nxBridge.SetConfigurationFeature(editedFeature);

            // Populate the tree from the loaded link hierarchy
            isBulkPopulating = true;
            rootNode = treeManager.PopulateFromLink(baseLink);
            isBulkPopulating = false;

            // Validate all selections and report missing references
            ValidateAndReportMissingReferences(rootNode);

            // Load tendons
            if (loadedTendons != null && loadedTendons.Count > 0)
            {
                PopulateTendonsFromList(loadedTendons);
            }

            // Select the root node
            if (rootNode != null)
            {
                treeControlKinematicTree.SelectNode(rootNode.TreeNode, true, true);
                treeManager.UpdateSelectedNodes();
            }
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("Load Configuration", NXMessageBox.DialogType.Error,
                $"Error loading configuration: {ex.Message}");

            // Fall back to creating a new root node
            var rootNode = treeManager.CreateRootNode("base_link");
            treeControlKinematicTree.SelectNode(rootNode.TreeNode, true, true);
            treeManager.UpdateSelectedNodes();
        }
    }

    /// <summary>
    /// Validates all selections (CSYS, axis, bodies) for all nodes in the tree.
    /// Reports what's missing and offers to clear invalid references.
    /// </summary>
    private bool ValidateAndReportMissingReferences(NXLinkNode rootNode)
    {
        if (rootNode == null)
            return false;

        Part workPart = theSession.Parts.Work;
        var missingReferences = new List<string>();
        var nodesWithMissing = new List<NXLinkNode>();

        // Collect all nodes
        var allNodes = new List<NXLinkNode>();
        CollectAllNodes(rootNode, allNodes);

        // Validate each node
        foreach (var node in allNodes)
        {
            var nodeMissing = new List<string>();

            // Validate CSYS
            if (!string.IsNullOrEmpty(node.CoordinateSystemHandle))
            {
                var csys = NXPersistentId.FindCoordinateSystemByKey(workPart, node.CoordinateSystemHandle);
                if (csys == null)
                {
                    nodeMissing.Add("Coordinate System");
                }
            }

            // Validate Axis (only for non-root nodes)
            // Skip validation for CSYS axis magic keywords - they don't resolve to DatumAxis objects
            if (!node.IsRootNode && !string.IsNullOrEmpty(node.JointAxisHandle) && !Joint.IsAxisFromCsys(node.JointAxisHandle))
            {
                var axis = NXPersistentId.FindAxisByKey(workPart, node.JointAxisHandle);
                if (axis == null)
                {
                    nodeMissing.Add("Joint Axis");
                }
            }

            // Validate Visual Bodies/Components
            if (node.VisualBodiesHandles != null && node.VisualBodiesHandles.Length > 0)
            {
                int missingCount = CountMissingHandles(workPart, node.VisualBodiesHandles);
                if (missingCount > 0)
                {
                    nodeMissing.Add($"Visual Bodies/Components ({missingCount} missing)");
                }
            }

            // Validate Collision Bodies/Components
            if (node.CollisionBodiesHandles != null && node.CollisionBodiesHandles.Length > 0)
            {
                int missingCount = CountMissingHandles(workPart, node.CollisionBodiesHandles);
                if (missingCount > 0)
                {
                    nodeMissing.Add($"Collision Bodies/Components ({missingCount} missing)");
                }
            }

            // Validate Inertial Bodies/Components
            if (node.InertialBodiesHandles != null && node.InertialBodiesHandles.Length > 0)
            {
                int missingCount = CountMissingHandles(workPart, node.InertialBodiesHandles);
                if (missingCount > 0)
                {
                    nodeMissing.Add($"Inertial Bodies/Components ({missingCount} missing)");
                }
            }

            if (nodeMissing.Count > 0)
            {
                missingReferences.Add($"• {node.LinkName}: {string.Join(", ", nodeMissing)}");
                nodesWithMissing.Add(node);
            }
        }

        // If there are missing references, show dialog and offer to clear them
        if (missingReferences.Count > 0)
        {
            string message = "The following references could not be found in the current document:\n\n" +
                "(In some cases this is resolved by fully loading the assembly)\n\n" +
                string.Join("\n", missingReferences) +
                "\n\nWould you like to clear these invalid references?\n\n" +
                "Choose 'Yes' to clear, 'No' to keep the invalid references\n" +
                "Choose 'Yes' if you are importing from other CAD software or different assemblies";

            int result = theUI.NXMessageBox.Show("Missing References", NXMessageBox.DialogType.Question, message);

            if (result == 1) // Yes
            {
                ClearMissingReferences(nodesWithMissing, workPart);
                theUI.NXMessageBox.Show("Missing References", NXMessageBox.DialogType.Information,
                    "Invalid references have been cleared.");
            }

            return false;
        }

        return true;
    }

    /// <summary>
    /// Clears references that point to non-existent objects.
    /// </summary>
    private void ClearMissingReferences(List<NXLinkNode> nodes, Part workPart)
    {
        foreach (var node in nodes)
        {
            // Check and clear CSYS
            if (!string.IsNullOrEmpty(node.CoordinateSystemHandle))
            {
                var csys = NXPersistentId.FindCoordinateSystemByKey(workPart, node.CoordinateSystemHandle);
                if (csys == null)
                {
                    node.CoordinateSystemHandle = "";
                }
            }

            // Check and clear Axis
            if (!node.IsRootNode && !string.IsNullOrEmpty(node.JointAxisHandle))
            {
                var axis = NXPersistentId.FindAxisByKey(workPart, node.JointAxisHandle);
                if (axis == null && !Joint.IsAxisFromCsys(node.JointAxisHandle))
                {
                    node.JointAxisHandle = "";
                }
            }

            // Check and clear Visual Bodies/Components - keep only valid ones
            if (node.VisualBodiesHandles != null && node.VisualBodiesHandles.Length > 0)
            {
                node.VisualBodiesHandles = FilterValidHandles(workPart, node.VisualBodiesHandles);
            }

            // Check and clear Collision Bodies/Components - keep only valid ones
            if (node.CollisionBodiesHandles != null && node.CollisionBodiesHandles.Length > 0)
            {
                node.CollisionBodiesHandles = FilterValidHandles(workPart, node.CollisionBodiesHandles);
            }

            // Check and clear Inertial Bodies/Components - keep only valid ones
            if (node.InertialBodiesHandles != null && node.InertialBodiesHandles.Length > 0)
            {
                node.InertialBodiesHandles = FilterValidHandles(workPart, node.InertialBodiesHandles);
            }

            // Update display
            node.UpdateAllColumns();
        }
    }

    /// <summary>
    /// Counts how many handles in the array cannot be resolved (are missing).
    /// Handles both body and component keys.
    /// </summary>
    private int CountMissingHandles(Part workPart, string[] handles)
    {
        if (handles == null || handles.Length == 0)
            return 0;

        int missingCount = 0;
        foreach (var handle in handles)
        {
            if (!IsHandleValid(workPart, handle))
                missingCount++;
        }
        return missingCount;
    }

    /// <summary>
    /// Filters handles to keep only those that can be resolved.
    /// Handles both body and component keys.
    /// </summary>
    private string[] FilterValidHandles(Part workPart, string[] handles)
    {
        if (handles == null || handles.Length == 0)
            return Array.Empty<string>();

        var validHandles = new List<string>();
        foreach (var handle in handles)
        {
            if (IsHandleValid(workPart, handle))
                validHandles.Add(handle);
        }
        return validHandles.ToArray();
    }

    /// <summary>
    /// Checks if a handle (body or component key) can be resolved to an existing object.
    /// </summary>
    private bool IsHandleValid(Part workPart, string handle)
    {
        if (string.IsNullOrEmpty(handle))
            return false;

        if (NXPersistentId.IsComponentKey(handle))
        {
            var component = NXPersistentId.FindComponentByKey(workPart, handle);
            return component != null;
        }
        else
        {
            var bodies = NXPersistentId.FindBodiesByKeys(workPart, new List<string> { handle });
            return bodies.Count > 0;
        }
    }

    /// <summary>
    /// Recursively collects all nodes in the tree.
    /// </summary>
    private void CollectAllNodes(NXLinkNode node, List<NXLinkNode> nodes)
    {
        if (node == null)
            return;

        nodes.Add(node);

        foreach (var child in NXTreeManager.GetChildren(node))
        {
            CollectAllNodes(child, nodes);
        }
    }

    private void GetSelectedNodes(out NXLinkNode selectedNode, out List<NXLinkNode> selectedNodes, out bool multiselect)
    {
        selectedNodes = treeManager.SelectedNodes;
        multiselect = false;
        selectedNode = null;

        if (selectedNodes != null)
        {
            multiselect = selectedNodes.Count > 1;

            if (!multiselect && selectedNodes.Count > 0)
            {
                selectedNode = treeManager.SelectedNodes[0];
            }
        }
    }

    public int UpdateCallback(NXOpen.BlockStyler.UIBlock block)
    {
        try
        {
            if (isUpdatingUI)
                return 0;

            GetSelectedNodes(out NXLinkNode selectedNode, out List<NXLinkNode> selectedNodes, out bool multiselect);

            if (block == enumLinkType && !multiselect)
            {
                OnLinkTypeChanged(selectedNode);
            }
            else if (block == enumSelectBodyComponents)
            {
                OnEnumSelectBodyComponentsChanged();
            }
            else if (block == enumJointType)
            {
                OnJointTypeChanged(selectedNodes);
            }
            else if (block == selectionCsys)
            {
                OnCoordinateSystemChanged(selectedNodes);
            }
            else if (block == toggleFlipAxis)
            {
                OnToggleFlipAxisClicked(selectedNodes);
            }
            else if (block == selectionAxis)
            {
                OnJointAxisChanged(selectedNodes);
            }
            else if (block == enumAxisFromCsys)
            {
                OnEnumAxisFromCsysChanged(selectedNodes);
            }
            else if (block == bodySelectInertial && !multiselect)
            {
                componentToLinkCache = null;
                OnInertialBodiesChanged(selectedNode);
            }
            else if (block == bodySelectCollision && !multiselect)
            {
                componentToLinkCache = null;
                OnCollisionBodiesChanged(selectedNode);
            }
            else if (block == bodySelectVisual && !multiselect)
            {
                componentToLinkCache = null;
                OnVisualBodiesChanged(selectedNode);
            }
            else if (block == buttonCreateSerialChain && !multiselect)
            {
                OnCreateSerialChainClicked(selectedNode);
            }
            else if (block == buttonInsertChildLink && !multiselect)
            {
                OnInsertChildLinkClicked(selectedNode);
            }
            else if (block == buttonInsertParentLink && !multiselect)
            {
                OnInsertParentLinkClicked(selectedNode);
            }
            else if (block == togglePureInertial && !multiselect)
            {
                OnTogglePureInertialClicked(selectedNode);
            }
            else if (block == togglePureVisual && !multiselect)
            {
                OnTogglePureVisualClicked(selectedNode);
            }
            else if (block == buttonExport)
            {
                OnExportButtonClicked();
            }
            else if (block == buttonSave)
            {
                SaveConfiguration();
            }
            else if (block == buttonImportTree)
            {
                OnImportTreeClicked();
            }
            else if (block == buttonExportTree)
            {
                OnExportTreeClicked();
            }
            else if (block == pointSelectRouting)
            {
                OnRoutingPointChanged();
            }
            else if (block == buttonAddTendon)
            {
                OnAddTendonClicked();
            }
            else if (block == stringTendonName)
            {
                OnTendonNameChanged();
            }
            else if (block == buttonAddRoutingElement)
            {
                OnAddRoutingElementClicked();
            }
            else if (block == enumTendonRoutingType)
            {
                OnRoutingTypeChanged();
            }
            else if (block == enumTendonParentLink)
            {
                OnParentLinkChanged();
            }
            else if (block == togglePointAutoAddRouting)
            {
                UpdateTendonTabUI();
            }
            else if (block == doubleTendonCoefficient)
            {
                OnCoefficientChanged();
            }
            else if (block == toggleShowThroughCSYS || block == toggleShowThroughCurves || block == toggleShowThroughPoints)
            {
                UpdateQuickSettings();
            }
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
        }
        return 0;
    }

    private void PrintComponentGuidsRecursive(Component component, System.Text.StringBuilder sb, string indent)
    {
        if (component == null)
            return;

        // Get component GUID if it exists
        string componentGuid = "(none)";
        try
        {
            if (component.HasUserAttribute("URDF_COMPONENT_ID", NXObject.AttributeType.String, -1))
            {
                componentGuid = component.GetStringUserAttribute("URDF_COMPONENT_ID", -1);
            }
        }
        catch { }

        sb.AppendLine($"{indent}Component: {component.Name}");
        sb.AppendLine($"{indent}  GUID: {componentGuid}");

        // Get the component's prototype part and print its objects
        Part componentPart = component.Prototype as Part;
        if (componentPart != null)
        {
            PrintPartGuids(componentPart, sb, indent + "  ");
        }

        // Recurse through children
        Component[] children = component.GetChildren();
        if (children != null && children.Length > 0)
        {
            foreach (Component child in children)
            {
                PrintComponentGuidsRecursive(child, sb, indent + "  ");
            }
        }
    }

    private void PrintPartGuids(Part part, System.Text.StringBuilder sb, string indent)
    {
        // Print Bodies with GUIDs
        int bodyCount = 0;
        foreach (Body body in part.Bodies)
        {
            bodyCount++;
            string bodyGuid = "(none)";
            try
            {
                if (body.HasUserAttribute("URDF_BODY_ID", NXObject.AttributeType.String, -1))
                {
                    bodyGuid = body.GetStringUserAttribute("URDF_BODY_ID", -1);
                }
            }
            catch { }

            if (bodyGuid != "(none)")
            {
                sb.AppendLine($"{indent}Body[{bodyCount}]: Tag={body.Tag}, GUID={bodyGuid}");
            }
        }
        if (bodyCount > 0)
        {
            sb.AppendLine($"{indent}Total Bodies: {bodyCount}");
        }

        // Print Coordinate Systems with GUIDs
        int csysCount = 0;
        foreach (CartesianCoordinateSystem csys in part.CoordinateSystems)
        {
            csysCount++;
            string csysGuid = "(none)";
            try
            {
                if (csys.HasUserAttribute("URDF_CSYS_ID", NXObject.AttributeType.String, -1))
                {
                    csysGuid = csys.GetStringUserAttribute("URDF_CSYS_ID", -1);
                }
            }
            catch { }

            if (csysGuid != "(none)")
            {
                sb.AppendLine($"{indent}CSYS[{csysCount}]: Tag={csys.Tag}, GUID={csysGuid}");
            }
        }
        if (csysCount > 0)
        {
            sb.AppendLine($"{indent}Total Coordinate Systems: {csysCount}");
        }

        // Print Datum Axes with GUIDs
        int axisCount = 0;
        foreach (DisplayableObject datum in part.Datums)
        {
            if (datum is DatumAxis axis)
            {
                axisCount++;
                string axisGuid = "(none)";
                try
                {
                    if (axis.HasUserAttribute("URDF_AXIS_ID", NXObject.AttributeType.String, -1))
                    {
                        axisGuid = axis.GetStringUserAttribute("URDF_AXIS_ID", -1);
                    }
                }
                catch { }

                if (axisGuid != "(none)")
                {
                    sb.AppendLine($"{indent}Axis[{axisCount}]: Tag={axis.Tag}, GUID={axisGuid}");
                }
            }
        }
        if (axisCount > 0)
        {
            sb.AppendLine($"{indent}Total Datum Axes: {axisCount}");
        }
    }


    //------------------------------------------------------------------------------
    // Tree Callbacks
    //------------------------------------------------------------------------------

    /// <summary>
    /// Called when a tree node is selected.
    /// </summary>
    public void OnSelectCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID, bool selected)
    {
        UpdateKinematicTabUI();
    }

    /// <summary>
    /// Called when the context menu should be shown.
    /// </summary>
    public void OnMenuCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID)
    {
        try
        {
            // Create the menu
            TreeListMenu menu = tree.CreateMenu();

            bool multiselect = treeManager.SelectedNodes.Count > 1;

            if (node != null)
            {
                var linkNode = treeManager.GetNode(node);

                if (multiselect)
                {
                    menu.AddMenuItem(MenuRemoveNode, "Remove Links");
                }
                else
                {
                    if (!linkNode.IsSite)
                    {
                        menu.AddMenuItem(MenuAddChild, "Add Child Link");
                        menu.AddMenuItem(MenuAddSite, "Add Site");
                    }

                    if (linkNode != null && !linkNode.IsRootNode)
                    {
                        menu.AddMenuItem(MenuAddSibling, "Add Sibling Link");
                        menu.AddMenuItem(MenuInsertParent, "Insert Parent Link");
                        menu.AddSeparator();
                        menu.AddMenuItem(MenuRemoveNode, "Remove Link");
                        menu.AddSeparator();
                        if (!linkNode.IsSite)
                        {
                            menu.AddMenuItem(MenuConvertToRoot, "Convert to Root Link");
                        }
                    }

                    if (linkNode.IsRootNode)
                    {
                        menu.AddMenuItem(MenuAddNewRoot, "Add New Root Link");
                    }

                    // Move options
                    if (node.PreviousSiblingNode != null || node.NextSiblingNode != null)
                    {
                        menu.AddSeparator();
                        if (node.PreviousSiblingNode != null)
                        {
                            menu.AddMenuItem(MenuMoveUp, "Move Up");
                        }
                        if (node.NextSiblingNode != null)
                        {
                            menu.AddMenuItem(MenuMoveDown, "Move Down");
                        }
                    }
                }
            }

            // Set the menu on the tree
            tree.SetMenu(menu);

            // Dispose the menu after setting it on the tree
            menu.Dispose();
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
        }
    }

    /// <summary>
    /// Called when a context menu item is selected.
    /// </summary>
    public void OnMenuSelectionCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int menuItemID)
    {
        try
        {
            bool multiselect = treeManager.SelectedNodes.Count > 1;

            var linkNode = treeManager.GetNode(node);
            if (linkNode == null)
                return;

            switch (menuItemID)
            {
                case MenuAddChild:
                    AddChildNode(linkNode);
                    break;
                case MenuAddSite:
                    AddChildSiteNode(linkNode);
                    break;
                case MenuAddSibling:
                    AddSiblingNode(linkNode);
                    break;
                case MenuInsertParent:
                    InsertParentNode(linkNode);
                    break;
                case MenuRemoveNode:
                    if (multiselect)
                    {
                        RemoveNodes(treeManager.SelectedNodes);
                    }
                    else
                    {
                        RemoveNode(linkNode);
                    }
                    break;
                case MenuMoveUp:
                    MoveNodeUp(linkNode);
                    break;
                case MenuMoveDown:
                    MoveNodeDown(linkNode);
                    break;
                case MenuAddNewRoot:
                    AddNewRoot(linkNode);
                    break;
                case MenuConvertToRoot:
                    ConvertToRoot(linkNode);
                    break;

            }
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
        }
    }

    /// <summary>
    /// Determines if a node can be dragged.
    /// </summary>
    public Node.DragType IsDragAllowedCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID)
    {
        var linkNode = treeManager.GetNode(node);
        if (linkNode == null || linkNode.IsRootNode || treeManager.SelectedNodes.Count > 1)
        {
            return Node.DragType.None;
        }

        return Node.DragType.All;
    }

    /// <summary>
    /// Determines if a drop is allowed on a target node.
    /// </summary>
    public Node.DropType IsDropAllowedCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID, NXOpen.BlockStyler.Node targetNode, int targetColumnID)
    {
        if (node == null || targetNode == null)
            return Node.DropType.None;

        // Don't allow dropping on self or ancestors
        var current = targetNode;
        while (current != null)
        {
            if (current == node)
                return Node.DropType.None;
            current = current.ParentNode;
        }

        // Don't allow dropping a site onto another site
        var draggedNode = treeManager.GetNode(node);
        var dropTarget = treeManager.GetNode(targetNode);
        if (dropTarget != null && dropTarget.IsSite)
            return Node.DropType.None;

        return Node.DropType.On;
    }

    /// <summary>
    /// Called when a node is dropped onto another node.
    /// </summary>
    public bool OnDropCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node[] nodes, int columnID, NXOpen.BlockStyler.Node targetNode, int targetColumnID, Node.DropType dropType, int dropMenuItemId)
    {
        try
        {
            if (nodes == null || nodes.Length == 0 || targetNode == null)
                return false;

            foreach (var node in nodes)
            {
                var linkNode = treeManager.GetNode(node);
                var targetLinkNode = treeManager.GetNode(targetNode);

                if (linkNode == null || targetLinkNode == null)
                    continue;

                // Use NXTreeManager to handle the reparenting
                // This updates both the model and the tree control
                treeManager.ReparentNode(linkNode, targetLinkNode, dropType);
            }

            return true;
        }
        catch (Exception ex)
        {
            theUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
            return false;
        }
    }

    /// <summary>
    /// Called when label editing begins.
    /// </summary>
    public Tree.BeginLabelEditState OnBeginLabelEditCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID)
    {
        if (columnID == NXLinkNode.ColumnLink)
            return Tree.BeginLabelEditState.Allow;

        return Tree.BeginLabelEditState.Disallow;
    }

    /// <summary>
    /// Called when label editing ends.
    /// </summary>
    public Tree.EndLabelEditState OnEndLabelEditCallback(NXOpen.BlockStyler.Tree tree, NXOpen.BlockStyler.Node node, int columnID, string editedText)
    {
        try
        {
            NXLinkNode selectedNode = treeManager.GetNode(node);

            if (columnID != NXLinkNode.ColumnLink)
                return Tree.EndLabelEditState.RejectText;

            if (string.IsNullOrWhiteSpace(editedText))
            {
                theUI.NXMessageBox.Show("Validation Error", NXMessageBox.DialogType.Warning, "Link name cannot be empty.");
                return Tree.EndLabelEditState.RejectText;
            }

            var linkNode = treeManager.GetNode(node);
            if (linkNode != null)
            {
                linkNode.LinkName = editedText;

                // Update the string block if this is the selected node
                if (selectedNode == linkNode)
                {
                    isUpdatingUI = true;
                    stringLinkName.Value = editedText;
                    isUpdatingUI = false;
                }
            }

            return Tree.EndLabelEditState.AcceptText;
        }
        catch (Exception)
        {
            return Tree.EndLabelEditState.RejectText;
        }
    }

    public void OnStringLinkNameKeystroke(StringBlock block, string text)
    {
        GetSelectedNodes(out NXLinkNode selectedNode, out List<NXLinkNode> selectedNodes, out bool multiselect);

        if (multiselect || selectedNode == null)
        {
            return;
        }

        if (!string.IsNullOrEmpty(text))
        {
            selectedNode.LinkName = text;
            if (toggleAutoJointNaming.Value)
            {
                selectedNode.JointName = LinkNameToJointName(text);
                stringJointName.Value = selectedNode.JointName;
            }
            UpdateKinematicTabUI();
        }

        // Invalidate cache
        componentToLinkCache = null;
    }

    public void OnStringJointNameKeystroke(StringBlock block, string text)
    {
        GetSelectedNodes(out NXLinkNode selectedNode, out List<NXLinkNode> selectedNodes, out bool multiselect);

        if (multiselect || selectedNode == null)
        {
            return;
        }

        if (!string.IsNullOrEmpty(text))
        {
            selectedNode.JointName = text;
            UpdateKinematicTabUI();
        }
    }

    private static string LinkNameToJointName(string linkName)
    {
        // Match "link" as a word separated by _ or - delimiters
        string result = Regex.Replace(linkName, @"(?<=[_\-])link(?=[_\-]|$)|^link(?=[_\-])", m =>
        {
            if (m.Value == "LINK") return "JOINT";
            if (m.Value == "Link") return "Joint";
            return "joint";
        }, RegexOptions.IgnoreCase);

        if (result != linkName)
            return result;

        // Match "Link" at camelCase/PascalCase boundaries
        result = Regex.Replace(linkName, @"(?<=\p{Ll})Link|^Link(?=\p{Lu}|\p{Ll})|^link(?=\p{Lu})", m =>
            m.Value[0] == 'L' ? "Joint" : "joint");

        if (result != linkName)
            return result;

        return "joint_" + linkName;
    }

    //------------------------------------------------------------------------------
    // Tree Tools
    //------------------------------------------------------------------------------

    private void OnCreateSerialChainClicked(NXLinkNode node)
    {
        if (node != null && integerNumLinksInSerialChain.Value > 0)
        {
            for (int i = 0; i < integerNumLinksInSerialChain.Value; i++)
            {
                node = treeManager.CreateChildNode(node);
            }
            treeControlKinematicTree.SelectNode(node.TreeNode, true, true);
        }
    }

    private void OnInsertChildLinkClicked(NXLinkNode node)
    {
        if (node != null)
        {
            AddChildNode(node);
        }
    }

    private void OnInsertParentLinkClicked(NXLinkNode node)
    {
        if (node != null)
        {
            InsertParentNode(node);
        }
    }

    private void OnImportTreeClicked()
    {
        try
        {
            // Open file dialog to select text file
            using (var openFileDialog = new OpenFileDialog())
            {
                openFileDialog.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
                openFileDialog.Title = "Import Tree Structure";
                openFileDialog.DefaultExt = "txt";

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    string filePath = openFileDialog.FileName;

                    // Parse the file
                    if (TreeTextImporter.TryParse(System.IO.File.ReadAllText(filePath), out var importedRoot, out string error))
                    {
                        // Confirm if tree already has content
                        var existingRoot = treeManager.GetRootNode();
                        if (existingRoot != null)
                        {
                            var result = theUI.NXMessageBox.Show("Import Tree",
                                NXMessageBox.DialogType.Question,
                                "This will replace the existing tree. Continue?");

                            if (result != 1) // 1 = Yes
                                return;

                            // Delete existing tree
                            treeControlKinematicTree.DeleteNode(existingRoot.TreeNode);
                        }

                        // Build tree from imported structure
                        BuildTreeFromImported(importedRoot);

                        // Expand all and select root
                        var newRoot = treeManager.GetRootNode();
                        if (newRoot != null)
                        {
                            newRoot.TreeNode.Expand(Node.ExpandOption.Expand);
                            treeControlKinematicTree.SelectNode(newRoot.TreeNode, true, true);
                        }

                        theUI.NXMessageBox.Show("Import Tree", NXMessageBox.DialogType.Information,
                            $"Successfully imported tree from:\n{filePath}");
                    }
                    else
                    {
                        theUI.NXMessageBox.Show("Import Tree", NXMessageBox.DialogType.Error,
                            $"Failed to parse file:\n{error}");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            logger.Error($"OnImportTreeClicked error: {ex.Message}");
            theUI.NXMessageBox.Show("Import Tree", NXMessageBox.DialogType.Error, ex.Message);
        }
    }

    private void OnExportTreeClicked()
    {
        try
        {
            var rootNode = treeManager.GetRootNode();
            if (rootNode == null)
            {
                theUI.NXMessageBox.Show("Export Tree", NXMessageBox.DialogType.Warning,
                    "No tree to export.");
                return;
            }

            // Build ImportedTreeNode structure from current tree
            var exportedRoot = BuildExportedTree(rootNode);
            string text = TreeTextImporter.Export(exportedRoot);

            // Open save file dialog
            using (var saveFileDialog = new SaveFileDialog())
            {
                saveFileDialog.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
                saveFileDialog.Title = "Export Tree Structure";
                saveFileDialog.DefaultExt = "txt";
                saveFileDialog.FileName = $"{rootNode.LinkName}_tree.txt";

                if (saveFileDialog.ShowDialog() == DialogResult.OK)
                {
                    System.IO.File.WriteAllText(saveFileDialog.FileName, text);
                    theUI.NXMessageBox.Show("Export Tree", NXMessageBox.DialogType.Information,
                        $"Successfully exported tree to:\n{saveFileDialog.FileName}");
                }
            }
        }
        catch (Exception ex)
        {
            logger.Error($"OnExportTreeClicked error: {ex.Message}");
            theUI.NXMessageBox.Show("Export Tree", NXMessageBox.DialogType.Error, ex.Message);
        }
    }

    private void BuildTreeFromImported(ImportedTreeNode importedRoot)
    {
        isBulkPopulating = true;

        // Create root node
        var rootNode = treeManager.CreateRootNode(importedRoot.LinkName);

        // Recursively create children
        foreach (var child in importedRoot.Children)
        {
            CreateNodeFromImported(child, rootNode);
        }

        isBulkPopulating = false;
    }

    private void CreateNodeFromImported(ImportedTreeNode imported, NXLinkNode parent)
    {
        var node = treeManager.CreateChildNode(parent);
        node.LinkName = imported.LinkName;
        node.JointName = imported.JointName ?? $"joint_{imported.LinkName}";
        node.JointType = imported.JointType;
        node.UpdateAllColumns();

        foreach (var child in imported.Children)
        {
            CreateNodeFromImported(child, node);
        }
    }

    private ImportedTreeNode BuildExportedTree(NXLinkNode nxNode)
    {
        var exported = new ImportedTreeNode(nxNode.LinkName, nxNode.JointName, nxNode.JointType);

        var children = NXTreeManager.GetChildren(nxNode);
        foreach (var child in children)
        {
            var childExported = BuildExportedTree(child);
            childExported.Parent = exported;
            exported.Children.Add(childExported);
        }

        return exported;
    }

    //------------------------------------------------------------------------------
    // Node Selection and UI Update
    //------------------------------------------------------------------------------

    private void CleanUpHighlighting()
    {
        //NXOpen.PartCleanup partCleanup;
        //partCleanup = theSession.NewPartCleanup();
        //partCleanup.TurnOffHighlighting = true;
        //partCleanup.DoCleanup();
        //partCleanup.Dispose();
    }

    private void UpdateKinematicTabUI()
    {
        treeManager.UpdateSelectedNodes();

        CleanUpHighlighting();

        isUpdatingUI = true;

        var selectedNodes = treeManager.SelectedNodes;

        // Undefined state, early out
        if (selectedNodes == null)
        {
            isUpdatingUI = false;
            return;
        }

        // No nodes selected, disable everything, early out
        if (selectedNodes.Count == 0)
        {
            groupLinkProperties.Enable = false;
            groupJointProperties.Enable = false;
            groupLinkBodies.Enable = false;
            groupTreeTools.Enable = false;

            isUpdatingUI = false;
            return;
        }

        bool multiSelect = false;
        bool multiSelectHasSite = false;
        bool multiSelectHasBase = false;

        if (selectedNodes.Count > 1)
        {
            multiSelect = true;

            foreach (var node in selectedNodes)
            {
                if (node.IsSite)
                {
                    multiSelectHasSite = true;
                }

                if (node.IsRootNode)
                {
                    multiSelectHasBase = true;
                }
            }
        }

        // Multiselect handling for groups
        if (multiSelect)
        {
            groupLinkProperties.Enable = false;
            groupLinkBodies.Enable = false;
            groupTreeTools.Enable = false;
        }
        else
        {
            groupLinkProperties.Enable = true;
            groupLinkBodies.Enable = true;
            groupTreeTools.Enable = true;
        }

        // Multiselect handling for individual UI Blocks
        if (multiSelect)
        {
            stringLinkName.Value = "-";
            stringJointName.Value = "-";
            labelParentLink.Label = "-";
            enumAxisFromCsys.ValueAsString = GetJointAxisFromCsysDisplayText("none");
            groupLinkProperties.Label = $"Link: (multiple)";

            groupJointProperties.Enable = !multiSelectHasBase;

            selectionAxis.Enable = !multiSelectHasSite;
            toggleFlipAxis.Enable = !multiSelectHasSite;
            enumJointType.Enable = !multiSelectHasSite;
            enumAxisFromCsys.Enable = !multiSelectHasSite;
        }

        groupJointProperties.Enable = true;

        if (!multiSelect)
        {
            var selectedNode = selectedNodes[0];
            bool isBaseLink = selectedNode.IsRootNode;
            bool isFixedJoint = selectedNode.JointType == "fixed";
            bool axisFromCsys = Joint.IsAxisFromCsys(selectedNode.JointAxisHandle);
            bool isSite = selectedNode.IsSite;

            stringLinkName.Value = selectedNode.LinkName;

            if (isBaseLink)
            {
                labelParentLink.Label = "(none)";
            }
            else
            {
                var parent = treeManager.GetParent(selectedNode);
                labelParentLink.Label = parent?.LinkName ?? "unknown";
            }

            // Base links and sites do not have joints
            stringJointName.Enable = !isBaseLink && !isSite;
            enumJointType.Enable = !isBaseLink && !isSite;

            // Sites cannot have bodies
            groupLinkBodies.Enable = !isSite;

            // Fixed joints do not have axes
            selectionAxis.Enable = !isBaseLink && !isFixedJoint && !isSite;
            toggleFlipAxis.Enable = !isBaseLink && !isFixedJoint && !isSite;
            enumAxisFromCsys.Enable = !isBaseLink && !isFixedJoint && !isSite;

            if (isBaseLink || isSite)
            {
                stringJointName.Value = "";
            }
            else
            {
                stringJointName.Value = selectedNode.JointName;
                enumJointType.ValueAsString = GetJointTypeDisplayText(selectedNode.JointType);

                // Update axis visibility based on joint type
                string jointType = selectedNode.JointType;
                selectionAxis.Enable = !isFixedJoint && !axisFromCsys;
                enumAxisFromCsys.Enable = !isFixedJoint;
            }

            UpdateCSYSSelector(selectedNode);

            if (!isSite)
            {
                UpdateBodySelectors(selectedNode);
            }

            if (!isBaseLink && !isSite && !isFixedJoint)
            {
                UpdateAxisSelector(selectedNode);
            }

            if (isSite)
            {
                enumLinkType.ValueAsString = "Site";
                groupLinkProperties.Label = $"Site: {selectedNode.LinkName}";
            }
            else
            {
                enumLinkType.ValueAsString = "Link";
                groupLinkProperties.Label = $"Link: {selectedNode.LinkName}";
            }
        }

        if (lastSelectedKinematicBlock != null && lastSelectedKinematicBlock.Enable)
        {
            lastSelectedKinematicBlock.Focus();
        }

        isUpdatingUI = false;
    }

    /// <summary>
    /// Called when a new node is created.
    /// </summary>
    private void OnNodeCreated(NXLinkNode node)
    {
        // Auto-expand parent node
        var parentNode = node.TreeNode.ParentNode;
        if (parentNode != null)
        {
            parentNode.Expand(Node.ExpandOption.Expand);
        }

        if (!isBulkPopulating)
        {
            UpdateKinematicTabUI();
        }
    }

    private void EnableAllControls(bool enable)
    {
        groupLinkProperties.Enable = enable;
        groupJointProperties.Enable = enable;
        groupLinkBodies.Enable = enable;
        groupTreeTools.Enable = enable;
        groupOptions.Enable = enable;
        treeControlKinematicTree.Enable = enable;
        buttonExport.Enable = enable;
        buttonSave.Enable = enable;
        groupTendons.Enable = enable;
        groupRoutingElements.Enable = enable;
    }

    /// <summary>
    /// Updates body selector controls with the current node's body selections.
    /// </summary>
    private void UpdateBodySelectors(NXLinkNode node)
    {
        try
        {
            SetBodyCollectorFromHandles(bodySelectVisual, node.VisualBodiesHandles);
            SetBodyCollectorFromHandles(bodySelectCollision, node.CollisionBodiesHandles);
            SetBodyCollectorFromHandles(bodySelectInertial, node.InertialBodiesHandles);

            // Update pure visual/inertial toggles based on current selections
            int visualCount = node.VisualBodiesHandles?.Length ?? 0;
            int inertialCount = node.InertialBodiesHandles?.Length ?? 0;

            // Set toggle values from stored data
            togglePureVisual.Value = node.PureVisual;
            togglePureInertial.Value = node.PureInertial;

            // Handle mutual exclusivity enable states
            // Pure Visual is enabled only if inertial is empty and visual has items
            togglePureVisual.Enable = (inertialCount == 0 && visualCount > 0);
            // Pure Inertial is enabled only if visual is empty and inertial has items
            togglePureInertial.Enable = (visualCount == 0 && inertialCount > 0);
        }
        catch (Exception)
        {
            // Body selectors may not be fully initialized
        }
    }

    /// <summary>
    /// Sets the selected objects in a BodyCollector from persistent handles.
    /// Supports both Body and Component handles.
    /// </summary>
    private void SetBodyCollectorFromHandles(SelectObject bodyCollector, string[] handles)
    {
        if (bodyCollector == null)
            return;

        var handleList = handles != null ? new List<string>(handles) : new List<string>();
        var objects = GetObjectsFromHandles(handleList);
        bodyCollector.SetSelectedObjects(objects);
    }

    private void UpdateCSYSSelector(NXLinkNode node)
    {
        try
        {
            string csysHandle = node.CoordinateSystemHandle;
            if (!string.IsNullOrEmpty(csysHandle))
            {
                CartesianCoordinateSystem csysObject = GetCoordinateSystemFromGuid(csysHandle);
                if (csysObject != null)
                {
                    selectionCsys.SetSelectedObjects(new TaggedObject[] { csysObject });
                }
                else
                {
                    selectionCsys.SetSelectedObjects(new TaggedObject[0]);
                }
            }
            else
            {
                selectionCsys.SetSelectedObjects(new TaggedObject[0]);
            }
        }
        catch (Exception)
        {
            // TODO: handle exceptions
        }
    }

    private void UpdateAxisSelector(NXLinkNode node)
    {
        if (node.IsRootNode)
        {
            // Clear selection for root node (base link has no axis)
            selectionAxis.SetSelectedObjects(new TaggedObject[0]);
            enumAxisFromCsys.ValueAsString = GetJointAxisFromCsysDisplayText("none");
            return;
        }

        try
        {
            string axisHandle = node.JointAxisHandle;
            if (!string.IsNullOrEmpty(axisHandle))
            {
                // Check if the axis handle is a CSYS axis magic keyword
                if (Joint.IsAxisFromCsys(axisHandle))
                {
                    // Set the enum dropdown to reflect the CSYS axis selection
                    string axisKey = "none";
                    if (axisHandle == Joint.AxisFromCsysX) axisKey = "x";
                    else if (axisHandle == Joint.AxisFromCsysY) axisKey = "y";
                    else if (axisHandle == Joint.AxisFromCsysZ) axisKey = "z";

                    enumAxisFromCsys.ValueAsString = GetJointAxisFromCsysDisplayText(axisKey);
                    selectionAxis.SetSelectedObjects(new TaggedObject[0]);
                }
                else
                {
                    // Regular DatumAxis handle - try to resolve it
                    enumAxisFromCsys.ValueAsString = GetJointAxisFromCsysDisplayText("none");
                    DatumAxis axisObject = GetAxisFromGuid(axisHandle);
                    if (axisObject != null)
                    {
                        selectionAxis.SetSelectedObjects(new TaggedObject[] { axisObject });
                    }
                    else
                    {
                        selectionAxis.SetSelectedObjects(new TaggedObject[0]);
                    }
                }
            }
            else
            {
                selectionAxis.SetSelectedObjects(new TaggedObject[0]);
                enumAxisFromCsys.ValueAsString = GetJointAxisFromCsysDisplayText("none");
            }

            toggleFlipAxis.Value = node.FlipAxis;
        }
        catch (Exception)
        {
            // Joint selectors may not be fully initialized
        }
    }

    //------------------------------------------------------------------------------
    // UI Change Handlers
    //------------------------------------------------------------------------------

    private void OnLinkTypeChanged(NXLinkNode node)
    {
        if (node == null) return;

        string selectedType = enumLinkType.ValueAsString;

        if (selectedType == "Site")
        {
            node.IsSite = true;
            node.JointType = "fixed";
            node.JointName = "";
            node.JointAxisHandle = "";
            node.VisualBodiesHandles = Array.Empty<string>();
            node.CollisionBodiesHandles = Array.Empty<string>();
            node.InertialBodiesHandles = Array.Empty<string>();
            node.PureVisual = false;
            node.PureInertial = false;

            if (!node.LinkName.StartsWith("site_"))
            {
                node.LinkName = "site_" + node.LinkName;
            }
        }
        else
        {
            node.IsSite = false;
        }

        node.UpdateAllColumns();

        UpdateKinematicTabUI();
    }

    private void OnJointTypeChanged(IEnumerable<NXLinkNode> nodes)
    {
        foreach (var node in nodes)
        {
            OnJointTypeChanged(node);
        }

        UpdateKinematicTabUI();
    }

    private void OnJointTypeChanged(NXLinkNode node)
    {
        if (!node.IsRootNode)
        {
            string selectedValue = enumJointType.ValueAsString;
            string jointType = GetJointTypeFromDisplayText(selectedValue);
            node.JointType = jointType;
            node.UpdateAllColumns();  // Update display columns
        }
    }

    private void OnCoordinateSystemChanged(IEnumerable<NXLinkNode> nodes)
    {
        foreach (var node in nodes)
        {
            OnCoordinateSystemChanged(node);
        }

        UpdateKinematicTabUI();

        if (nodes.Count() > 1 && toggleAutoAdvanceCsys.Value)
        {
            if (enumAxisFromCsys.ValueAsString != "none")
            {
                bodySelectInertial.Focus();
            }
            else
            {
                selectionAxis.Focus();
            }
        }
    }

    private void OnCoordinateSystemChanged(NXLinkNode node)
    {
        try
        {
            TaggedObject[] selectedObjects = selectionCsys.GetSelectedObjects();
            if (selectedObjects == null || selectedObjects.Length == 0)
            {
                if (toggleDeselectGuardDatums.Value && !string.IsNullOrEmpty(node.CoordinateSystemHandle))
                {
                    if (!ConfirmDeselection("Coordinate System"))
                        return;
                }
                node.CoordinateSystemHandle = "";
                node.UpdateAllColumns();
                return;
            }

            if (selectedObjects[0] is CartesianCoordinateSystem csys)
            {
                if (WaveLinker.IsFromDifferentPart(csys) || !NXPersistentId.CoordinateSystemHasKey(csys))
                {
                    var behavior = WaveLinker.ParseSelectionBehavior(enumSelectionBehavior.ValueAsString);
                    var crossPartResult = WaveLinker.HandleCrossPartSelection(
                        csys, behavior,
                        c => WaveLinker.CreateWaveDatumLink(theSession.Parts.Work, c, $"CSYS - {node.LinkName}"),
                        suppressOwnershipWarning);

                    if (crossPartResult.WarningShown)
                        ShowOwnershipWarning();

                    if (!crossPartResult.Allowed)
                    {
                        selectionCsys.SetSelectedObjects(new TaggedObject[0]);
                        return;
                    }

                    if (crossPartResult.ResolvedObject != csys)
                    {
                        csys = crossPartResult.ResolvedObject;
                        selectionCsys.SetSelectedObjects(new TaggedObject[] { csys });
                    }
                }

                string compositeKey = NXPersistentId.GetOrCreateCoordinateSystemKey(csys);
                node.CoordinateSystemHandle = compositeKey ?? "";
                node.UpdateAllColumns();

                if (toggleAutoAdvanceCsys.Value)
                {
                    selectionAxis.Focus();
                }
            }
        }
        catch (Exception ex)
        {
            logger.Warning($"OnCoordinateSystemChanged error: {ex.Message}");
        }
    }

    private void OnJointAxisChanged(IEnumerable<NXLinkNode> nodes)
    {
        foreach (var node in nodes)
        {
            OnJointAxisChanged(node);
        }

        UpdateKinematicTabUI();

        if (nodes.Count() > 1 && toggleAutoAdvanceCsys.Value)
        {
            bodySelectInertial.Focus();
        }
    }

    private void OnJointAxisChanged(NXLinkNode node)
    {
        try
        {
            if (node.IsRootNode)
                return;

            TaggedObject[] selectedObjects = selectionAxis.GetSelectedObjects();
            if (selectedObjects == null || selectedObjects.Length == 0)
            {
                if (toggleDeselectGuardDatums.Value && !string.IsNullOrEmpty(node.JointAxisHandle))
                {
                    if (!ConfirmDeselection("Joint Axis"))
                        return;
                }
                node.JointAxisHandle = "";
                node.UpdateAllColumns();
                return;
            }

            if (selectedObjects[0] is DatumAxis axis)
            {
                if (WaveLinker.IsFromDifferentPart(axis) || !NXPersistentId.AxisHasKey(axis))
                {
                    var behavior = WaveLinker.ParseSelectionBehavior(enumSelectionBehavior.ValueAsString);
                    var crossPartResult = WaveLinker.HandleCrossPartSelection(
                        axis, behavior,
                        a => WaveLinker.CreateWaveAxisLink(theSession.Parts.Work, a, $"Axis - {node.LinkName}"),
                        suppressOwnershipWarning);

                    if (crossPartResult.WarningShown)
                        ShowOwnershipWarning();

                    if (!crossPartResult.Allowed)
                    {
                        selectionAxis.SetSelectedObjects(new TaggedObject[0]);
                        return;
                    }

                    if (crossPartResult.ResolvedObject != axis)
                    {
                        axis = crossPartResult.ResolvedObject;
                        selectionAxis.SetSelectedObjects(new TaggedObject[] { axis });
                    }
                }

                string compositeKey = NXPersistentId.GetOrCreateAxisKey(axis);
                node.JointAxisHandle = compositeKey ?? "";
                node.UpdateAllColumns();
            }
        }
        catch (Exception ex)
        {
            logger.Warning($"OnJointAxisChanged error: {ex.Message}");
        }
    }

    private void OnEnumAxisFromCsysChanged(IEnumerable<NXLinkNode> nodes)
    {
        foreach (var node in nodes)
        {
            OnEnumAxisFromCsysChanged(node);
        }

        UpdateKinematicTabUI();
    }

    private void OnEnumAxisFromCsysChanged(NXLinkNode node)
    {
        if (!node.IsRootNode && node.JointType != "fixed")
        {
            string selectedValue = enumAxisFromCsys.ValueAsString;
            string axisKey = GetJointAxisFromCsysFromDisplayText(selectedValue);

            // Set the magic keyword in the axis handle based on the selection
            switch (axisKey)
            {
                case "x":
                    node.JointAxisHandle = Joint.AxisFromCsysX;
                    break;
                case "y":
                    node.JointAxisHandle = Joint.AxisFromCsysY;
                    break;
                case "z":
                    node.JointAxisHandle = Joint.AxisFromCsysZ;
                    break;
                case "none":
                default:
                    node.JointAxisHandle = "";
                    break;
            }

            // Clear the datum axis selection (UI only) since we're using CSYS axis
            if (axisKey != "none")
            {
                selectionAxis.SetSelectedObjects(new TaggedObject[] { });
                selectionAxis.Enable = false;
            }
            else
            {
                selectionAxis.Enable = true;
            }
        }

        node.UpdateAllColumns();
    }

    private void OnEnumSelectBodyComponentsChanged()
    {
        LinkSelectionMode mode = GetLinkSelectModeFromDisplayText(enumSelectBodyComponents.ValueAsString);

        SetBodySelectionMask(mode);
    }

    private void OnVisualBodiesChanged(NXLinkNode node)
    {
        try
        {
            TaggedObject[] currentSelection = bodySelectVisual.GetSelectedObjects();
            if ((currentSelection == null || currentSelection.Length == 0) &&
                toggleDeselectGuardBodies.Value && node.VisualBodiesHandles != null && node.VisualBodiesHandles.Length > 0)
            {
                if (!ConfirmDeselection("Visual Bodies"))
                {
                    UpdateKinematicTabUI();
                    return;
                }
            }

            var allowSelection = ProcessBodyCrossPartSelections(bodySelectVisual, node.LinkName, "Visual Body");

            if (!allowSelection)
            {
                // This essentially cancels the selection and reverts the selection to the previous state
                UpdateKinematicTabUI();
                return;
            }

            node.VisualBodiesHandles = GetSelectedBodyKeys(bodySelectVisual).ToArray();
            node.UpdateAllColumns();  // Update display columns

            // Handle mutual exclusivity of Pure Visual/Pure Inertial toggles
            // When visual bodies are selected, disable Pure Inertial option
            int visualCount = bodySelectVisual.GetSelectedObjects()?.Length ?? 0;
            int inertialCount = bodySelectInertial.GetSelectedObjects()?.Length ?? 0;

            if (visualCount > 0)
            {
                togglePureInertial.Value = false;
                togglePureInertial.Enable = false;
                node.PureInertial = false;
            }

            // Re-enable Pure Visual if inertial is empty and visual has items
            if (visualCount == 0 && inertialCount > 0)
            {
                togglePureInertial.Enable = true;
            }
            if (visualCount > 0 && inertialCount == 0)
            {
                togglePureVisual.Enable = true;
            }

            if (visualCount > 0 && toggleAutoAdvanceBody.Value)
            {
                bodySelectInertial.Focus();
            }
        }
        catch (Exception ex)
        {
            logger.Warning($"OnVisualBodiesChanged error: {ex.Message}");
        }
    }

    private void OnCollisionBodiesChanged(NXLinkNode node)
    {
        try
        {
            TaggedObject[] currentSelection = bodySelectCollision.GetSelectedObjects();
            if ((currentSelection == null || currentSelection.Length == 0) &&
                toggleDeselectGuardBodies.Value && node.CollisionBodiesHandles != null && node.CollisionBodiesHandles.Length > 0)
            {
                if (!ConfirmDeselection("Collision Bodies"))
                {
                    UpdateKinematicTabUI();
                    return;
                }
            }

            var allowSelection = ProcessBodyCrossPartSelections(bodySelectCollision, node.LinkName, "Collision Body");

            if (!allowSelection)
            {
                // This essentially cancels the selection and reverts the selection to the previous state
                UpdateKinematicTabUI();
                return;
            }

            node.CollisionBodiesHandles = GetSelectedBodyKeys(bodySelectCollision).ToArray();
            node.UpdateAllColumns();  // Update display columns

            if (node.CollisionBodiesHandles.Length > 0 && toggleAutoAdvanceBody.Value)
            {
                bodySelectVisual.Focus();
            }
        }
        catch (Exception ex)
        {
            logger.Warning($"OnCollisionBodiesChanged error: {ex.Message}");
        }
    }

    private void OnInertialBodiesChanged(NXLinkNode node)
    {
        try
        {
            TaggedObject[] currentSelection = bodySelectInertial.GetSelectedObjects();
            if ((currentSelection == null || currentSelection.Length == 0) &&
                toggleDeselectGuardBodies.Value && node.InertialBodiesHandles != null && node.InertialBodiesHandles.Length > 0)
            {
                if (!ConfirmDeselection("Inertial Bodies"))
                {
                    UpdateKinematicTabUI();
                    return;
                }
            }

            var allowSelection = ProcessBodyCrossPartSelections(bodySelectInertial, node.LinkName, "Inertial Body");

            if (!allowSelection)
            {
                // This essentially cancels the selection and reverts the selection to the previous state
                UpdateKinematicTabUI();
                return;
            }

            node.InertialBodiesHandles = GetSelectedBodyKeys(bodySelectInertial).ToArray();
            node.UpdateAllColumns();  // Update display columns

            // Handle mutual exclusivity of Pure Visual/Pure Inertial toggles
            // When inertial bodies are selected, disable Pure Visual option
            int visualCount = bodySelectVisual.GetSelectedObjects()?.Length ?? 0;
            int inertialCount = bodySelectInertial.GetSelectedObjects()?.Length ?? 0;

            if (inertialCount > 0)
            {
                togglePureVisual.Value = false;
                togglePureVisual.Enable = false;
                node.PureVisual = false;
            }

            // Re-enable Pure Inertial if visual is empty and inertial has items
            if (visualCount == 0 && inertialCount > 0)
            {
                togglePureInertial.Enable = true;
            }
            if (visualCount > 0 && inertialCount == 0)
            {
                togglePureVisual.Enable = true;
            }

            if (inertialCount > 0 && toggleAutoAdvanceBody.Value)
            {
                bodySelectCollision.Focus();
            }
        }
        catch (Exception ex)
        {
            logger.Warning($"OnInertialBodiesChanged error: {ex.Message}");
        }
    }

    /// <summary>
    /// Checks body selections for read-only parts and warns the user.
    /// Body WAVE linking is not automated — users should prefer component selection
    /// or create WAVE links manually.
    /// </summary>
    private bool ProcessBodyCrossPartSelections(SelectObject bodyCollector, string linkName, string bodyLabel)
    {
        var behavior = WaveLinker.ParseSelectionBehavior(enumSelectionBehavior.ValueAsString);

        TaggedObject[] selectedObjects = bodyCollector.GetSelectedObjects();
        if (selectedObjects == null || selectedObjects.Length == 0)
            return true;

        var lastSelectedObject = selectedObjects.Last();

        if (lastSelectedObject is Body body
            && WaveLinker.IsFromDifferentPart(body)
            && WaveLinker.IsOwningPartReadOnly(body)
            && !NXPersistentId.BodyHasKey(body))
        {
            if (!suppressBodyOwnershipWarning)
                ShowBodyOwnershipWarning();

            if (behavior == SelectionBehavior.IgnoreReadOnly
                || behavior == SelectionBehavior.OnlyWaveLinkReadOnly
                || behavior == SelectionBehavior.AlwaysCreateWaveLinks)
                return false;

            if (behavior == SelectionBehavior.AllowReadOnly)
            {
                return true;
            }
        }

        return true;
    }

    /// <summary>
    /// Gets GUIDs for all selected bodies in a body collector.
    /// </summary>
    /// <summary>
    /// Gets composite keys for all selected bodies in a body collector.
    /// Uses NXPersistentId to get/create composite keys (ComponentPath|ObjectGUID) on prototypes.
    /// </summary>
    private List<string> GetSelectedBodyKeys(SelectObject bodyCollector)
    {
        var keys = new List<string>();
        if (bodyCollector == null)
            return keys;

        TaggedObject[] selectedObjects = bodyCollector.GetSelectedObjects();
        if (selectedObjects == null)
            return keys;

        foreach (TaggedObject obj in selectedObjects)
        {
            // Use unified key creation that handles both Body and Component objects
            string key = NXPersistentId.GetOrCreateSelectionKey(obj);
            if (!string.IsNullOrEmpty(key))
                keys.Add(key);
        }

        return keys;
    }

    private void OnTogglePureVisualClicked(NXLinkNode node)
    {
        if (node != null)
        {
            node.PureVisual = togglePureVisual.Value;
        }
    }

    private void OnTogglePureInertialClicked(NXLinkNode node)
    {
        if (node != null)
        {
            node.PureInertial = togglePureInertial.Value;
        }
    }

    private void OnToggleFlipAxisClicked(IEnumerable<NXLinkNode> nodes)
    {
        foreach (var node in nodes)
        {
            OnToggleFlipAxisClicked(node);
        }

        UpdateKinematicTabUI();
    }

    private void OnToggleFlipAxisClicked(NXLinkNode node)
    {
        if (node != null)
        {
            node.FlipAxis = toggleFlipAxis.Value;
            node.UpdateAllColumns();  // Update display columns
        }
    }

    private void OnExportButtonClicked()
    {
        if (ValidateAndReportMissingReferences(treeManager.GetRootNode()) && ValidateTree())
        {
            // Build LinkNode tree from NXLinkNode tree
            LinkNode baseNode = BuildLinkNodeFromNXLinkNode(treeManager.GetRootNode());

            // Create ExportHelper with the NX bridge
            var exporter = new ExportHelper(nxBridge);

            // Build the robot from the tree view
            bool exportSuccess = exporter.CreateRobotFromTreeView(baseNode);

            if (exportSuccess)
            {
                // Attach tendons to the robot for export
                if (exporter.Robot != null && tendons.Count > 0)
                {
                    exporter.Robot.Tendons.Clear();
                    exporter.Robot.Tendons.AddRange(tendons);
                    exporter.LocalizeTendonPositions();
                }
                EnableAllControls(false);
                // Show the export form
                AssemblyExportForm exportForm = new AssemblyExportForm(baseNode, exporter);
                exportForm.Exporter = exporter;

                // Own the form to the NX main window so it hides/restores with NX
                IntPtr nxHwnd = CADRobotExporter.Utilities.NativeWindowWrapper.GetHostMainWindowHandle();
                if (nxHwnd != IntPtr.Zero)
                {
                    exportForm.Show(new CADRobotExporter.Utilities.NativeWindowWrapper(nxHwnd));
                }
                else
                {
                    exportForm.Show();
                }
                // HACK: NX steals focus back after the BlockStyler callback returns,
                // so use a short timer to reclaim focus once NX is done.
                var focusTimer = new System.Windows.Forms.Timer { Interval = 200 };
                focusTimer.Tick += (s, ev) =>
                {
                    focusTimer.Stop();
                    focusTimer.Dispose();
                    if (!exportForm.IsDisposed)
                    {
                        exportForm.Activate();
                    }
                };
                focusTimer.Start();
                exportForm.FormClosed += new FormClosedEventHandler(OnAssemblyExportFormClosed);
            }
        }
    }

    private void OnAssemblyExportFormClosed(object sender, EventArgs e)
    {
        componentToLinkCache = null;
        // reload everything since form may have changed some properties
        LoadExistingConfiguration();
        EnableAllControls(true);
        UpdateKinematicTabUI();
        UpdateTendonTabUI();
    }

    /// <summary>
    /// Recursively builds a LinkNode tree from an NXLinkNode tree.
    /// </summary>
    private LinkNode BuildLinkNodeFromNXLinkNode(NXLinkNode nxNode)
    {
        if (nxNode == null)
            return null;

        nxNode.CheckIncomplete(out string whyIncomplete);

        var linkNode = new LinkNode
        {
            Link = nxNode.ToLink(),
            Name = nxNode.LinkName,
            Text = nxNode.LinkName,
            IsBaseNode = nxNode.IsRootNode,
            IsIncomplete = nxNode.IsIncomplete,
            WhyIncomplete = whyIncomplete
        };

        // Get child NXLinkNodes from the tree and recursively build
        foreach (var childNXNode in NXTreeManager.GetChildren(nxNode))
        {
            var childLinkNode = BuildLinkNodeFromNXLinkNode(childNXNode);
            if (childLinkNode != null)
            {
                linkNode.Nodes.Add(childLinkNode);
            }
        }

        return linkNode;
    }

    //------------------------------------------------------------------------------
    // Node Manipulation Methods
    //------------------------------------------------------------------------------

    private void AddChildNode(NXLinkNode parentNode)
    {
        var newNode = treeManager.CreateChildNode(parentNode);
        treeControlKinematicTree.SelectNode(newNode.TreeNode, true, true);
        treeManager.UpdateSelectedNodes();
        UpdateKinematicTabUI();
    }

    private void AddChildSiteNode(NXLinkNode parentNode)
    {
        var newNode = treeManager.CreateChildSiteNode(parentNode);
        treeControlKinematicTree.SelectNode(newNode.TreeNode, true, true);
        treeManager.UpdateSelectedNodes();
        UpdateKinematicTabUI();
    }

    private void AddSiblingNode(NXLinkNode node)
    {
        var parentTreeNode = node.TreeNode.ParentNode;
        if (parentTreeNode == null)
            return;

        var parentNode = treeManager.GetNode(parentTreeNode);
        if (parentNode != null)
        {
            var newNode = treeManager.CreateChildNode(parentNode);
            treeControlKinematicTree.SelectNode(newNode.TreeNode, true, true);
            treeManager.UpdateSelectedNodes();
        }
        UpdateKinematicTabUI();
    }

    private void InsertParentNode(NXLinkNode node)
    {
        var newParent = treeManager.InsertParentNode(node);
        if (newParent != null)
        {
            treeControlKinematicTree.SelectNode(newParent.TreeNode, true, true);
            treeManager.UpdateSelectedNodes();
            NXTreeManager.ExpandNodesRecursively(newParent);
        }
        UpdateKinematicTabUI();
    }

    private void RemoveNode(NXLinkNode node)
    {
        if (node.IsRootNode)
        {
            theUI.NXMessageBox.Show("Cannot Remove", NXMessageBox.DialogType.Warning,
                "Cannot remove the base link.");
            return;
        }

        // Confirm deletion if node has children
        var firstChild = node.TreeNode.FirstChildNode;
        if (firstChild != null)
        {
            int result = theUI.NXMessageBox.Show("Warning", NXMessageBox.DialogType.Question,
                "Removing a link will also remove all child links. Continue?");
            if (result != 1) // 1 == yes
            {
                return;
            }
        }

        var parentTreeNode = node.TreeNode.ParentNode;
        if (parentTreeNode != null)
        {
            treeControlKinematicTree.SelectNode(parentTreeNode, true, true);
            treeManager.UpdateSelectedNodes();
        }

        treeControlKinematicTree.DeleteNode(node.TreeNode);

        // Invalidate cache
        componentToLinkCache = null;
        UpdateKinematicTabUI();
    }

    private void RemoveNodes(IEnumerable<NXLinkNode> nodes)
    {
        foreach (var node in nodes)
        {
            if (node.IsRootNode)
            {
                theUI.NXMessageBox.Show("Cannot Remove", NXMessageBox.DialogType.Warning,
                    "Cannot remove the base link.");
                return;
            }
        }

        foreach (var node in nodes)
        {
            try
            {
                treeControlKinematicTree.DeleteNode(node.TreeNode);
            }
            catch (NXException e)
            {
                // dirty way to delete nodes
            }
        }

        treeControlKinematicTree.SelectNode(treeControlKinematicTree.RootNode, true, true);

        // Invalidate cache
        componentToLinkCache = null;
        UpdateKinematicTabUI();
    }

    private void MoveNodeUp(NXLinkNode node)
    {
        var prevSibling = node.TreeNode.PreviousSiblingNode;
        if (prevSibling == null)
            return;

        // Move node before its previous sibling
        var targetNode = treeManager.GetNode(prevSibling);
        if (targetNode != null)
        {
            var movedNode = treeManager.ReparentNode(node, targetNode, Node.DropType.Before);

            // Re-select the moved node (use returned node since original was deleted)
            if (movedNode != null)
            {
                treeControlKinematicTree.SelectNode(movedNode.TreeNode, true, true);
                NXTreeManager.ExpandNodesRecursively(movedNode);
            }
        }
        UpdateKinematicTabUI();
    }

    private void MoveNodeDown(NXLinkNode node)
    {
        var nextSibling = node.TreeNode.NextSiblingNode;
        if (nextSibling == null)
            return;

        // Move node after its next sibling
        var targetNode = treeManager.GetNode(nextSibling);
        if (targetNode != null)
        {
            var movedNode = treeManager.ReparentNode(node, targetNode, Node.DropType.After);

            // Re-select the moved node (use returned node since original was deleted)
            if (movedNode != null)
            {
                treeControlKinematicTree.SelectNode(movedNode.TreeNode, true, true);
                NXTreeManager.ExpandNodesRecursively(movedNode);
            }
        }
        UpdateKinematicTabUI();
    }

    private void AddNewRoot(NXLinkNode node)
    {
        if (!node.IsRootNode)
        {
            return;
        }

        var newRootNode = treeManager.InsertRootNode(node);

        if (newRootNode != null)
        {
            treeControlKinematicTree.SelectNode(newRootNode.TreeNode, true, true);
            treeManager.UpdateSelectedNodes();
            NXTreeManager.ExpandNodesRecursively(newRootNode);
        }
        UpdateKinematicTabUI();
    }

    private void ConvertToRoot(NXLinkNode node)
    {
        if (node == null || node.IsRootNode)
            return;

        var currentRoot = treeManager.GetRootNode();
        if (currentRoot == null)
            return;

        int result = theUI.NXMessageBox.Show(
            "Convert to Root Link",
            NXMessageBox.DialogType.Question,
            "Convert \"" + node.LinkName + "\" to the root link? \"" + currentRoot.LinkName + "\" will become its child.");
        if (result != 1)
            return;

        var newRoot = treeManager.ReplaceRootNode(node, currentRoot);
        if (newRoot != null)
        {
            treeControlKinematicTree.SelectNode(newRoot.TreeNode, true, true);
            treeManager.UpdateSelectedNodes();
            NXTreeManager.ExpandNodesRecursively(newRoot);
        }
        UpdateKinematicTabUI();
    }

    //------------------------------------------------------------------------------
    // Data Persistence and Validation
    //------------------------------------------------------------------------------

    /// <summary>
    /// Validates the entire tree for completeness.
    /// </summary>
    private bool ValidateTree()
    {
        // Check for sites with children
        var sitesWithChildren = treeManager.GetSitesWithChildren();
        if (sitesWithChildren.Count > 0)
        {
            string names = string.Join(", ", sitesWithChildren.Select(n => n.LinkName));
            theUI.NXMessageBox.Show("Validation Error", NXMessageBox.DialogType.Warning,
                "Sites cannot have children. The following sites have child nodes: " + names);

            treeControlKinematicTree.SelectNode(sitesWithChildren[0].TreeNode, true, true);
            treeManager.UpdateSelectedNodes();
            return false;
        }

        // Check for duplicate link names
        if (!treeManager.CheckLinkNamesUnique(out var duplicateLinkNames))
        {
            string message = "Duplicate link names found: " + string.Join(", ", duplicateLinkNames);
            theUI.NXMessageBox.Show("Validation Error", NXMessageBox.DialogType.Warning, message);
            return false;
        }

        // Check for duplicate joint names
        if (!treeManager.CheckJointNamesUnique(out var duplicateJointNames))
        {
            string message = "Duplicate joint names found: " + string.Join(", ", duplicateJointNames);
            theUI.NXMessageBox.Show("Validation Error", NXMessageBox.DialogType.Warning, message);
            return false;
        }

        // Check for incomplete nodes
        var incompleteNodes = treeManager.GetIncompleteNodes();
        if (incompleteNodes.Count > 0)
        {
            string message = "The following links are incomplete:\n";
            foreach (var node in incompleteNodes)
            {
                node.CheckIncomplete(out string reason);
                message += $"- {node.LinkName}: {reason}\n";
            }
            theUI.NXMessageBox.Show("Validation Error", NXMessageBox.DialogType.Warning, message);

            // Select the first incomplete node
            var firstIncomplete = incompleteNodes[0];
            treeControlKinematicTree.SelectNode(firstIncomplete.TreeNode, true, true);
            treeManager.UpdateSelectedNodes();

            return false;
        }

        return true;
    }

    //------------------------------------------------------------------------------
    // Helper Methods
    //------------------------------------------------------------------------------

    private void SetBodySelectionMask(LinkSelectionMode mode)
    {
        Selection.MaskTriple convergentBodySelectionMask;
        convergentBodySelectionMask.Type = UFConstants.UF_solid_type;
        convergentBodySelectionMask.Subtype = UFConstants.UF_solid_body_subtype;
        convergentBodySelectionMask.SolidBodySubtype = UFConstants.UF_UI_SEL_FEATURE_CONVERGENT_SOLID_BODY;

        Selection.MaskTriple bodySelectionMask;
        bodySelectionMask.Type = UFConstants.UF_solid_type;
        bodySelectionMask.Subtype = UFConstants.UF_solid_body_subtype;
        bodySelectionMask.SolidBodySubtype = UFConstants.UF_UI_SEL_FEATURE_SOLID_BODY;

        Selection.MaskTriple componentSelectionMask;
        componentSelectionMask.Type = UFConstants.UF_component_type;
        componentSelectionMask.Subtype = UFConstants.UF_component_subtype;
        componentSelectionMask.SolidBodySubtype = 0;

        List<Selection.MaskTriple> selectionMasks = new List<Selection.MaskTriple>();

        switch (mode)
        {
            case LinkSelectionMode.Both:
                selectionMasks.Add(convergentBodySelectionMask);
                selectionMasks.Add(bodySelectionMask);
                selectionMasks.Add(componentSelectionMask);
                break;
            case LinkSelectionMode.Bodies:
                selectionMasks.Add(convergentBodySelectionMask);
                selectionMasks.Add(bodySelectionMask);
                break;
            case LinkSelectionMode.Components:
                selectionMasks.Add(componentSelectionMask);
                break;
        }

        Selection.MaskTriple[] selectionMasksArray = selectionMasks.ToArray();

        bodySelectInertial.SetSelectionFilter(Selection.SelectionAction.ClearAndEnableSpecific, selectionMasksArray);
        bodySelectCollision.SetSelectionFilter(Selection.SelectionAction.ClearAndEnableSpecific, selectionMasksArray);
        bodySelectVisual.SetSelectionFilter(Selection.SelectionAction.ClearAndEnableSpecific, selectionMasksArray);
    }

    private static string GetJointTypeDisplayText(string jointType)
    {
        if (string.IsNullOrEmpty(jointType))
            return "Fixed";

        switch (jointType.ToLower())
        {
            case "revolute": return "Revolute";
            case "continuous": return "Continuous";
            case "prismatic": return "Prismatic";
            case "fixed": return "Fixed";
            case "floating": return "Floating";
            case "planar": return "Planar";
            default: return "Fixed";
        }
    }

    private static string GetJointTypeFromDisplayText(string displayText)
    {
        if (string.IsNullOrEmpty(displayText))
            return "fixed";

        switch (displayText)
        {
            case "Revolute": return "revolute";
            case "Continuous": return "continuous";
            case "Prismatic": return "prismatic";
            case "Fixed": return "fixed";
            case "Floating": return "floating";
            case "Planar": return "planar";
            default: return "fixed";
        }
    }

    private static string GetJointAxisFromCsysDisplayText(string axis)
    {
        if (string.IsNullOrEmpty(axis))
        {
            return "No, Use selection";
        }

        switch (axis)
        {
            case "x": return "X Axis";
            case "y": return "Y Axis";
            case "z": return "Z Axis";
            case "none":
            default:
                return "No, Use selection";
        }
    }

    private static string GetJointAxisFromCsysFromDisplayText(string displayText)
    {
        if (string.IsNullOrEmpty(displayText))
        {
            return "none";
        }

        switch (displayText)
        {
            case "X Axis": return "x";
            case "Y Axis": return "y";
            case "Z Axis": return "z";
            case "No, Use selection":
            default:
                return "none";
        }
    }

    private static LinkSelectionMode GetLinkSelectModeFromDisplayText(string displayText)
    {
        if (string.IsNullOrEmpty(displayText))
        {
            return LinkSelectionMode.Both;
        }

        switch (displayText)
        {
            case "Only Bodies": return LinkSelectionMode.Bodies;
            case "Only Components": return LinkSelectionMode.Components;
            default:
            case "Both Bodies/Components": return LinkSelectionMode.Both;
        }
    }

    /// <summary>
    /// Retrieves Bodies and Components from their persistent keys for UI display.
    /// Returns objects as-is (Components are not expanded to bodies).
    /// </summary>
    private TaggedObject[] GetObjectsFromHandles(List<string> handles)
    {
        if (handles == null || handles.Count == 0)
            return Array.Empty<TaggedObject>();

        Part workPart = theSession.Parts.Work;
        return NXPersistentId.ResolveKeysToObjectsForUI(workPart, handles).ToArray();
    }

    /// <summary>
    /// Retrieves a CartesianCoordinateSystem from its persistent GUID.
    /// </summary>
    private CartesianCoordinateSystem GetCoordinateSystemFromGuid(string guid)
    {
        if (string.IsNullOrEmpty(guid))
            return null;

        Part workPart = theSession.Parts.Work;
        return NXPersistentId.FindCoordinateSystemByKey(workPart, guid);
    }

    /// <summary>
    /// Retrieves a DatumAxis from its persistent GUID.
    /// </summary>
    private DatumAxis GetAxisFromGuid(string guid)
    {
        if (string.IsNullOrEmpty(guid))
            return null;

        Part workPart = theSession.Parts.Work;
        return NXPersistentId.FindAxisByKey(workPart, guid);
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
            theUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
        }
        return plist;
    }

    private void UpdateQuickSettings()
    {
        Part workPart = theSession.Parts.Work;

        workPart.Preferences.ScreenVisualization.CsysShowThrough = toggleShowThroughCSYS.Value;
        workPart.Preferences.ScreenVisualization.CurveShowThrough = toggleShowThroughCurves.Value;
        workPart.Preferences.ScreenVisualization.PointShowThrough = toggleShowThroughPoints.Value;
    }

    /// <summary>
    /// Shows a warning when the user selects geometry from a component part they may not own.
    /// </summary>
    private void ShowOwnershipWarning()
    {
        int result = theUI.NXMessageBox.Show("Read-Only Warning", NXMessageBox.DialogType.Question,
            "You've selected an object from a part you don't have write-access to.\n\n" +
            "Storing selection data requires editing that part. " +
            "Consider changing the Selection Behavior to create WAVE Links.\n\n\n" +
            "Would you like to suppress this warning for the rest of this session?");

        if (result == 1)
        {
            suppressOwnershipWarning = true;
        }
    }

    /// <summary>
    /// Shows a warning specific to body selections from read-only parts.
    /// Automatic WAVE linking is not available for bodies.
    /// </summary>
    private void ShowBodyOwnershipWarning()
    {
        int result = theUI.NXMessageBox.Show("Read-Only Warning", NXMessageBox.DialogType.Question,
            "You've selected a body from a part you don't have write-access to.\n" +
            "Storing selection data requires editing that part.\n\n" +
            "Prefer selecting entire Components rather than individual bodies " +
            "(change 'Link Selection Mode' to 'Only Components').\n\n\n" +
            "Would you like to suppress this warning for the rest of this session?");

        if (result == 1)
        {
            suppressBodyOwnershipWarning = true;
        }
    }

    private bool ConfirmDeselection(string fieldLabel)
    {
        int result = theUI.NXMessageBox.Show("Confirm Clear", NXMessageBox.DialogType.Question,
            "Are you sure you want to clear the " + fieldLabel + " selection?");
        return result == 1;
    }

    //------------------------------------------------------------------------------
    // Help System
    //------------------------------------------------------------------------------

    /// <summary>
    /// Initializes the help system by loading the help map file and pushing the context.
    /// When user presses F1 or clicks the ? button, the URL from the map file is opened.
    /// The map file (RobotExporterHelp.map) should be in the /startup folder alongside the the built dll
    /// </summary>
    private void InitializeHelpSystem()
    {
        try
        {
            string appDir = System.IO.Path.GetDirectoryName(
                System.Reflection.Assembly.GetExecutingAssembly().Location);
            string mapFilePath = System.IO.Path.Combine(appDir, HelpMapFileName);

            if (!System.IO.File.Exists(mapFilePath))
            {
                logger.Warning($"Help map file not found: {mapFilePath}");
                return;
            }

            // Load the help map file
            ufSession.Help.LoadMapFile(mapFilePath);

            // Push the help context - this sets what happens when F1 is pressed
            ufSession.Help.PushContext(HelpContextMain);
            helpContextPushed = true;

            logger.Information($"Help system initialized with map file: {mapFilePath}");
        }
        catch (Exception ex)
        {
            logger.Warning($"Failed to initialize help system: {ex.Message}");
        }
    }

    /// <summary>
    /// Cleans up the help system by popping the context.
    /// This reverts to the previous help context (main NX help).
    /// </summary>
    private void CleanupHelpSystem()
    {
        try
        {
            if (helpContextPushed)
            {
                ufSession.Help.PopContext();
                helpContextPushed = false;
                logger.Information("Help system context popped");
            }
        }
        catch (Exception ex)
        {
            logger.Warning($"Failed to cleanup help system: {ex.Message}");
        }
    }
}
