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
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;
using SolidWorks.Interop.swpublished;

using CADRobotExporter.UI;
using CADRobotExporter.CAD;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.Utilities;

namespace CADRobotExporter.SW
{
    // Adding a new line
    //
    /// <summary>
    /// Summary description for CADRobotExporter.
    /// </summary>
    [Guid("65c9fc17-6a74-45a3-8f84-55185900275d"), ComVisible(true)]
    public class SwAddin : ISwAddin
    {
        #region Static Variables

        private static readonly Serilog.ILogger logger = Logger.GetLogger();

        public static string LatestExportPath { get; private set; }

        #endregion Static Variables

        #region Local Variables

        private int addInID = 0;

        public const int mainCmdGroupID = 5;
        public const int itemIDCreateURDF = 0;
        public const int itemIDEditURDF = 1;
        public const int itemIDBackupConfiguration = 2;
        public const int itemIDImportConfiguration = 3;
        public const int itemIDDuplicateConfiguration = 4;
        public const int itemIDImportURDF = 5;

        string[] icons = new string[6];
        private int cmdIdxCreateURDF;
        private int cmdIdxEditURDF;
        private int cmdIdxBackupConfiuguration;
        private int cmdIdxImportConfiguration;
        private int cmdIdxDuplicateConfiguration;
        private int cmdIdxImportURDF;

        #region Event Handler Variables

        private SldWorks SwEventPtr = null;

        #endregion Event Handler Variables

        // Public Properties
        public ISldWorks SwApp { get; private set; } = null;

        public ICommandManager iCmdMgr { get; private set; } = null;

        public Hashtable OpenDocs { get; private set; } = new Hashtable();

        private ExportPropertyManager pm;

        #endregion Local Variables

        #region SolidWorks Registration

        private const string AddInTitle = "SuperDex CAD Exporter";
        private const string AddInDescription = "Exports robots from SolidWorks to .superdex_bot, URDF and MJCF";
        private const bool AddInLoadAtStartup = true;

        [ComRegisterFunction]
        public static void RegisterFunction(Type t)
        {
            try
            {
                Microsoft.Win32.RegistryKey hklm = Microsoft.Win32.Registry.LocalMachine;
                Microsoft.Win32.RegistryKey hkcu = Microsoft.Win32.Registry.CurrentUser;

                string keyname = "SOFTWARE\\SolidWorks\\Addins\\{" + t.GUID.ToString() + "}";
                logger.Information("Registering " + keyname);
                Microsoft.Win32.RegistryKey addinkey = hklm.CreateSubKey(keyname);
                addinkey.SetValue(null, 0);

                addinkey.SetValue("Description", AddInDescription);
                addinkey.SetValue("Title", AddInTitle);

                keyname = "Software\\SolidWorks\\AddInsStartup\\{" + t.GUID.ToString() + "}";
                logger.Information("Registering " + keyname);
                addinkey = hkcu.CreateSubKey(keyname);
                addinkey.SetValue(
                    null, Convert.ToInt32(AddInLoadAtStartup), Microsoft.Win32.RegistryValueKind.DWord);
            }
            catch (Exception e)
            {
                logger.Error(e.Message);
            }
        }

        [ComUnregisterFunction]
        public static void UnregisterFunction(Type t)
        {
            try
            {
                Microsoft.Win32.RegistryKey hklm = Microsoft.Win32.Registry.LocalMachine;
                Microsoft.Win32.RegistryKey hkcu = Microsoft.Win32.Registry.CurrentUser;

                string keyname = "SOFTWARE\\SolidWorks\\Addins\\{" + t.GUID.ToString() + "}";
                logger.Information("Unregistering " + keyname);
                hklm.DeleteSubKey(keyname, false);

                keyname = "Software\\SolidWorks\\AddInsStartup\\{" + t.GUID.ToString() + "}";
                logger.Information("Unregistering " + keyname);
                hkcu.DeleteSubKey(keyname, false);
            }
            catch (NullReferenceException nl)
            {
                logger.Error("There was a problem unregistering this dll: " + nl.Message);
                MessageBox.Show("There was a problem unregistering this dll: \n\"" +
                    nl.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
            catch (Exception e)
            {
                logger.Error("There was a problem unregistering this dll: " + e.Message);
                MessageBox.Show("There was a problem unregistering this dll: \n\"" +
                    e.Message + "\"\nContact your maintainer with the log file found at " +
                    Logger.GetLogFolder());
            }
        }

        #endregion SolidWorks Registration

        #region ISwAddin Implementation

        public SwAddin()
        {
            Logger.Setup();
            LatestExportPath = System.Environment.ExpandEnvironmentVariables("%HOMEDRIVE%%HOMEPATH%");
        }

        public static void SetLatestExportPath(string path)
        {
            LatestExportPath = path;
        }

        private void ExceptionHandler(object sender, ThreadExceptionEventArgs e)
        {
            logger.Warning("Exception encountered in Assembly export form", e.Exception);
        }

        private void UnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            logger.Error("Unhandled exception in Assembly Export form\nContact your maintainer " +
                "with the log file found at " +
                Logger.GetLogFolder(), (Exception)e.ExceptionObject);
        }

        public bool ConnectToSW(object ThisSW, int cookie)
        {
            logger.Information("Attempting to connect to SW");
            SwApp = (ISldWorks)ThisSW;
            addInID = cookie;

            //Setup callbacks
            logger.Information("Setting up callbacks");
            SwApp.SetAddinCallbackInfo(0, this, addInID);

            logger.Information("Setting up command manager");
            iCmdMgr = SwApp.GetCommandManager(cookie);

            logger.Information("Adding command manager");
            AddCommandMgr();

            logger.Information("Adding event handlers");
            SwEventPtr = (SldWorks)SwApp;
            OpenDocs = new Hashtable();
            AttachEventHandlers();

            ConfigurationSerialization.RegisterUrdfAttribute((SldWorks)SwApp);

            logger.Information("Linking superdex_mesh_cli Save As... menu");

            SwApp.AddFileSaveAsItem2(addInID, "OCC_FileSave", "superdex_mesh_cli file (*.glb/*.obj/*.stl)", "msh", (int)swDocumentTypes_e.swDocPART);
            SwApp.AddFileSaveAsItem2(addInID, "OCC_FileSave", "superdex_mesh_cli file (*.glb/*.obj/*.stl)", "msh", (int)swDocumentTypes_e.swDocASSEMBLY);

            return true;
        }

        public void OCC_FileSave(string sFileName)
        {
            // Example sFileName:
            // C:\Users\bar\Documents\foo robot\bar baz.OCC bar bazOCC w

            MeshSaveForm form = new MeshSaveForm((SldWorks)SwApp, sFileName);

            form.Show();
        }

        public bool CompareIDs(int[] storedIDs, int[] addinIDs)
        {
            List<int> storedList = new List<int>(storedIDs);
            List<int> addinList = new List<int>(addinIDs);

            addinList.Sort();
            storedList.Sort();

            if (addinList.Count != storedList.Count)
            {
                return false;
            }
            else
            {

                for (int i = 0; i < addinList.Count; i++)
                {
                    if (addinList[i] != storedList[i])
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        public bool DisconnectFromSW()
        {
            BitmapHandler.Instance.CleanFiles();

            RemoveCommandMgr();
            DetachEventHandlers();

            SwApp.RemoveFileSaveAsItem2(addInID, "OCC_FileSave", "superdex_mesh_cli file (*.glb/*.obj/*.stl)", "msh", (int)swDocumentTypes_e.swDocPART);
            SwApp.RemoveFileSaveAsItem2(addInID, "OCC_FileSave", "superdex_mesh_cli file (*.glb/*.obj/*.stl)", "msh", (int)swDocumentTypes_e.swDocASSEMBLY);

            Marshal.ReleaseComObject(iCmdMgr);
            iCmdMgr = null;
            Marshal.ReleaseComObject(SwApp);
            SwApp = null;
            //The addin _must_ call GC.Collect() here in order to retrieve all managed code pointers
            GC.Collect();
            GC.WaitForPendingFinalizers();

            GC.Collect();
            GC.WaitForPendingFinalizers();

            logger.Information("Disconnecting plugin from SolidWorks");
            return true;
        }

        #endregion ISwAddin Implementation

        #region UI Methods

        public void AddCommandMgr()
        {
            Assembly thisAssembly;
            thisAssembly = System.Reflection.Assembly.GetAssembly(this.GetType());

            icons[0] = BitmapHandler.Instance.CreateFileFromResourceBitmap("CADRobotExporter.Icons.main_20px.png", thisAssembly);
            icons[1] = BitmapHandler.Instance.CreateFileFromResourceBitmap("CADRobotExporter.Icons.main_32px.png", thisAssembly);
            icons[2] = BitmapHandler.Instance.CreateFileFromResourceBitmap("CADRobotExporter.Icons.main_40px.png", thisAssembly);
            icons[3] = BitmapHandler.Instance.CreateFileFromResourceBitmap("CADRobotExporter.Icons.main_64px.png", thisAssembly);
            icons[4] = BitmapHandler.Instance.CreateFileFromResourceBitmap("CADRobotExporter.Icons.main_96px.png", thisAssembly);
            icons[5] = BitmapHandler.Instance.CreateFileFromResourceBitmap("CADRobotExporter.Icons.main_128px.png", thisAssembly);

            int cmdGroupErr = 0;
            bool ignorePrevious = false;

            object registryIDs;
            //get the ID information stored in the registry
            bool getDataResult = iCmdMgr.GetGroupDataFromRegistry(mainCmdGroupID, out registryIDs);

            int[] knownIDs = new int[4] { itemIDCreateURDF, itemIDEditURDF, itemIDImportConfiguration, itemIDBackupConfiguration };

            if (getDataResult)
            {
                if (!CompareIDs((int[])registryIDs, knownIDs)) //if the IDs don't match, reset the commandGroup
                {
                    ignorePrevious = true;
                }
            }

            ICommandGroup cmdGroup;

            cmdGroup = iCmdMgr.CreateCommandGroup2(mainCmdGroupID, "Robotics", "SuperDex CAD Exporter and related tools", "", -1, ignorePrevious, ref cmdGroupErr);

            cmdGroup.MainIconList = icons;
            cmdGroup.IconList = icons;
            cmdGroup.ShowInDocumentType = (int)swDocTemplateTypes_e.swDocTemplateTypeASSEMBLY;

            int menuToolbarOption = (int)(swCommandItemType_e.swMenuItem | swCommandItemType_e.swToolbarItem);
            cmdIdxCreateURDF = cmdGroup.AddCommandItem2(
                "New Robot Configuration",
                -1,
                "Create a new Robot configuration separate from any existing ones.",
                "New Robot Configuration",
                0,
                "CreateNewURDF",
                "",
                itemIDCreateURDF,
                menuToolbarOption);
            cmdIdxEditURDF = cmdGroup.AddCommandItem2(
                "Edit and Export Robot Configuration",
                -1,
                "Edit the selected Robot Configuration and go through the export process.",
                "Edit and Export Robot Configuration",
                1,
                "EditAndExportURDF",
                "",
                itemIDEditURDF,
                menuToolbarOption);
            cmdIdxBackupConfiuguration = cmdGroup.AddCommandItem2(
                "Back-up configuration",
                -1,
                "Exports selected configuration as an .xml and .json file for import into other models or safekeeping.",
                "Back-up configuration",
                2,
                "ExportURDFConfiguration",
                "",
                itemIDBackupConfiguration,
                menuToolbarOption);
            cmdIdxImportConfiguration = cmdGroup.AddCommandItem2(
                "Import configuration",
                -1,
                "Imports backed-up configurations (.xml and .json) file into the current assembly. Note that this does not import .urdf files.",
                "Import configuration",
                3,
                "ImportURDFConfiguration",
                "",
                itemIDImportConfiguration,
                menuToolbarOption);
            cmdIdxDuplicateConfiguration = cmdGroup.AddCommandItem2(
                "Duplicate configuration",
                -1,
                "Duplicates the currently selected configuration into a new one.",
                "Duplicate configuration",
                4,
                "DuplicateURDFConfiguration",
                "",
                itemIDDuplicateConfiguration,
                menuToolbarOption);
            cmdIdxImportURDF = cmdGroup.AddCommandItem2(
                "Import URDF",
                -1,
                "Imports a URDF file and creates coordinate systems at joint origins.",
                "Import URDF",
                5,
                "ImportURDF",
                "",
                itemIDImportURDF,
                menuToolbarOption);

            cmdGroup.HasToolbar = true;
            cmdGroup.HasMenu = true;

            bool cmdMgrResult = cmdGroup.Activate();

            CommandTab cmdTab;
            cmdTab = iCmdMgr.GetCommandTab((int)swDocumentTypes_e.swDocASSEMBLY, "Robotics");

            if (cmdTab != null & !getDataResult | ignorePrevious)
            {
                bool res = iCmdMgr.RemoveCommandTab(cmdTab);
                cmdTab = null;
            }

            if (cmdTab == null)
            {
                cmdTab = iCmdMgr.AddCommandTab((int)swDocumentTypes_e.swDocASSEMBLY, "Robotics");

                // Main buttons
                CommandTabBox MainCmdTabBox = cmdTab.AddCommandTabBox();

                List<int> MainCmdIDs = new List<int>();
                List<int> textTypes = new List<int>();

                MainCmdIDs.Add(cmdGroup.get_CommandID(cmdIdxCreateURDF));
                textTypes.Add((int)swCommandTabButtonTextDisplay_e.swCommandTabButton_TextBelow);

                MainCmdIDs.Add(cmdGroup.get_CommandID(cmdIdxEditURDF));
                textTypes.Add((int)swCommandTabButtonTextDisplay_e.swCommandTabButton_TextBelow);

                cmdMgrResult = MainCmdTabBox.AddCommands(MainCmdIDs.ToArray(), textTypes.ToArray());

                // Utility buttons
                CommandTabBox ExtraCmdTabBox = cmdTab.AddCommandTabBox();

                List<int> ExtraCmdIDs = new List<int>();
                List<int> textTypes1 = new List<int>();

                ExtraCmdIDs.Add(cmdGroup.get_CommandID(cmdIdxBackupConfiuguration));
                textTypes1.Add((int)swCommandTabButtonTextDisplay_e.swCommandTabButton_TextHorizontal);

                ExtraCmdIDs.Add(cmdGroup.get_CommandID(cmdIdxImportConfiguration));
                textTypes1.Add((int)swCommandTabButtonTextDisplay_e.swCommandTabButton_TextHorizontal);

                ExtraCmdIDs.Add(cmdGroup.get_CommandID(cmdIdxDuplicateConfiguration));
                textTypes1.Add((int)swCommandTabButtonTextDisplay_e.swCommandTabButton_TextHorizontal);

                cmdMgrResult = ExtraCmdTabBox.AddCommands(ExtraCmdIDs.ToArray(), textTypes1.ToArray());

                cmdTab.AddSeparator(ExtraCmdTabBox, ExtraCmdIDs[0]);
            }
        }

        public void RemoveCommandMgr()
        {
            logger.Information("Removing assembly export from menus");

            iCmdMgr.RemoveCommandGroup(mainCmdGroupID);
        }

        #endregion UI Methods

        #region UI Callbacks

        public void CreateNewURDF()
        {
            if (CheckExportFormOpen())
            {
                return;
            }

            try
            {
                pm = new ExportPropertyManager((SldWorks)SwApp);
                pm.LoadConfigTree(null);
                pm.Show();
            }
            catch (Exception e)
            {
                MessageBox.Show("There was an error creating a new Exporter Configuration.\n\n" +
                    e.ToString());
            }
        }

        public void EditAndExportURDF()
        {
            if (CheckExportFormOpen())
            {
                return;
            }

            LinkNode rootNode = null;
            bool error = false;
            Feature exporterFeature = null;

            try
            {
                rootNode = ConfigurationSerialization.LoadBaseNodeFromSelection(SwApp.ActiveDoc, out error, out exporterFeature);
            }
            catch (Exception e)
            {
                MessageBox.Show("There was an error trying to load the selected Exporter Configuration.\n\n" +
                    e.ToString());
                return;
            }

            if (rootNode == null || error)
            {
                return;
            }

            try
            {
                pm = new ExportPropertyManager((SldWorks)SwApp);
                ((SolidworksBridge)pm.Exporter.CadBridge).ExporterFeature = exporterFeature;
                pm.LoadConfigTree(rootNode);

                // Load saved tendons
                var bridge = (SolidworksBridge)pm.Exporter.CadBridge;
                List<Tendon> loadedTendons = bridge.LoadTendons();
                pm.PopulateTendonsFromList(loadedTendons);

                pm.Show();
            }
            catch (Exception e)
            {
                MessageBox.Show("There was an error trying to create the Property Manager page.\n\n" +
                    e.ToString());
            }
        }

        public void ExportURDFConfiguration()
        {
            if (!ConfigurationSerialization.LoadRawStringDataFromSelection(
                (SldWorks)SwApp,
                out string robotName,
                out string urdfConfiguration,
                out string exporterConfiguration,
                out string tendonData)) {
                return;
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
                    return;
                string basePath = folderDialog.SelectedPath;
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

                string xmlPath = Path.Combine(basePath, $"{robotName}.{timestamp}.urdfConfiguration.xml");
                File.WriteAllText(xmlPath, urdfConfiguration);
                string jsonPath = Path.Combine(basePath, $"{robotName}.{timestamp}.exporterConfiguration.json");
                File.WriteAllText(jsonPath, exporterConfiguration);

                if (!string.IsNullOrEmpty(tendonData))
                {
                    string tendonPath = Path.Combine(basePath, $"{robotName}.{timestamp}.tendons.xml");
                    File.WriteAllText(tendonPath, tendonData);
                }
            }
        }

        public void ImportURDFConfiguration()
        {
            using (var openDialog = new OpenFileDialog())
            {
                openDialog.Title = "Select Robot Configuration XML";
                openDialog.Filter = "XML files (*.xml)|*.xml";

                if (openDialog.ShowDialog() != DialogResult.OK)
                    return;
                string xmlPath = openDialog.FileName;
                string directory = Path.GetDirectoryName(xmlPath);
                string fileName = Path.GetFileName(xmlPath);

                if (!fileName.EndsWith(".urdfConfiguration.xml"))
                {
                    MessageBox.Show("Invalid file. Expected *.urdfConfiguration.xml");
                    return;
                }
                string baseName = fileName.Replace(".urdfConfiguration.xml", "");
                string jsonPath = Path.Combine(directory, $"{baseName}.exporterConfiguration.json");
                string tendonPath = Path.Combine(directory, $"{baseName}.tendons.xml");
                string urdfConfiguration = File.ReadAllText(xmlPath);
                string exporterConfiguration = "";
                string tendonData = "";
                if (File.Exists(jsonPath))
                {
                    exporterConfiguration = File.ReadAllText(jsonPath);
                }
                else
                {
                    MessageBox.Show(
                        $"Companion file not found: {baseName}.exporterConfiguration.json\n\nProceeding without exporter configuration.",
                        "Warning",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning);
                }
                if (File.Exists(tendonPath))
                {
                    tendonData = File.ReadAllText(tendonPath);
                }
                string robotName = baseName.Split('.')[0];
                ConfigurationSerialization.CreateNewExporterFeatureFromRawData(
                    (SldWorks)SwApp,
                    robotName,
                    urdfConfiguration,
                    exporterConfiguration,
                    tendonData);
            }
        }

        public void DuplicateURDFConfiguration()
        {
            if (!ConfigurationSerialization.LoadRawStringDataFromSelection(
                (SldWorks)SwApp,
                out string robotName,
                out string urdfConfiguration,
                out string exporterConfiguration,
                out string tendonData))
            {
                return;
            }

            if (string.IsNullOrEmpty(robotName))
            {
                robotName = "default";
            }

            ConfigurationSerialization.CreateNewExporterFeatureFromRawData((SldWorks)SwApp, robotName, urdfConfiguration, exporterConfiguration, tendonData);
        }

        public void ImportURDF()
        {
            ModelDoc2 activeDoc = SwApp.ActiveDoc;
            if (activeDoc == null)
            {
                MessageBox.Show("Please open an assembly document first.", "Import URDF", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            if (activeDoc.GetType() != (int)swDocumentTypes_e.swDocASSEMBLY)
            {
                MessageBox.Show("URDF import is only supported for assembly documents.", "Import URDF", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            using (var form = new RobotImportForm((SldWorks)SwApp))
            {
                if (form.ShowDialog() != DialogResult.OK)
                    return;

                string urdfPath = form.UrdfFilePath;

                try
                {
                    var config = new Import.URDFImportConfiguration
                    {
                        UrdfFilePath = urdfPath,
                        CreateCoordinateSystems = form.CreateCoordinateSystems,
                        CreateRobotConfigurationFeature = form.CreateRobotConfiguration,
                        MeshBasePath = Path.GetDirectoryName(urdfPath),
                        BaseCoordinateSystemName = form.SelectedCoordinateSystem
                    };

                    var result = Import.URDFImporter.Import((ISldWorks)SwApp, activeDoc, config);

                    if (!result.Success)
                    {
                        MessageBox.Show($"Failed to import URDF file:\n\n{result.ErrorMessage}", "Import URDF", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }

                    int jointCount = result.CreatedCoordinateSystems.Count;
                    int totalLinks = CountLinks(result.RootLink);

                    string warningsText = "";
                    if (result.Warnings.Count > 0)
                    {
                        warningsText = $"\n\nWarnings:\n- " + string.Join("\n- ", result.Warnings);
                    }

                    MessageBox.Show(
                        $"URDF Import Complete!\n\n" +
                        $"Robot: {result.RobotName}\n" +
                        $"Links: {totalLinks}\n" +
                        $"Coordinate Systems Created: {jointCount}" +
                        warningsText,
                        "Import URDF",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Information);
                }
                catch (Exception ex)
                {
                    logger.Error($"URDF import failed: {ex.Message}", ex);
                    MessageBox.Show($"Failed to import URDF file:\n\n{ex.Message}", "Import URDF", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private int CountLinks(RobotDescription.Link link)
        {
            int count = 1;
            foreach (var child in link.Children)
            {
                count += CountLinks(child);
            }
            return count;
        }

        public bool CheckExportFormOpen()
        {
            if (AssemblyExportForm.FormIsOpen)
            {
                System.Windows.MessageBox.Show(
                    "There is already a SuperDex CAD Exporter window open.\n\nPlease close the window first before restarting the export process.",
                    "SuperDex CAD Exporter",
                    System.Windows.MessageBoxButton.OK,
                    System.Windows.MessageBoxImage.None,
                    System.Windows.MessageBoxResult.OK,
                    System.Windows.MessageBoxOptions.DefaultDesktopOnly);
                if (AssemblyExportForm.Instance != null)
                {
                    AssemblyExportForm.Instance.WindowState = FormWindowState.Normal;
                    AssemblyExportForm.Instance.Focus();
                    return true;
                }
            }

            return false;
        }

        #endregion UI Callbacks

        #region Event Methods

        public bool AttachEventHandlers()
        {
            AttachSwEvents();
            //Listen for events on all currently open docs
            AttachEventsToAllDocuments();
            return true;
        }

        private bool AttachSwEvents()
        {
            try
            {
                SwEventPtr.ActiveDocChangeNotify +=
                    new DSldWorksEvents_ActiveDocChangeNotifyEventHandler(OnDocChange);
                SwEventPtr.DocumentLoadNotify2 +=
                    new DSldWorksEvents_DocumentLoadNotify2EventHandler(OnDocLoad);
                SwEventPtr.FileNewNotify2 +=
                    new DSldWorksEvents_FileNewNotify2EventHandler(OnFileNew);
                SwEventPtr.ActiveModelDocChangeNotify +=
                    new DSldWorksEvents_ActiveModelDocChangeNotifyEventHandler(OnModelChange);
                SwEventPtr.FileOpenPostNotify +=
                    new DSldWorksEvents_FileOpenPostNotifyEventHandler(FileOpenPostNotify);
                return true;
            }
            catch (Exception e)
            {
                logger.Error("Attaching SW events failed", e);
                return false;
            }
        }

        private bool DetachSwEvents()
        {
            try
            {
                SwEventPtr.ActiveDocChangeNotify -=
                    new DSldWorksEvents_ActiveDocChangeNotifyEventHandler(OnDocChange);
                SwEventPtr.DocumentLoadNotify2 -=
                    new DSldWorksEvents_DocumentLoadNotify2EventHandler(OnDocLoad);
                SwEventPtr.FileNewNotify2 -=
                    new DSldWorksEvents_FileNewNotify2EventHandler(OnFileNew);
                SwEventPtr.ActiveModelDocChangeNotify -=
                    new DSldWorksEvents_ActiveModelDocChangeNotifyEventHandler(OnModelChange);
                SwEventPtr.FileOpenPostNotify -=
                    new DSldWorksEvents_FileOpenPostNotifyEventHandler(FileOpenPostNotify);
                return true;
            }
            catch (Exception e)
            {
                logger.Error("Attaching SW events failed", e);
                return false;
            }
        }

        public void AttachEventsToAllDocuments()
        {
            ModelDoc2 modDoc = (ModelDoc2)SwApp.GetFirstDocument();
            while (modDoc != null)
            {
                if (!OpenDocs.Contains(modDoc))
                {
                    AttachModelDocEventHandler(modDoc);
                }
                else if (OpenDocs.Contains(modDoc))
                {
                    DocumentEventHandler docHandler = (DocumentEventHandler)OpenDocs[modDoc];
                    if (docHandler != null)
                    {
                        bool connected = docHandler.ConnectModelViews();
                        if (!connected)
                        {
                            logger.Warning("Failed to connect to model views");
                        }
                    }
                }

                modDoc = (ModelDoc2)modDoc.GetNext();
            }
        }

        public bool AttachModelDocEventHandler(ModelDoc2 modDoc)
        {
            if (modDoc == null)
            {
                return false;
            }

            if (!OpenDocs.Contains(modDoc))
            {
                DocumentEventHandler docHandler;
                switch (modDoc.GetType())
                {
                    case (int)swDocumentTypes_e.swDocPART:
                        {
                            docHandler = new PartEventHandler(modDoc, this);
                            break;
                        }
                    case (int)swDocumentTypes_e.swDocASSEMBLY:
                        {
                            docHandler = new AssemblyEventHandler(modDoc, this);
                            break;
                        }
                    case (int)swDocumentTypes_e.swDocDRAWING:
                        {
                            docHandler = new DrawingEventHandler(modDoc, this);
                            break;
                        }
                    default:
                        {
                            return false; //Unsupported document type
                        }
                }
                docHandler.AttachEventHandlers();
                OpenDocs.Add(modDoc, docHandler);
            }
            return true;
        }

        public bool DetachModelEventHandler(ModelDoc2 modDoc)
        {
            OpenDocs.Remove(modDoc);
            return true;
        }

        public bool DetachEventHandlers()
        {
            DetachSwEvents();

            //Close events on all currently open docs
            DocumentEventHandler docHandler;
            int numKeys = OpenDocs.Count;
            object[] keys = new Object[numKeys];

            //Remove all document event handlers
            OpenDocs.Keys.CopyTo(keys, 0);
            foreach (ModelDoc2 key in keys)
            {
                docHandler = (DocumentEventHandler)OpenDocs[key];
                docHandler.DetachEventHandlers(); //This also removes the pair from the hash
                docHandler = null;
            }
            return true;
        }

        #endregion Event Methods

        #region Event Handlers

        //Events
        public int OnDocChange()
        {
            return 0;
        }

        public int OnDocLoad(string docTitle, string docPath)
        {
            return 0;
        }

        private int FileOpenPostNotify(string FileName)
        {
            AttachEventsToAllDocuments();
            return 0;
        }

        public int OnFileNew(object newDoc, int docType, string templateName)
        {
            AttachEventsToAllDocuments();
            return 0;
        }

        public int OnModelChange()
        {
            return 0;
        }

        #endregion Event Handlers
    }
}

#endif
