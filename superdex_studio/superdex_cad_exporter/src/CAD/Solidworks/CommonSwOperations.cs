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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using Serilog;

using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

using CADRobotExporter.RobotDescription;
using CADRobotExporter.UI;
using CADRobotExporter.Utilities;

namespace CADRobotExporter.CAD
{
    public static class CommonSwOperations
    {
        private static readonly ILogger logger = Logger.GetLogger();

        // https://www.codestack.net/solidworks-api/document/assembly/components/show-with-dependents/
        [DllImport("User32.dll", EntryPoint = "SendMessageA", CharSet = CharSet.Ansi)]
        private static extern int SendMessage(IntPtr hWnd, int wMsg, int wParam, int lParam);
        private const int WM_COMMAND = 0x111;
        private const int SHOW_WITH_DEPENDENTS_CMD = 33227;

        private static void ShowWithDependents(ISldWorks swApp, Component2 comp)
        {
            comp.Select4(false, null, false);
            Frame swFrame = (Frame)swApp.Frame();
            IntPtr hWnd = new IntPtr(swFrame.GetHWnd());
            int _ = SendMessage(hWnd, WM_COMMAND, SHOW_WITH_DEPENDENTS_CMD, 0);
        }

        //Selects the components of a link. Helps highlight when the associated node is
        // selected from the tree
        public static void SelectComponents(ModelDoc2 model, Link Link, bool clearSelection, int mark = -1)
        {
            if (clearSelection)
            {
                model.ClearSelection2(true);
            }
            SelectionMgr manager = model.SelectionManager;
            SelectData data = manager.CreateSelectData();
            data.Mark = mark;
            SelectComponents(model, Link.SWVisualComponents, false);
            foreach (Link child in Link.Children)
            {
                SelectComponents(model, child, false, mark);
            }
        }

        //Selects components from a list.
        public static void SelectComponents(
            ModelDoc2 model, List<Component2> components, bool clearSelection = true, int mark = -1)
        {
            if (clearSelection)
            {
                model.ClearSelection2(true);
            }
            SelectionMgr manager = model.SelectionManager;
            SelectData data = manager.CreateSelectData();
            data.Mark = mark;
            foreach (Component2 component in components)
            {
                component.Select4(true, data, false);
            }
        }

        public static void DeselectByMark(ModelDoc2 model, int mark)
        {
            SelectionMgr manager = model.SelectionManager;
            int numObjects = manager.GetSelectedObjectCount2(mark);
            int[] objectIdx = new int[numObjects];
            for (int i = 0; i < numObjects; i++)
            {
                objectIdx[i] = i + 1;
            }
            manager.DeSelect2(objectIdx, mark);
        }

        //Finds the selected components and returns them, used when pulling the items from
        // the selection box because it would be too hard for SolidWorks to allow you to
        // access the selectionbox components directly.
        public static void GetSelectedComponents(
            ModelDoc2 model, List<Component2> Components, int Mark = -1)
        {
            SelectionMgr selectionManager = model.SelectionManager;
            Components.Clear();
            for (int i = 0; i < selectionManager.GetSelectedObjectCount2(Mark); i++)
            {
                object obj = selectionManager.GetSelectedObject6(i + 1, Mark);
                Component2 comp = obj as Component2;
                if (comp != null)
                {
                    Components.Add(comp);
                }
            }
        }

        public static bool SelectionHasFaceOrBody(ModelDoc2 model, int Mark = -1)
        {
            SelectionMgr selectionManager = model.SelectionManager;
            for (int i = 0; i < selectionManager.GetSelectedObjectCount2(Mark); i++)
            {
                int type = selectionManager.GetSelectedObjectType3(i + 1, Mark);
                if (type == (int)swSelectType_e.swSelFACES
                    || type == (int)swSelectType_e.swSelSURFACEBODIES
                    || type == (int)swSelectType_e.swSelSOLIDBODIES)
                {
                    return true;
                }
            }

            return false;
        }

        public static void SelectCoordinateSystem(ModelDoc2 model, string name, int mark = -1)
        {
            model.Extension.SelectByID2(
                name, "COORDSYS", 0, 0, 0, true, mark, null, 0);
        }

        public static void SelectAxis(ModelDoc2 model, string name, int mark = -1)
        {
            model.Extension.SelectByID2(
                name, "AXIS", 0, 0, 0, true, mark, null, 0);
        }

        public static string GetSelectedObjectName(ModelDoc2 model, int mark = -1)
        {
            SelectionMgr selectionManager = model.SelectionManager;
            Feature selectedObject = selectionManager.GetSelectedObject6(1, mark);
            return selectedObject.GetNameForSelection(out _);
        }

        //finds all the hidden components, which will be added to a new display state. Also
        // used when exporting STLs, so that hidden components remain hidden
        public static List<Component2> FindHiddenComponents(object[] varComp)
        {
            List<Component2> hiddenComp = new List<Component2>();
            foreach (object obj in varComp)
            {
                Component2 comp = (Component2)obj;
                if (comp.IsHidden(false))
                {
                    hiddenComp.Add(comp);
                }
            }
            return hiddenComp;
        }

        //Except for an exclusionary list, this shows all the components
        public static void ShowAllComponents(ModelDoc2 model, List<Component2> hiddenComponents)
        {
            AssemblyDoc assyDoc = (AssemblyDoc)model;
            List<Component2> componentsToShow = new List<Component2>();
            object[] varComps = assyDoc.GetComponents(false);
            foreach (Component2 comp in varComps)
            {
                componentsToShow.Add(comp);
            }
            ShowComponents(componentsToShow);
            HideComponents(hiddenComponents);
        }

        public static void HideAllComponents(ModelDoc2 model)
        {
            AssemblyDoc assyDoc = (AssemblyDoc)model;
            List<Component2> componentsToHide = new List<Component2>();
            object[] varComps = assyDoc.GetComponents(false);
            foreach (Component2 comp in varComps)
            {
                componentsToHide.Add(comp);
            }
            HideComponents(componentsToHide);
        }

        public static void ShowComponentsWithDependents(ISldWorks swApp, List<Component2> components)
        {
            foreach (var component in components)
            {
                if (component.GetChildren() != null)
                {
                    ShowWithDependents(swApp, component);
                }
                component.Visible = (int)swComponentVisibilityState_e.swComponentVisible;
            }
        }

        //Shows the components in the list. Useful for exporting STLs
        public static void ShowComponents(List<Component2> components)
        {
            foreach (var component in components)
            {
                component.Visible = (int)swComponentVisibilityState_e.swComponentVisible;
            }
        }

        //Hides the components from a list
        public static void HideComponents(List<Component2> components)
        {
            foreach (var component in components)
            {
                component.Visible = (int)swComponentVisibilityState_e.swComponentHidden;
            }
        }

        public static void RetrieveSWComponentPIDs(ModelDoc2 model, LinkNode node)
        {
            if (node.Link.SWVisualComponents != null)
            {
                node.Link.SWComponentPIDs = new List<byte[]>();
                foreach (IComponent2 comp in node.Link.SWVisualComponents)
                {
                    byte[] PID = model.Extension.GetPersistReference3(comp);
                    node.Link.SWComponentPIDs.Add(PID);
                }
            }
            if (node.Link.SWCollisionComponents != null)
            {
                node.Link.SWCollisionComponentPIDs = new List<byte[]>();
                foreach (IComponent2 comp in node.Link.SWCollisionComponents)
                {
                    byte[] PID = model.Extension.GetPersistReference3(comp);
                    node.Link.SWCollisionComponentPIDs.Add(PID);
                }
            }
            if (node.Link.SWInertialComponents != null)
            {
                node.Link.SWInertialComponentPIDs = new List<byte[]>();
                foreach (IComponent2 comp in node.Link.SWInertialComponents)
                {
                    byte[] PID = model.Extension.GetPersistReference3(comp);
                    node.Link.SWInertialComponentPIDs.Add(PID);
                }
            }

            if (node.Link.Joint.SWRefAxisFeature != null)
            {
                byte[] axisPID = model.Extension.GetPersistReference3(node.Link.Joint.SWRefAxisFeature);
                node.Link.Joint.AxisPID = axisPID;
            }

            if (node.Link.Joint.SWCoordinateSystemFeature != null)
            {
                byte[] coordinateSystemPID = model.Extension.GetPersistReference3(node.Link.Joint.SWCoordinateSystemFeature);
                node.Link.Joint.CoordinateSystemPID = coordinateSystemPID;
            }

            foreach (LinkNode child in node.Nodes)
            {
                RetrieveSWComponentPIDs(model, child);
            }
        }

        //Converts the SW component references to PIDs
        public static void SaveSWComponents(ModelDoc2 model, Link Link)
        {
            model.ClearSelection2(true);
            Link.SWComponentPIDs = SaveSWComponents(model, Link.SWVisualComponents);
            Link.SWCollisionComponentPIDs = SaveSWComponents(model, Link.SWCollisionComponents);
            Link.SWInertialComponentPIDs = SaveSWComponents(model, Link.SWInertialComponents);

            foreach (Link Child in Link.Children)
            {
                SaveSWComponents(model, Child);
            }
        }

        //Converts SW component references to PIDs
        public static List<byte[]> SaveSWComponents(ModelDoc2 model, List<Component2> components)
        {
            List<byte[]> PIDs = new List<byte[]>();
            foreach (Component2 component in components)
            {
                byte[] PID = SaveSWComponent(model, component);
                if (PID != null)
                {
                    PIDs.Add(PID);
                }
            }
            return PIDs;
        }

        public static byte[] SaveSWComponent(ModelDoc2 model, Component2 component)
        {
            if (component != null)
            {
                return model.Extension.GetPersistReference3(component);
            }
            return null;
        }

        // Converts the PIDs to actual references to the components and proceeds recursively
        // through the child nodes
        public static void LoadSWComponents(ModelDoc2 model, LinkNode node, List<string> problemLinks)
        {
            logger.Information("Loading SolidWorks components for " +
                node.Link.Name + " from " + model.GetPathName());

            node.Link.SWVisualComponents = LoadSWComponents(model, node.Link.SWComponentPIDs);
            node.Link.SWCollisionComponents = LoadSWComponents(model, node.Link.SWCollisionComponentPIDs);
            node.Link.SWInertialComponents = LoadSWComponents(model, node.Link.SWInertialComponentPIDs);
            LoadRefAxisAndCoordsys(model, node);
            if (node.Link.SWVisualComponents.Count != node.Link.SWComponentPIDs.Count
                || node.Link.SWCollisionComponents.Count != node.Link.SWCollisionComponentPIDs.Count
                || node.Link.SWInertialComponents.Count != node.Link.SWInertialComponentPIDs.Count)
            {
                problemLinks.Add(node.Link.Name);
                logger.Error("Link " + node.Link.Name + " did not fully load all components");
            }
            logger.Information("Loaded " + node.Link.SWVisualComponents.Count + " visual components for link " + node.Link.Name);
            logger.Information("Loaded " + node.Link.SWCollisionComponents.Count + " collision components for link " + node.Link.Name);
            logger.Information("Loaded " + node.Link.SWInertialComponents.Count + " inertial components for link " + node.Link.Name);

            foreach (LinkNode Child in node.Nodes)
            {
                LoadSWComponents(model, Child, problemLinks);
            }
        }

        public static bool LoadRefAxisAndCoordsys(ModelDoc2 model, LinkNode node)
        {
            bool success = true;

            string axisName = node.Link.Joint.AxisName;
            if (node.Link.Joint.AxisPID != null && !Joint.IsAxisFromCsys(axisName))
            {
                Feature refAxis = LoadSWFeature(model, node.Link.Joint.AxisPID);
                if (refAxis == null)
                {
                    string byteAsString = PIDToString(node.Link.Joint.AxisPID);
                    logger.Warning("RefAxis feature with PID " + byteAsString + " failed to load");
                    success = false;
                }
                else
                {
                    node.Link.Joint.SWRefAxisFeature = refAxis;
                    node.Link.Joint.AxisName = refAxis.GetNameForSelection(out _);
                }
            }
            else
            {
                if (!string.IsNullOrEmpty(axisName) && !Joint.IsAxisFromCsys(axisName))
                {
                    // This was probably saved with a previous vresion so we'll populate the field now
                    model.ClearSelection2(true);
                    bool selected =
                        model.Extension.SelectByID2(axisName, "AXIS", 0, 0, 0, false, 0, null, 0);
                    if (selected)
                    {
                        Feature refAxis = model.SelectionManager.GetSelectedObject6(1, -1);
                        node.Link.Joint.SWRefAxisFeature = refAxis;
                    }
                    else
                    {
                        MessageBox.Show("Couldn't find " + axisName + ", this reference will be missing, you will have to re-select this Axis.");
                        success = false;
                    }
                    model.ClearSelection2(true);
                }
            }

            if (node.Link.Joint.CoordinateSystemPID != null)
            {
                Feature coordSys = LoadSWFeature(model, node.Link.Joint.CoordinateSystemPID);
                if (coordSys == null)
                {
                    string byteAsString = PIDToString(node.Link.Joint.CoordinateSystemPID);
                    logger.Warning("CoordSys feature with PID " + byteAsString + " failed to load");
                    success = false;
                }
                else
                {
                    node.Link.Joint.SWCoordinateSystemFeature = coordSys;
                    node.Link.Joint.CoordinateSystemName = coordSys.GetNameForSelection(out _);
                }
            }
            else
            {
                string coordSysName = node.Link.Joint.CoordinateSystemName;
                if (!string.IsNullOrEmpty(coordSysName))
                {
                    // This was probably saved with a previous vresion so we'll populate the field now
                    model.ClearSelection2(true);
                    bool selected =
                        model.Extension.SelectByID2(coordSysName, "COORDSYS", 0, 0, 0, true, 0, null, 0);
                    if (selected)
                    {
                        Feature coordSys = model.SelectionManager.GetSelectedObject6(1, -1);
                        node.Link.Joint.SWCoordinateSystemFeature = coordSys;
                    }
                    else
                    {
                        MessageBox.Show("Couldn't find " + coordSysName + ", this reference will be missing, you will have to re-select this Coordinate System.");
                        success = false;
                    }
                    model.ClearSelection2(true);
                }
            }

            return success;
        }

        // Converts the PIDs to actual references to the components
        public static List<Component2> LoadSWComponents(ModelDoc2 model, List<byte[]> PIDs)
        {
            List<Component2> components = new List<Component2>();
            foreach (byte[] PID in PIDs)
            {
                string byteAsString = PIDToString(PID);
                logger.Information("Loading component with PID " + byteAsString);
                Component2 comp = LoadSWComponent(model, PID);
                if (comp == null)
                {
                    logger.Warning("Component with PID " + byteAsString + " failed to load");
                }
                else
                {
                    components.Add(comp);
                    logger.Information("Successfully loaded component " + comp.GetPathName());
                }
            }
            return components;
        }

        public static Feature LoadSWFeature(ModelDoc2 model, byte[] PID)
        {
            string byteAsString = PIDToString(PID);
            if (PID == null)
            {
                throw new System.Exception("PID " + byteAsString + " was null. Is the configuration corrupted?");
            }

            object obj = model.Extension.GetObjectByPersistReference3(PID, out int Errors);
            if (Errors == 0)
            {
                return (Feature)obj;
            }
            switch ((swPersistReferencedObjectStates_e)Errors)
            {
                case swPersistReferencedObjectStates_e.swPersistReferencedObject_Deleted:
                    logger.Error("The feature associated with PID " + byteAsString + " was deleted");
                    break;

                case swPersistReferencedObjectStates_e.swPersistReferencedObject_Invalid:
                    logger.Error("The feature associated with PID " + byteAsString + " was found to be invalid");
                    break;

                case swPersistReferencedObjectStates_e.swPersistReferencedObject_Suppressed:
                    logger.Error("The feature associated with PID " + byteAsString + " is suppressed");
                    break;

                case swPersistReferencedObjectStates_e.swPersistReferencedObject_Ok:
                    break;

                default:
                    logger.Error("The feature associated with PID " + byteAsString +
                        " was not loaded due to an unspecified error (" + Errors + ")");
                    break;
            }
            return null;
        }

        // Converts a single PID to a Component2 object
        public static Component2 LoadSWComponent(ModelDoc2 model, byte[] PID)
        {
            string byteAsString = PIDToString(PID);
            if (PID == null)
            {
                throw new System.Exception("PID " + byteAsString + " was null. Is the configuration corrupted?");
            }

            object obj = model.Extension.GetObjectByPersistReference3(PID, out int Errors);
            if (Errors == 0)
            {
                return (Component2)obj;
            }
            switch ((swPersistReferencedObjectStates_e)Errors)
            {
                case swPersistReferencedObjectStates_e.swPersistReferencedObject_Deleted:
                    logger.Error("The component associated with PID " + byteAsString + " was deleted");
                    break;

                case swPersistReferencedObjectStates_e.swPersistReferencedObject_Invalid:
                    logger.Error("The component associated with PID " + byteAsString + " was found to be invalid");
                    break;

                case swPersistReferencedObjectStates_e.swPersistReferencedObject_Suppressed:
                    logger.Error("The component associated with PID " + byteAsString + " is suppressed");
                    break;

                case swPersistReferencedObjectStates_e.swPersistReferencedObject_Ok:
                    break;

                default:
                    logger.Error("The component associated with PID " + byteAsString +
                        " was not loaded due to an unspecified error (" + Errors + ")");
                    break;
            }
            return null;
        }

        public static string PIDToString(byte[] pid)
        {
            return Encoding.ASCII.GetString(pid);
        }

        public static Matrix GetRotationMatrix(MathTransform transform)
        {
            Matrix rot = new DenseMatrix(3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    rot.At(i, j, transform.ArrayData[i + 3 * j]);
                }
            }

            return rot;
        }

        public static Matrix<double> GetTransformation(MathTransform transform)
        {
            Matrix<double> m = new DenseMatrix(4);

            m[0, 0] = transform.ArrayData[0];
            m[1, 0] = transform.ArrayData[1];
            m[2, 0] = transform.ArrayData[2];
            m[0, 1] = transform.ArrayData[3];
            m[1, 1] = transform.ArrayData[4];
            m[2, 1] = transform.ArrayData[5];
            m[0, 2] = transform.ArrayData[6];
            m[1, 2] = transform.ArrayData[7];
            m[2, 2] = transform.ArrayData[8];

            m[0, 3] = transform.ArrayData[9];
            m[1, 3] = transform.ArrayData[10];
            m[2, 3] = transform.ArrayData[11];
            m[3, 3] = transform.ArrayData[12];
            return m;
        }
    }
}

#endif
