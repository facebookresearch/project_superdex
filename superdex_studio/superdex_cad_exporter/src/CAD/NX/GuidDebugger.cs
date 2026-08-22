/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using NXOpen;
using NXOpen.Assemblies;
using NXOpen.BlockStyler;
using NXOpen.MenuBar;
using NXOpen.UF;

namespace CADRobotExporter.CAD.NX
{
    public class GuidDebugger : IDisposable
    {
        private static Session theSession = null;
        private static NXOpen.UI theUI = null;
        private static UFSession ufSession = null;
        private string theDlxFileName;
        private NXOpen.BlockStyler.BlockDialog theDialog;
        private NXOpen.BlockStyler.Group group0;
        private NXOpen.BlockStyler.SelectObject selectionObject;
        private NXOpen.BlockStyler.Button buttonClearGUID;
        private NXOpen.BlockStyler.MultilineString multilineStringOutput;

        private static readonly string[] ALL_GUID_ATTRS = new string[]
        {
            NXPersistentId.COMPONENT_GUID_ATTR,
            NXPersistentId.BODY_GUID_ATTR,
            NXPersistentId.CSYS_GUID_ATTR,
            NXPersistentId.AXIS_GUID_ATTR,
            NXPersistentId.POINT_GUID_ATTR,
            NXPersistentId.GENERIC_GUID_ATTR,
        };

        public GuidDebugger()
        {
            try
            {
                theSession = Session.GetSession();
                theUI = NXOpen.UI.GetUI();
                ufSession = UFSession.GetUFSession();
                theDlxFileName = "GuidDebugger.dlx";
                theDialog = theUI.CreateDialog(theDlxFileName);
                theDialog.AddApplyHandler(new NXOpen.BlockStyler.BlockDialog.Apply(apply_cb));
                theDialog.AddOkHandler(new NXOpen.BlockStyler.BlockDialog.Ok(ok_cb));
                theDialog.AddUpdateHandler(new NXOpen.BlockStyler.BlockDialog.Update(update_cb));
                theDialog.AddInitializeHandler(new NXOpen.BlockStyler.BlockDialog.Initialize(initialize_cb));
                theDialog.AddDialogShownHandler(new NXOpen.BlockStyler.BlockDialog.DialogShown(dialogShown_cb));
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }

        public static MenuBarManager.CallbackStatus LaunchGuidDebugger(MenuButtonEvent buttonEvent)
        {
            if (Session.GetSession().Parts.Work == null)
            {
                return MenuBarManager.CallbackStatus.Cancel;
            }

            GuidDebugger theDebugger = null;
            try
            {
                theDebugger = new GuidDebugger();
                theDebugger.Launch();
            }
            catch (Exception ex)
            {
                NXOpen.UI.GetUI().NXMessageBox.Show("GUID Debugger", NXMessageBox.DialogType.Error, ex.ToString());
            }
            finally
            {
                theDebugger?.Dispose();
            }
            return MenuBarManager.CallbackStatus.Continue;
        }

        public static int GetUnloadOption(string arg)
        {
            return System.Convert.ToInt32(Session.LibraryUnloadOption.AtTermination);
        }

        public NXOpen.BlockStyler.BlockDialog.DialogResponse Launch()
        {
            NXOpen.BlockStyler.BlockDialog.DialogResponse dialogResponse = NXOpen.BlockStyler.BlockDialog.DialogResponse.Invalid;
            try
            {
                dialogResponse = theDialog.Launch();
            }
            catch (Exception ex)
            {
                theUI.NXMessageBox.Show("GUID Debugger", NXMessageBox.DialogType.Error, ex.ToString());
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

        public void initialize_cb()
        {
            try
            {
                group0 = (NXOpen.BlockStyler.Group)theDialog.TopBlock.FindBlock("group0");
                selectionObject = (NXOpen.BlockStyler.SelectObject)theDialog.TopBlock.FindBlock("selectionObject");
                buttonClearGUID = (NXOpen.BlockStyler.Button)theDialog.TopBlock.FindBlock("buttonClearGUID");
                multilineStringOutput = (NXOpen.BlockStyler.MultilineString)theDialog.TopBlock.FindBlock("multilineStringOutput");
            }
            catch (Exception ex)
            {
                theUI.NXMessageBox.Show("GUID Debugger", NXMessageBox.DialogType.Error, ex.ToString());
            }
        }

        public void dialogShown_cb()
        {
            try
            {
                multilineStringOutput.SetValue(new string[] { "Select object(s) and click Fetch GUID." });
            }
            catch (Exception ex)
            {
                theUI.NXMessageBox.Show("GUID Debugger", NXMessageBox.DialogType.Error, ex.ToString());
            }
        }

        public int apply_cb()
        {
            return 0;
        }

        public int update_cb(NXOpen.BlockStyler.UIBlock block)
        {
            try
            {
                if (block == selectionObject)
                {
                    FetchGuids();
                }
                else if (block == buttonClearGUID)
                {
                    ClearGuids();
                }
            }
            catch (Exception ex)
            {
                theUI.NXMessageBox.Show("GUID Debugger", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return 0;
        }

        public int ok_cb()
        {
            return 0;
        }

        private void SetOutput(List<string> lines)
        {
            multilineStringOutput.SetValue(lines.ToArray());
        }

        private void FetchGuids()
        {
            TaggedObject[] objects = selectionObject.GetSelectedObjects();
            if (objects == null || objects.Length == 0)
            {
                SetOutput(new List<string> { "No objects selected." });
                return;
            }

            List<string> lines = new List<string>();

            foreach (TaggedObject taggedObj in objects)
            {
                NXObject obj = taggedObj as NXObject;
                if (obj == null) continue;

                lines.Add($"=== {obj.GetType().Name} | Tag: {obj.Tag} | Handle: {GetHandle(obj)} ===");
                lines.Add($"  Name: {GetName(obj)}");
                lines.Add($"  IsOccurrence: {IsOccurrence(obj)}");

                lines.Add("  [Direct Attributes]");
                AppendAttributes(lines, obj, "    ");

                if (IsOccurrence(obj))
                {
                    NXObject prototype = GetPrototype(obj);
                    if (prototype != null && prototype.Tag != obj.Tag)
                    {
                        lines.Add($"  [Prototype] Tag: {prototype.Tag} | Handle: {GetHandle(prototype)} | Name: {GetName(prototype)}");
                        AppendAttributes(lines, prototype, "    ");
                    }
                }

                Component owningComp = GetOwningComponent(obj);
                if (owningComp != null)
                {
                    lines.Add($"  [Owning Component] {owningComp.DisplayName} Tag: {owningComp.Tag} | Handle: {GetHandle(owningComp)}");
                    AppendAttributes(lines, owningComp, "    ");
                }

                NXObject targetForFeature = IsOccurrence(obj) ? GetPrototype(obj) : obj;
                NXOpen.Features.Feature owningFeature = GetOwningFeature(targetForFeature);
                if (owningFeature != null)
                {
                    lines.Add($"  [Owning Feature] {owningFeature.GetFeatureName()} Tag: {owningFeature.Tag} | Handle: {GetHandle(owningFeature)}");
                    AppendAttributes(lines, owningFeature, "    ");
                }

                if (obj is Component comp)
                {
                    lines.Add($"  [Component Info]");
                    lines.Add($"    DisplayName: {comp.DisplayName}");
                    if (comp.Parent != null)
                        lines.Add($"    Parent: {comp.Parent.DisplayName}");
                    AppendAttributes(lines, comp, "    ");
                }

                lines.Add("");
            }

            SetOutput(lines);
        }

        private void ClearGuids()
        {
            TaggedObject[] objects = selectionObject.GetSelectedObjects();
            if (objects == null || objects.Length == 0)
            {
                SetOutput(new List<string> { "No objects selected." });
                return;
            }

            List<string> lines = new List<string>();
            int cleared = 0;

            foreach (TaggedObject taggedObj in objects)
            {
                NXObject obj = taggedObj as NXObject;
                if (obj == null) continue;

                NXObject target = IsOccurrence(obj) ? GetPrototype(obj) : obj;

                lines.Add($"Clearing: {GetName(obj)} (target tag: {target.Tag})");

                foreach (string attr in ALL_GUID_ATTRS)
                {
                    if (target.HasUserAttribute(attr, NXObject.AttributeType.String, -1))
                    {
                        string oldValue = target.GetStringUserAttribute(attr, -1);
                        if (!string.IsNullOrEmpty(oldValue))
                        {
                            target.SetUserAttribute(attr, -1, "", Update.Option.Now);
                            lines.Add($"  Cleared {attr} = {oldValue}");
                            cleared++;
                        }
                    }
                }

                NXOpen.Features.Feature feature = GetOwningFeature(target);
                if (feature != null)
                {
                    foreach (string attr in ALL_GUID_ATTRS)
                    {
                        if (feature.HasUserAttribute(attr, NXObject.AttributeType.String, -1))
                        {
                            string oldValue = feature.GetStringUserAttribute(attr, -1);
                            if (!string.IsNullOrEmpty(oldValue))
                            {
                                feature.SetUserAttribute(attr, -1, "", Update.Option.Now);
                                lines.Add($"  Cleared {attr} on feature {feature.GetFeatureName()} = {oldValue}");
                                cleared++;
                            }
                        }
                    }
                }
            }

            lines.Add("");
            lines.Add($"Total attributes cleared: {cleared}");
            SetOutput(lines);
        }

        private void AppendAttributes(List<string> lines, NXObject obj, string indent)
        {
            bool found = false;
            foreach (string attr in ALL_GUID_ATTRS)
            {
                try
                {
                    if (obj.HasUserAttribute(attr, NXObject.AttributeType.String, -1))
                    {
                        string val = obj.GetStringUserAttribute(attr, -1);
                        lines.Add($"{indent}{attr} = \"{val}\"");
                        found = true;
                    }
                }
                catch { }
            }
            if (!found)
            {
                lines.Add($"{indent}(no URDF attributes)");
            }
        }

        private static bool IsOccurrence(NXObject obj)
        {
            try { return obj.IsOccurrence; }
            catch { return false; }
        }

        private static NXObject GetPrototype(NXObject obj)
        {
            try
            {
                if (obj.IsOccurrence)
                    return obj.Prototype as NXObject ?? obj;
            }
            catch { }
            return obj;
        }

        private static Component GetOwningComponent(NXObject obj)
        {
            try
            {
                if (obj.IsOccurrence)
                    return obj.OwningComponent;
            }
            catch { }
            return null;
        }

        private static NXOpen.Features.Feature GetOwningFeature(NXObject obj)
        {
            if (obj == null) return null;
            try
            {
                ufSession.Modl.AskObjectFeat(obj.Tag, out Tag featureTag);
                if (featureTag != Tag.Null)
                    return NXOpen.Utilities.NXObjectManager.Get(featureTag) as NXOpen.Features.Feature;
            }
            catch { }
            return null;
        }

        private static string GetName(NXObject obj)
        {
            try { return obj.Name; }
            catch { return "(unnamed)"; }
        }

        private static string GetHandle(NXObject obj)
        {
            try
            {
                ufSession.Tag.AskHandleFromTag(obj.Tag, out string handle);
                return handle;
            }
            catch { return "(unknown)"; }
        }
    }
}
