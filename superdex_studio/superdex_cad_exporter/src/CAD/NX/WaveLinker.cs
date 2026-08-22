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
using System;

namespace CADRobotExporter.CAD.NX
{
    public enum SelectionBehavior
    {
        IgnoreReadOnly,
        AlwaysCreateWaveLinks,
        OnlyWaveLinkReadOnly,
        AllowReadOnly
    }

    public struct CrossPartResult<T> where T : NXObject
    {
        public bool Allowed;
        public T ResolvedObject;
        public bool WarningShown;
    }

    /// <summary>
    /// Creates WAVE Geometry Links (associative copies) of geometry from child components
    /// into the work part. This avoids modifying parts the user does not own.
    /// </summary>
    public static class WaveLinker
    {
        private static readonly Serilog.ILogger logger = Utilities.Logger.GetLogger();

        private static readonly string[] AllGuidAttributes = new string[]
        {
            NXPersistentId.CSYS_GUID_ATTR,
            NXPersistentId.AXIS_GUID_ATTR,
            NXPersistentId.POINT_GUID_ATTR,
            NXPersistentId.BODY_GUID_ATTR,
        };


        /// <summary>
        /// Clears any copied URDF GUID attributes from a WAVE-linked object.
        /// WAVE links copy attributes from the source, which causes GUID conflicts.
        /// </summary>
        private static void ClearCopiedGuidAttributes(NXObject obj)
        {
            if (obj == null)
                return;

            foreach (string attrName in AllGuidAttributes)
            {
                try
                {
                    if (obj.HasUserAttribute(attrName, NXObject.AttributeType.String, -1))
                    {
                        obj.SetUserAttribute(attrName, -1, "", Update.Option.Now);
                    }
                }
                catch (Exception)
                {
                    // Some derived objects may not allow attribute deletion; ignore
                }
            }
        }

        public static bool IsFromDifferentPart(NXObject obj)
        {
            return obj != null && obj.IsOccurrence;
        }

        /// <summary>
        /// Checks if the object lives in a part the user cannot edit.
        /// In Teamcenter, a part opened as read-only means the user doesn't have it checked out.
        /// </summary>
        public static bool IsOwningPartReadOnly(NXObject obj)
        {
            if (obj == null)
                return false;

            try
            {
                Part owningPart = (Part)obj.OwningPart;
                if (owningPart == null)
                    return false;

                Part workPart = Session.GetSession().Parts.Work;

                // For occurrences, check the prototype's owning part
                if (obj.IsOccurrence)
                {
                    NXObject prototype = obj.Prototype as NXObject;
                    if (prototype != null)
                    {
                        owningPart = (Part)prototype.OwningPart;

                        if (owningPart == workPart)
                            return false;
                    }
                }

                return owningPart != null && !owningPart.HasWriteAccess;
            }
            catch (Exception)
            {
                // If we can't determine ownership, assume it might be read-only
                return obj.IsOccurrence;
            }
        }

        /// <summary>
        /// Centralized logic for handling cross-part selections based on the SelectionBehavior enum.
        /// Returns whether the selection is allowed, and the resolved object (WAVE-linked or original).
        /// </summary>
        public static CrossPartResult<T> HandleCrossPartSelection<T>(
            T obj,
            SelectionBehavior behavior,
            Func<T, T> createWaveLink,
            bool suppressWarning) where T : NXObject
        {
            var result = new CrossPartResult<T>
            {
                Allowed = true,
                ResolvedObject = obj,
                WarningShown = false
            };

            if (!IsFromDifferentPart(obj))
                return result;

            switch (behavior)
            {
                case SelectionBehavior.AlwaysCreateWaveLinks:
                    T linked = createWaveLink(obj);
                    if (linked != null)
                        result.ResolvedObject = linked;
                    break;

                case SelectionBehavior.OnlyWaveLinkReadOnly:
                    if (IsOwningPartReadOnly(obj))
                    {
                        T linkedRO = createWaveLink(obj);
                        if (linkedRO != null)
                            result.ResolvedObject = linkedRO;
                    }
                    break;

                case SelectionBehavior.IgnoreReadOnly:
                    if (IsOwningPartReadOnly(obj))
                    {
                        result.Allowed = false;
                        if (!suppressWarning)
                            result.WarningShown = true;
                    }
                    break;

                case SelectionBehavior.AllowReadOnly:
                    if (IsOwningPartReadOnly(obj) && !suppressWarning)
                    {
                        result.WarningShown = true;
                    }
                    break;
            }

            return result;
        }

        public static SelectionBehavior ParseSelectionBehavior(string value)
        {
            switch (value)
            {
                case "Always create WAVE Links":
                    return SelectionBehavior.AlwaysCreateWaveLinks;
                case "Only create WAVE Links for Read-Only":
                    return SelectionBehavior.OnlyWaveLinkReadOnly;
                case "Allow Read-Only":
                    return SelectionBehavior.AllowReadOnly;
                case "Ignore Read-Only":
                default:
                    return SelectionBehavior.IgnoreReadOnly;
            }
        }

        /// <summary>
        /// Given an object that may be a WAVE-linked entity in the work part, retrieves the
        /// source component it was linked from via the UF Wave API.
        /// Uses AskLinkXform → AskAssyCtxtPartOcc to resolve the source component.
        /// </summary>
        public static Component GetSourceComponentFromWaveLink(NXObject obj)
        {
            if (obj == null)
                return null;

            Part workPart = Session.GetSession().Parts.Work;
            UFSession ufSession = UFSession.GetUFSession();

            try
            {
                Feature feature = NXPersistentId.GetOwningFeature(obj);
                if (feature == null)
                    return null;

                Tag xformTag;
                try
                {
                    ufSession.Wave.AskLinkXform(feature.Tag, out xformTag);
                }
                catch
                {
                    return null;
                }

                if (xformTag == Tag.Null)
                    return null;

                Component rootComponent = workPart.ComponentAssembly?.RootComponent;
                if (rootComponent == null)
                    return null;

                ufSession.So.AskAssyCtxtPartOcc(xformTag, rootComponent.Tag, out Tag sourceCompTag);
                if (sourceCompTag != Tag.Null)
                {
                    return NXOpen.Utilities.NXObjectManager.Get(sourceCompTag) as Component;
                }

                return null;
            }
            catch (Exception)
            {
                return null;
            }
        }


        /// <summary>
        /// Creates a WAVE Datum Link of a CartesianCoordinateSystem into the work part.
        /// Returns the new CSYS that lives in the work part.
        /// </summary>
        public static CartesianCoordinateSystem CreateWaveDatumLink(Part workPart, CartesianCoordinateSystem csys, string name = null)
        {
            WaveLinkBuilder waveLinkBuilder = null;

            try
            {
                waveLinkBuilder = workPart.BaseFeatures.CreateWaveLinkBuilder(null);

                var waveDatumBuilder = waveLinkBuilder.WaveDatumBuilder;
                var extractFaceBuilder = waveLinkBuilder.ExtractFaceBuilder;
                var mirrorBodyBuilder = waveLinkBuilder.MirrorBodyBuilder;

                waveLinkBuilder.Type = WaveLinkBuilder.Types.DatumLink;

                extractFaceBuilder.ParentPart = ExtractFaceBuilder.ParentPartType.OtherPart;
                mirrorBodyBuilder.ParentPartType = MirrorBodyBuilder.ParentPart.OtherPart;

                waveDatumBuilder.Associative = true;
                waveDatumBuilder.MakePositionIndependent = false;
                waveDatumBuilder.HideOriginal = false;
                waveDatumBuilder.InheritDisplayProperties = false;
                waveDatumBuilder.DisplayScale = 2.0;

                waveDatumBuilder.Datums.Add(csys);

                NXObject commitResult = waveLinkBuilder.Commit();

                if (commitResult is Feature feature)
                {
                    if (!string.IsNullOrEmpty(name))
                        feature.SetName(name);
                    ClearCopiedGuidAttributes(feature);
                    NXObject[] entities = feature.GetEntities();
                    foreach (var entity in entities)
                    {
                        ClearCopiedGuidAttributes(entity);
                        if (entity is CartesianCoordinateSystem linkedCsys)
                        {
                            return linkedCsys;
                        }
                    }
                }

                logger.Warning("WAVE DatumLink committed but could not extract CSYS from feature entities");
                return null;
            }
            catch (Exception ex)
            {
                logger.Error($"CreateWaveDatumLink failed: {ex.Message}");
                return null;
            }
            finally
            {
                waveLinkBuilder?.Destroy();
            }
        }

        /// <summary>
        /// Creates a WAVE Point Link of a Point into the work part.
        /// Returns the new Point that lives in the work part.
        /// </summary>
        public static Point CreateWavePointLink(Part workPart, Point point, string name = null)
        {
            WaveLinkBuilder waveLinkBuilder = null;

            try
            {
                waveLinkBuilder = workPart.BaseFeatures.CreateWaveLinkBuilder(null);

                var wavePointBuilder = waveLinkBuilder.WavePointBuilder;
                var extractFaceBuilder = waveLinkBuilder.ExtractFaceBuilder;
                var mirrorBodyBuilder = waveLinkBuilder.MirrorBodyBuilder;

                waveLinkBuilder.Type = WaveLinkBuilder.Types.PointLink;

                extractFaceBuilder.ParentPart = ExtractFaceBuilder.ParentPartType.OtherPart;
                mirrorBodyBuilder.ParentPartType = MirrorBodyBuilder.ParentPart.OtherPart;

                wavePointBuilder.Associative = true;
                wavePointBuilder.MakePositionIndependent = false;
                wavePointBuilder.FixAtCurrentTimestamp = false;
                wavePointBuilder.InheritDisplayProperties = false;

                wavePointBuilder.Points.Add(point);

                NXObject commitResult = waveLinkBuilder.Commit();

                if (commitResult is Feature feature)
                {
                    if (!string.IsNullOrEmpty(name))
                        feature.SetName(name);
                    ClearCopiedGuidAttributes(feature);
                    NXObject[] entities = feature.GetEntities();
                    foreach (var entity in entities)
                    {
                        ClearCopiedGuidAttributes(entity);
                        if (entity is Point linkedPoint)
                        {
                            return linkedPoint;
                        }
                    }
                }

                logger.Warning("WAVE PointLink committed but could not extract Point from feature entities");
                return null;
            }
            catch (Exception ex)
            {
                logger.Error($"CreateWavePointLink failed: {ex.Message}");
                return null;
            }
            finally
            {
                waveLinkBuilder?.Destroy();
            }
        }

        /// <summary>
        /// Creates a WAVE Datum Link of a DatumAxis into the work part.
        /// Returns the new DatumAxis that lives in the work part.
        /// </summary>
        public static DatumAxis CreateWaveAxisLink(Part workPart, DatumAxis axis, string name = null)
        {
            WaveLinkBuilder waveLinkBuilder = null;

            try
            {
                waveLinkBuilder = workPart.BaseFeatures.CreateWaveLinkBuilder(null);

                var waveDatumBuilder = waveLinkBuilder.WaveDatumBuilder;
                var extractFaceBuilder = waveLinkBuilder.ExtractFaceBuilder;
                var mirrorBodyBuilder = waveLinkBuilder.MirrorBodyBuilder;

                waveLinkBuilder.Type = WaveLinkBuilder.Types.DatumLink;

                extractFaceBuilder.ParentPart = ExtractFaceBuilder.ParentPartType.OtherPart;
                mirrorBodyBuilder.ParentPartType = MirrorBodyBuilder.ParentPart.OtherPart;

                waveDatumBuilder.Associative = true;
                waveDatumBuilder.MakePositionIndependent = false;
                waveDatumBuilder.HideOriginal = false;
                waveDatumBuilder.InheritDisplayProperties = false;
                waveDatumBuilder.DisplayScale = 2.0;

                waveDatumBuilder.Datums.Add(axis);

                NXObject commitResult = waveLinkBuilder.Commit();

                if (commitResult is Feature feature)
                {
                    if (!string.IsNullOrEmpty(name))
                        feature.SetName(name);
                    ClearCopiedGuidAttributes(feature);
                    NXObject[] entities = feature.GetEntities();
                    foreach (var entity in entities)
                    {
                        ClearCopiedGuidAttributes(entity);
                        if (entity is DatumAxis linkedAxis)
                        {
                            return linkedAxis;
                        }
                    }
                }

                logger.Warning("WAVE DatumLink (axis) committed but could not extract DatumAxis from feature entities");
                return null;
            }
            catch (Exception ex)
            {
                logger.Error($"CreateWaveAxisLink failed: {ex.Message}");
                return null;
            }
            finally
            {
                waveLinkBuilder?.Destroy();
            }
        }

        /// <summary>
        /// Creates a WAVE Body Link of a Body into the work part.
        /// Returns the new Body that lives in the work part.
        /// ** THIS IS NOT WORKING RELIABLY **
        /// </summary>
        public static Body CreateWaveBodyLink(Part workPart, Body body, string name = null)
        {
            WaveLinkBuilder waveLinkBuilder = null;

            try
            {
                waveLinkBuilder = workPart.BaseFeatures.CreateWaveLinkBuilder(null);

                var extractFaceBuilder = waveLinkBuilder.ExtractFaceBuilder;

                waveLinkBuilder.Type = WaveLinkBuilder.Types.BodyLink;

                extractFaceBuilder.FaceOption = ExtractFaceBuilder.FaceOptionType.AllBodyFaces;
                extractFaceBuilder.AngleTolerance = 45.0;
                extractFaceBuilder.ParentPart = ExtractFaceBuilder.ParentPartType.OtherPart;

                extractFaceBuilder.Associative = true;
                extractFaceBuilder.MakePositionIndependent = false;
                extractFaceBuilder.FixAtCurrentTimestamp = false;
                extractFaceBuilder.HideOriginal = false;
                extractFaceBuilder.InheritDisplayProperties = false;
                extractFaceBuilder.CopyThreads = true;
                extractFaceBuilder.FeatureOption = ExtractFaceBuilder.FeatureOptionType.OneFeatureForAllBodies;
                extractFaceBuilder.CopyGroups = false;
                extractFaceBuilder.InheritMaterial = true;
                waveLinkBuilder.InheritMaterial = true;

                var scCollector = extractFaceBuilder.ExtractBodyCollector;
                Body[] bodies = new Body[] { body };

                var bodyDumbRule = workPart.ScRuleFactory.CreateRuleBodyDumb(bodies);
                SelectionIntentRule[] rules = new SelectionIntentRule[] { bodyDumbRule };
                scCollector.ReplaceRules(rules, false);

                bool validated = waveLinkBuilder.Validate();
                NXObject commitResult = waveLinkBuilder.Commit();

                if (commitResult is Feature feature)
                {
                    if (!string.IsNullOrEmpty(name))
                        feature.SetName(name);
                    ClearCopiedGuidAttributes(feature);
                    NXObject[] entities = feature.GetEntities();
                    foreach (var entity in entities)
                    {
                        ClearCopiedGuidAttributes(entity);
                        if (entity is Body linkedBody)
                        {
                            return linkedBody;
                        }
                    }
                }

                logger.Warning("WAVE BodyLink committed but could not extract Body from feature entities");
                return null;
            }
            catch (Exception ex)
            {
                logger.Error($"CreateWaveBodyLink failed: {ex.Message}");
                return null;
            }
            finally
            {
                waveLinkBuilder?.Destroy();
            }
        }
    }
}
