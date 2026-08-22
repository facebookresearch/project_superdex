/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if NX

using NXOpen;
using NXOpen.Assemblies;
using NXOpen.Features;
using NXOpen.UF;
using CADRobotExporter.Utilities;
using System;
using System.Collections.Generic;
using System.Text;

namespace CADRobotExporter.CAD.NX
{
    /// <summary>
    /// Provides GUID-based persistent identification for NX objects.
    ///
    /// Key design principles:
    /// 1. Always work with PROTOTYPES, not occurrences - occurrences are transient
    /// 2. Store GUIDs on owning Feature when available, fallback to object directly (STEP imports)
    /// 3. Use Component Path to uniquely identify instances of the same part
    /// 4. Composite key format: "ComponentPath|ObjectGUID" (e.g., "Root/SubAsm/Part|abc-123")
    /// </summary>
    public static class NXPersistentId
    {
        private static readonly Serilog.ILogger logger = Logger.GetLogger();
        private static readonly UFSession ufSession = UFSession.GetUFSession();

        public static readonly string COMPONENT_GUID_ATTR = "URDF_COMPONENT_ID";
        public static readonly string BODY_GUID_ATTR = "URDF_BODY_ID";
        public static readonly string CSYS_GUID_ATTR = "URDF_CSYS_ID";
        public static readonly string AXIS_GUID_ATTR = "URDF_AXIS_ID";
        public static readonly string POINT_GUID_ATTR = "URDF_POINT_ID";
        public static readonly string GENERIC_GUID_ATTR = "URDF_OBJECT_ID";

        private const char PATH_SEPARATOR = '/';
        private const char KEY_SEPARATOR = '|';

        #region Component Path Utilities (Using Component GUIDs)

        /// <summary>
        /// Gets or creates a GUID for a Component.
        /// Component GUIDs are stored on the Component itself (in the assembly file).
        /// </summary>
        public static string GetOrCreateComponentGuid(Component component)
        {
            if (component == null)
                return null;

            try
            {
                if (component.HasUserAttribute(COMPONENT_GUID_ATTR, NXObject.AttributeType.String, -1))
                {
                    string existingGuid = component.GetStringUserAttribute(COMPONENT_GUID_ATTR, -1);
                    if (!string.IsNullOrEmpty(existingGuid))
                        return existingGuid;
                }

                string newGuid = Guid.NewGuid().ToString();
                component.SetUserAttribute(COMPONENT_GUID_ATTR, -1, newGuid, Update.Option.Now);
                logger.Debug($"GetOrCreateComponentGuid: Created GUID '{newGuid}' for component '{component.Name}'");
                return newGuid;
            }
            catch (Exception ex)
            {
                logger.Warning($"GetOrCreateComponentGuid: Failed - {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Builds a unique component path string using Component GUIDs.
        /// Format: "CompGUID1/CompGUID2/CompGUID3" (from root to leaf)
        /// </summary>
        public static string GetComponentPath(Component component)
        {
            if (component == null)
                return "";

            var pathParts = new List<string>();
            Component current = component;

            while (current != null)
            {
                string compGuid = GetOrCreateComponentGuid(current);
                if (!string.IsNullOrEmpty(compGuid))
                    pathParts.Add(compGuid);

                try
                {
                    current = current.Parent;
                }
                catch
                {
                    current = null;
                }
            }

            // Reverse to get root-to-leaf order
            pathParts.Reverse();
            return string.Join(PATH_SEPARATOR.ToString(), pathParts);
        }

        /// <summary>
        /// Finds a component by its GUID.
        /// </summary>
        public static Component FindComponentByGuid(Part workPart, string componentGuid)
        {
            if (workPart == null || string.IsNullOrEmpty(componentGuid))
                return null;

            if (workPart.ComponentAssembly?.RootComponent == null)
                return null;

            return FindComponentByGuidRecursive(workPart.ComponentAssembly.RootComponent, componentGuid);
        }

        private static Component FindComponentByGuidRecursive(Component parent, string targetGuid)
        {
            if (parent == null)
                return null;

            // Check if this component has the target GUID
            try
            {
                if (parent.HasUserAttribute(COMPONENT_GUID_ATTR, NXObject.AttributeType.String, -1))
                {
                    string compGuid = parent.GetStringUserAttribute(COMPONENT_GUID_ATTR, -1);
                    if (string.Equals(compGuid, targetGuid, StringComparison.OrdinalIgnoreCase))
                        return parent;
                }
            }
            catch { }

            // Search children
            Component[] children = parent.GetChildren();
            if (children != null)
            {
                foreach (Component child in children)
                {
                    Component found = FindComponentByGuidRecursive(child, targetGuid);
                    if (found != null)
                        return found;
                }
            }

            return null;
        }

        /// <summary>
        /// Finds a component by navigating a path of Component GUIDs.
        /// </summary>
        public static Component FindComponentByPath(Part workPart, string componentPath)
        {
            if (workPart == null || string.IsNullOrEmpty(componentPath))
                return null;

            if (workPart.ComponentAssembly?.RootComponent == null)
                return null;

            string[] pathGuids = componentPath.Split(PATH_SEPARATOR);
            if (pathGuids.Length == 0)
                return null;

            // The last GUID in the path is our target component
            string targetGuid = pathGuids[pathGuids.Length - 1];

            logger.Debug($"FindComponentByPath: Looking for component with GUID '{targetGuid}'");

            Component found = FindComponentByGuid(workPart, targetGuid);
            if (found != null)
            {
                logger.Debug($"FindComponentByPath: Found component '{found.Name}'");
            }
            else
            {
                logger.Warning($"FindComponentByPath: Could not find component with GUID '{targetGuid}'");
            }

            return found;
        }

        /// <summary>
        /// Creates a composite key from component path and object GUID.
        /// Format: "ComponentPath|ObjectGUID"
        /// </summary>
        public static string CreateCompositeKey(string componentPath, string objectGuid)
        {
            if (string.IsNullOrEmpty(componentPath))
                return objectGuid; // No component = work part object

            return $"{componentPath}{KEY_SEPARATOR}{objectGuid}";
        }

        /// <summary>
        /// Parses a composite key into component path and object GUID.
        /// </summary>
        public static (string componentPath, string objectGuid) ParseCompositeKey(string compositeKey)
        {
            if (string.IsNullOrEmpty(compositeKey))
                return (null, null);

            int separatorIndex = compositeKey.LastIndexOf(KEY_SEPARATOR);
            if (separatorIndex < 0)
                return (null, compositeKey); // No component path, just GUID (work part object)

            string componentPath = compositeKey.Substring(0, separatorIndex);
            string objectGuid = compositeKey.Substring(separatorIndex + 1);

            return (componentPath, objectGuid);
        }

        #endregion

        #region Get or Create Composite Keys

        /// <summary>
        /// Gets or creates a composite key for any NXObject.
        /// Returns "ComponentPath|ObjectGUID" for occurrences, or just "ObjectGUID" for work part objects.
        /// </summary>
        public static string GetOrCreateObjectKey(NXObject obj)
        {
            if (obj == null)
                return null;

            if (obj is Body body)
                return GetOrCreateBodyKey(body);
            if (obj is CartesianCoordinateSystem csys)
                return GetOrCreateCoordinateSystemKey(csys);
            if (obj is DatumAxis axis)
                return GetOrCreateAxisKey(axis);
            if (obj is Point point)
                return GetOrCreatePointKey(point);

            // Generic fallback
            string componentPath = "";
            NXObject prototype = obj;

            if (obj.IsOccurrence)
            {
                Component owningComponent = GetOwningComponent(obj);
                if (owningComponent != null)
                    componentPath = GetComponentPath(owningComponent);
                prototype = GetPrototype(obj);
            }

            string guid = GetOrCreateGuidOnTarget(prototype, GENERIC_GUID_ATTR);

            //Feature feature = GetOwningFeature(prototype);
            //string guid = feature != null
            //    ? GetOrCreateGuidOnTarget(feature, GENERIC_GUID_ATTR)
            //    : GetOrCreateGuidOnTarget(prototype, GENERIC_GUID_ATTR);

            return CreateCompositeKey(componentPath, guid);
        }

        /// <summary>
        /// Gets or creates a composite key for a Body.
        /// </summary>
        public static string GetOrCreateBodyKey(Body body)
        {
            if (body == null)
                return null;

            string componentPath = "";
            Body prototype = body;

            if (body.IsOccurrence)
            {
                Component owningComponent = body.OwningComponent;
                if (owningComponent != null)
                    componentPath = GetComponentPath(owningComponent);
                prototype = GetPrototype(body) as Body ?? body;
            }

            string guid = GetOrCreateGuidOnTarget(prototype, BODY_GUID_ATTR);

            //Feature feature = GetOwningFeature(prototype);
            //string guid = feature != null
            //    ? GetOrCreateGuidOnTarget(feature, BODY_GUID_ATTR)
            //    : GetOrCreateGuidOnTarget(prototype, BODY_GUID_ATTR);

            return CreateCompositeKey(componentPath, guid);
        }

        public static bool BodyHasKey(Body body)
        {
            Body prototype = body;

            if (body.IsOccurrence)
            {
                prototype = GetPrototype(body) as Body ?? body;
            }

            if (prototype.HasUserAttribute(BODY_GUID_ATTR, NXObject.AttributeType.String, -1))
            {
                return prototype.GetStringUserAttribute(BODY_GUID_ATTR, -1) != "";
            }
            return false;
        }

        /// <summary>
        /// Gets or creates a composite key for a CartesianCoordinateSystem.
        /// </summary>
        public static string GetOrCreateCoordinateSystemKey(CartesianCoordinateSystem csys)
        {
            if (csys == null)
                return null;

            string componentPath = "";
            CartesianCoordinateSystem prototype = csys;

            if (csys.IsOccurrence)
            {
                Component owningComponent = csys.OwningComponent;
                if (owningComponent != null)
                    componentPath = GetComponentPath(owningComponent);
                prototype = GetPrototype(csys) as CartesianCoordinateSystem ?? csys;
            }

            string guid = GetOrCreateGuidOnTarget(prototype, CSYS_GUID_ATTR);

            //Feature feature = GetOwningFeature(prototype);
            //string guid = feature != null
            //    ? GetOrCreateGuidOnTarget(feature, CSYS_GUID_ATTR)
            //    : GetOrCreateGuidOnTarget(prototype, CSYS_GUID_ATTR);

            return CreateCompositeKey(componentPath, guid);
        }

        public static bool CoordinateSystemHasKey(CartesianCoordinateSystem csys)
        {
            CartesianCoordinateSystem prototype = csys;

            if (csys.IsOccurrence)
            {
                prototype = GetPrototype(csys) as CartesianCoordinateSystem ?? csys;
            }

            if (prototype.HasUserAttribute(CSYS_GUID_ATTR, NXObject.AttributeType.String, -1))
            {
                return prototype.GetStringUserAttribute(CSYS_GUID_ATTR, -1) != "";
            }
            return false;
        }

        /// <summary>
        /// Gets or creates a composite key for a DatumAxis.
        /// </summary>
        public static string GetOrCreateAxisKey(DatumAxis axis)
        {
            if (axis == null)
                return null;

            string componentPath = "";
            DatumAxis prototype = axis;

            if (axis.IsOccurrence)
            {
                Component owningComponent = axis.OwningComponent;
                if (owningComponent != null)
                    componentPath = GetComponentPath(owningComponent);
                prototype = GetPrototype(axis) as DatumAxis ?? axis;
            }

            string guid = GetOrCreateGuidOnTarget(prototype, AXIS_GUID_ATTR);

            //Feature feature = GetOwningFeature(prototype);
            //if (feature != null)
            //{
            //    guid = GetOrCreateGuidOnTarget(feature, AXIS_GUID_ATTR);
            //}
            //else
            //{
            //guid = GetOrCreateGuidOnTarget(prototype, AXIS_GUID_ATTR);
            //}

            return CreateCompositeKey(componentPath, guid);
        }

        public static bool AxisHasKey(DatumAxis axis)
        {
            DatumAxis prototype = axis;

            if (axis.IsOccurrence)
            {
                prototype = GetPrototype(axis) as DatumAxis ?? axis;
            }

            if (prototype.HasUserAttribute(AXIS_GUID_ATTR, NXObject.AttributeType.String, -1))
            {
                return prototype.GetStringUserAttribute(AXIS_GUID_ATTR, -1) != "";
            }
            return false;
        }

        /// <summary>
        /// Gets or creates a composite key for a Point.
        /// </summary>
        public static string GetOrCreatePointKey(Point point)
        {
            if (point == null)
                return null;

            string componentPath = "";
            Point prototype = point;

            if (point.IsOccurrence)
            {
                Component owningComponent = point.OwningComponent;
                if (owningComponent != null)
                    componentPath = GetComponentPath(owningComponent);
                prototype = GetPrototype(point) as Point ?? point;
            }

            string guid = GetOrCreateGuidOnTarget(prototype, POINT_GUID_ATTR);

            return CreateCompositeKey(componentPath, guid);
        }

        public static bool PointHasKey(Point point)
        {
            Point prototype = point;

            if (point.IsOccurrence)
            {
                prototype = GetPrototype(point) as Point ?? point;
            }

            if (prototype.HasUserAttribute(POINT_GUID_ATTR, NXObject.AttributeType.String, -1))
            {
                return prototype.GetStringUserAttribute(POINT_GUID_ATTR, -1) != "";
            }
            return false;
        }

        /// <summary>
        /// Gets composite keys for a list of bodies.
        /// </summary>
        public static List<string> GetOrCreateBodyKeys(IEnumerable<Body> bodies)
        {
            var keys = new List<string>();
            if (bodies == null)
                return keys;

            foreach (Body body in bodies)
            {
                string key = GetOrCreateBodyKey(body);
                if (!string.IsNullOrEmpty(key))
                    keys.Add(key);
            }
            return keys;
        }

        #endregion

        #region Find by Composite Key

        /// <summary>
        /// Finds a Body by its composite key (ComponentPath|ObjectGUID).
        /// </summary>
        public static Body FindBodyByKey(Part workPart, string compositeKey)
        {
            if (workPart == null || string.IsNullOrEmpty(compositeKey))
                return null;

            var (componentPath, objectGuid) = ParseCompositeKey(compositeKey);

            if (string.IsNullOrEmpty(objectGuid))
                return null;

            // If no component path, search work part directly
            if (string.IsNullOrEmpty(componentPath))
                return FindBodyInPart(workPart, objectGuid);

            // Find the specific component
            Component component = FindComponentByPath(workPart, componentPath);
            if (component == null)
            {
                logger.Warning($"Could not find component at path: {componentPath}");
                return null;
            }

            // Find the body in the component's part
            Part componentPart = component.Prototype as Part;
            if (componentPart == null)
                return null;

            if (!componentPart.IsFullyLoaded)
            {
                logger.Debug($"FindBodyByKey: Fully loading: {componentPart.Name}");
                componentPart.LoadFully();
            }

            Body prototypeBody = FindBodyInPart(componentPart, objectGuid);
            if (prototypeBody == null)
                return null;

            // Get the occurrence
            return component.FindOccurrence(prototypeBody) as Body ?? prototypeBody;
        }

        /// <summary>
        /// Finds a CartesianCoordinateSystem by its composite key.
        /// </summary>
        public static CartesianCoordinateSystem FindCoordinateSystemByKey(Part workPart, string compositeKey)
        {
            if (workPart == null || string.IsNullOrEmpty(compositeKey))
                return null;

            logger.Debug($"FindCoordinateSystemByKey: compositeKey = '{compositeKey}'");

            var (componentPath, objectGuid) = ParseCompositeKey(compositeKey);
            logger.Debug($"FindCoordinateSystemByKey: componentPath = '{componentPath ?? "(null)"}'");
            logger.Debug($"FindCoordinateSystemByKey: objectGuid = '{objectGuid}'");

            if (string.IsNullOrEmpty(objectGuid))
                return null;

            // If no component path, search work part directly
            if (string.IsNullOrEmpty(componentPath))
            {
                logger.Debug("FindCoordinateSystemByKey: No component path, searching work part directly");
                var result = FindCsysInPart(workPart, objectGuid);
                logger.Debug($"FindCoordinateSystemByKey: FindCsysInPart returned {(result != null ? $"tag={result.Tag}" : "null")}");
                return result;
            }

            // Find the specific component
            Component component = FindComponentByPath(workPart, componentPath);
            if (component == null)
            {
                logger.Warning($"FindCoordinateSystemByKey: Could not find component at path: {componentPath}");
                return null;
            }
            logger.Debug($"FindCoordinateSystemByKey: Found component: {component.Name}");

            // Find the csys in the component's part
            Part componentPart = component.Prototype as Part;
            if (componentPart == null)
            {
                logger.Warning("FindCoordinateSystemByKey: Component prototype is not a Part");
                return null;
            }

            if (!componentPart.IsFullyLoaded)
            {
                logger.Debug($"FindCoordinateSystemByKey: Fully loading: {componentPart.Name}");
                componentPart.LoadFully();
            }

            logger.Debug($"FindCoordinateSystemByKey: Component part: {componentPart.Name}");

            logger.Debug($"FindCoordinateSystemByKey: Searching for GUID '{objectGuid}' in part...");
            int csysCount = 0;
            foreach (CartesianCoordinateSystem csys in componentPart.CoordinateSystems)
            {
                csysCount++;
                bool hasAttr = false;
                string attrValue = null;
                try
                {
                    hasAttr = csys.HasUserAttribute(CSYS_GUID_ATTR, NXObject.AttributeType.String, -1);
                    if (hasAttr)
                        attrValue = csys.GetStringUserAttribute(CSYS_GUID_ATTR, -1);
                }
                catch { }
                logger.Debug($"  csys tag={csys.Tag}, hasAttr={hasAttr}, guid='{attrValue ?? "(none)"}'");
            }
            logger.Debug($"FindCoordinateSystemByKey: Total csys in part: {csysCount}");

            CartesianCoordinateSystem prototypeCsys = FindCsysInPart(componentPart, objectGuid);
            if (prototypeCsys == null)
            {
                logger.Warning($"FindCoordinateSystemByKey: Could not find csys with GUID '{objectGuid}' in part '{componentPart.Name}'");
                return null;
            }
            logger.Debug($"FindCoordinateSystemByKey: Found prototype csys tag={prototypeCsys.Tag}");

            // Get the occurrence
            var occurrence = component.FindOccurrence(prototypeCsys) as CartesianCoordinateSystem;
            logger.Debug($"FindCoordinateSystemByKey: FindOccurrence returned {(occurrence != null ? $"tag={occurrence.Tag}" : "null")}");
            return occurrence ?? prototypeCsys;
        }

        /// <summary>
        /// Finds a DatumAxis by its composite key.
        /// </summary>
        public static DatumAxis FindAxisByKey(Part workPart, string compositeKey)
        {
            if (workPart == null || string.IsNullOrEmpty(compositeKey))
                return null;

            logger.Debug($"FindAxisByKey: compositeKey = '{compositeKey}'");

            var (componentPath, objectGuid) = ParseCompositeKey(compositeKey);
            logger.Debug($"FindAxisByKey: componentPath = '{componentPath ?? "(null)"}'");
            logger.Debug($"FindAxisByKey: objectGuid = '{objectGuid}'");

            if (string.IsNullOrEmpty(objectGuid))
                return null;

            // If no component path, search work part directly
            if (string.IsNullOrEmpty(componentPath))
            {
                logger.Debug("FindAxisByKey: No component path, searching work part directly");
                var result = FindAxisInPart(workPart, objectGuid);
                logger.Debug($"FindAxisByKey: FindAxisInPart returned {(result != null ? $"tag={result.Tag}" : "null")}");
                return result;
            }

            // Find the specific component
            Component component = FindComponentByPath(workPart, componentPath);
            if (component == null)
            {
                logger.Warning($"FindAxisByKey: Could not find component at path: {componentPath}");
                return null;
            }
            logger.Debug($"FindAxisByKey: Found component: {component.Name}");

            // Find the axis in the component's part
            Part componentPart = component.Prototype as Part;
            if (componentPart == null)
            {
                logger.Warning("FindAxisByKey: Component prototype is not a Part");
                return null;
            }
            if (!componentPart.IsFullyLoaded)
            {
                logger.Debug($"FindAxisByKey: Fully loading: {componentPart.Name}");
                componentPart.LoadFully();
            }
            logger.Debug($"FindAxisByKey: Component part: {componentPart.Name}");

            // Debug: List all axes in the part and check for GUID attributes
            logger.Debug($"FindAxisByKey: Searching for GUID '{objectGuid}' in part...");
            int axisCount = 0;
            foreach (DisplayableObject datum in componentPart.Datums)
            {
                if (datum is DatumAxis axis)
                {
                    axisCount++;
                    bool hasAttr = false;
                    string attrValue = null;
                    try
                    {
                        hasAttr = axis.HasUserAttribute(AXIS_GUID_ATTR, NXObject.AttributeType.String, -1);
                        if (hasAttr)
                            attrValue = axis.GetStringUserAttribute(AXIS_GUID_ATTR, -1);
                    }
                    catch { }
                    logger.Debug($"  axis tag={axis.Tag}, hasAttr={hasAttr}, guid='{attrValue ?? "(none)"}'");
                }
            }
            logger.Debug($"FindAxisByKey: Total axes in part: {axisCount}");

            DatumAxis prototypeAxis = FindAxisInPart(componentPart, objectGuid);
            if (prototypeAxis == null)
            {
                logger.Warning($"FindAxisByKey: Could not find axis with GUID '{objectGuid}' in part '{componentPart.Name}'");
                return null;
            }
            logger.Debug($"FindAxisByKey: Found prototype axis tag={prototypeAxis.Tag}");

            // Get the occurrence
            var occurrence = component.FindOccurrence(prototypeAxis) as DatumAxis;
            logger.Debug($"FindAxisByKey: FindOccurrence returned {(occurrence != null ? $"tag={occurrence.Tag}" : "null")}");
            return occurrence ?? prototypeAxis;
        }

        /// <summary>
        /// Finds a Point by its composite key.
        /// </summary>
        public static Point FindPointByKey(Part workPart, string compositeKey)
        {
            if (workPart == null || string.IsNullOrEmpty(compositeKey))
                return null;

            var (componentPath, objectGuid) = ParseCompositeKey(compositeKey);

            if (string.IsNullOrEmpty(objectGuid))
                return null;

            if (string.IsNullOrEmpty(componentPath))
                return FindPointInPart(workPart, objectGuid);

            Component component = FindComponentByPath(workPart, componentPath);
            if (component == null)
            {
                logger.Warning($"FindPointByKey: Could not find component at path: {componentPath}");
                return null;
            }

            Part componentPart = component.Prototype as Part;
            if (componentPart == null)
                return null;

            if (!componentPart.IsFullyLoaded)
            {
                logger.Debug($"FindPointByKey: Fully loading: {componentPart.Name}");
                componentPart.LoadFully();
            }

            Point prototypePoint = FindPointInPart(componentPart, objectGuid);
            if (prototypePoint == null)
                return null;

            return component.FindOccurrence(prototypePoint) as Point ?? prototypePoint;
        }

        /// <summary>
        /// Finds multiple bodies by their composite keys.
        /// </summary>
        public static List<Body> FindBodiesByKeys(Part workPart, IEnumerable<string> compositeKeys)
        {
            var bodies = new List<Body>();
            if (workPart == null || compositeKeys == null)
                return bodies;

            foreach (string key in compositeKeys)
            {
                Body body = FindBodyByKey(workPart, key);
                if (body != null)
                    bodies.Add(body);
                else
                    logger.Warning($"Could not find body with key '{key}'");
            }
            return bodies;
        }

        #endregion

        #region Component Selection Support

        private const string COMPONENT_KEY_PREFIX = "COMPONENT:";

        /// <summary>
        /// Determines if a key represents a Component (vs a Body or other object).
        /// </summary>
        public static bool IsComponentKey(string key)
        {
            return !string.IsNullOrEmpty(key) && key.StartsWith(COMPONENT_KEY_PREFIX);
        }

        /// <summary>
        /// Gets or creates a persistent key for a Component.
        /// Format: "COMPONENT:componentPath"
        /// </summary>
        public static string GetOrCreateComponentKey(Component component)
        {
            if (component == null)
                return null;

            string componentPath = GetComponentPath(component);
            if (string.IsNullOrEmpty(componentPath))
                return null;

            return COMPONENT_KEY_PREFIX + componentPath;
        }

        /// <summary>
        /// Finds a Component by its key (must start with COMPONENT: prefix).
        /// </summary>
        public static Component FindComponentByKey(Part workPart, string key)
        {
            if (workPart == null || string.IsNullOrEmpty(key))
                return null;

            if (!IsComponentKey(key))
            {
                logger.Warning($"FindComponentByKey: Key '{key}' is not a component key");
                return null;
            }

            string componentPath = key.Substring(COMPONENT_KEY_PREFIX.Length);
            return FindComponentByPath(workPart, componentPath);
        }

        /// <summary>
        /// Gets or creates a persistent key for any selectable object (Body or Component).
        /// This is the unified entry point for selection persistence.
        /// </summary>
        public static string GetOrCreateSelectionKey(TaggedObject obj)
        {
            if (obj == null)
                return null;

            if (obj is Body body)
                return GetOrCreateBodyKey(body);

            if (obj is Component component)
                return GetOrCreateComponentKey(component);

            // Fallback for other NXObjects
            if (obj is NXObject nxObj)
                return GetOrCreateObjectKey(nxObj);

            return null;
        }

        /// <summary>
        /// Gets all solid bodies contained within a Component, recursively including sub-components.
        /// </summary>
        public static List<Body> GetBodiesFromComponent(Component component)
        {
            var bodies = new List<Body>();
            if (component == null)
                return bodies;

            try
            {
                // Get the part that this component is an instance of
                Part componentPart = component.Prototype as Part;
                if (componentPart != null)
                {
                    if (!componentPart.IsFullyLoaded)
                    {
                        componentPart.LoadFully();
                    }

                    // Get bodies from this component's part
                    foreach (Body body in componentPart.Bodies)
                    {
                        // Filter to solid bodies only (exclude sheet bodies)
                        if (body.IsSheetBody)
                            continue;

                        try
                        {
                            // Get the occurrence (instance) of this body in the assembly context
                            Body occurrence = component.FindOccurrence(body) as Body;
                            if (occurrence != null)
                            {
                                bodies.Add(occurrence);
                            }
                        }
                        catch (Exception ex)
                        {
                            logger.Debug($"GetBodiesFromComponent: Could not get occurrence for body: {ex.Message}");
                            bodies.Add(body);
                        }
                    }
                }

                // Recursively process child components (sub-assemblies)
                Component[] children = component.GetChildren();
                if (children != null)
                {
                    foreach (Component child in children)
                    {
                        bodies.AddRange(GetBodiesFromComponent(child));
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"GetBodiesFromComponent: Error processing component '{component.Name}': {ex.Message}");
            }

            if (bodies.Count == 0)
            {
                logger.Warning($"GetBodiesFromComponent: Component '{component.Name}' contains no solid bodies");
            }

            return bodies;
        }

        /// <summary>
        /// Resolves keys to Body objects, expanding Component keys to their contained bodies.
        /// Use this for export/inertial calculations where we need actual Body objects.
        /// </summary>
        public static List<Body> ResolveKeysToBodiesExpandingComponents(Part workPart, IEnumerable<string> keys, bool onlyShown)
        {
            var bodies = new List<Body>();
            var seenTags = new HashSet<Tag>(); // For deduplication

            if (workPart == null || keys == null)
                return bodies;

            foreach (string key in keys)
            {
                if (string.IsNullOrEmpty(key))
                    continue;

                if (IsComponentKey(key))
                {
                    // Expand component to its contained bodies
                    Component component = FindComponentByKey(workPart, key);
                    if (component != null && !component.IsBlanked && !component.IsSuppressed)
                    {
                        List<Body> componentBodies = GetBodiesFromComponent(component);
                        foreach (Body body in componentBodies)
                        {
                            if (body != null && !seenTags.Contains(body.Tag))
                            {
                                if (onlyShown && body.IsBlanked)
                                {
                                    continue;
                                }
                                bodies.Add(body);
                                seenTags.Add(body.Tag);
                            }
                        }
                    }
                    else
                    {
                        logger.Warning($"ResolveKeysToBodiessExpandingComponents: Could not find component for key '{key}'");
                    }
                }
                else
                {
                    // Regular body key
                    Body body = FindBodyByKey(workPart, key);
                    if (body != null && !seenTags.Contains(body.Tag))
                    {
                        if (onlyShown && body.IsBlanked)
                        {
                            continue;
                        }
                        bodies.Add(body);
                        seenTags.Add(body.Tag);
                    }
                    else if (body == null)
                    {
                        logger.Warning($"ResolveKeysToBodiessExpandingComponents: Could not find body for key '{key}'");
                    }
                }
            }

            return bodies;
        }

        /// <summary>
        /// Resolves keys to TaggedObjects for UI display.
        /// Returns Component or Body objects as-is (no expansion).
        /// Use this for restoring UI selections where we want to show what the user originally selected.
        /// </summary>
        public static List<TaggedObject> ResolveKeysToObjectsForUI(Part workPart, IEnumerable<string> keys)
        {
            var objects = new List<TaggedObject>();

            if (workPart == null || keys == null)
                return objects;

            foreach (string key in keys)
            {
                if (string.IsNullOrEmpty(key))
                    continue;

                if (IsComponentKey(key))
                {
                    Component component = FindComponentByKey(workPart, key);
                    if (component != null)
                        objects.Add(component);
                    else
                        logger.Warning($"ResolveKeysToObjectsForUI: Could not find component for key '{key}'");
                }
                else
                {
                    Body body = FindBodyByKey(workPart, key);
                    if (body != null)
                        objects.Add(body);
                    else
                        logger.Warning($"ResolveKeysToObjectsForUI: Could not find body for key '{key}'");
                }
            }

            return objects;
        }

        #endregion

        #region Part-level search (find object by GUID within a single part)

        private static Body FindBodyInPart(Part part, string guid)
        {
            if (part == null || string.IsNullOrEmpty(guid))
                return null;

            try
            {
                // Search features first
                foreach (Feature feature in part.Features)
                {
                    if (HasMatchingGuid(feature, BODY_GUID_ATTR, guid))
                    {
                        Body body = GetBodyFromFeature(feature);
                        if (body != null)
                            return body;
                    }
                }

                // Search Bodies collection (STEP imports)
                foreach (Body body in part.Bodies)
                {
                    if (HasMatchingGuid(body, BODY_GUID_ATTR, guid))
                        return body;
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"Error finding body in part: {ex.Message}");
            }

            return null;
        }

        private static CartesianCoordinateSystem FindCsysInPart(Part part, string guid)
        {
            if (part == null || string.IsNullOrEmpty(guid))
                return null;

            try
            {
                // Search DatumCsys features
                foreach (Feature feature in part.Features)
                {
                    if (feature is DatumCsys datumCsys && HasMatchingGuid(feature, CSYS_GUID_ATTR, guid))
                        return GetCsysFromFeature(datumCsys);
                }

                // Search CoordinateSystems collection (STEP imports)
                foreach (CartesianCoordinateSystem csys in part.CoordinateSystems)
                {
                    if (HasMatchingGuid(csys, CSYS_GUID_ATTR, guid))
                        return csys;
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"Error finding csys in part: {ex.Message}");
            }

            return null;
        }

        private static DatumAxis FindAxisInPart(Part part, string guid)
        {
            if (part == null || string.IsNullOrEmpty(guid))
                return null;

            try
            {
                // Search DatumAxisFeature features
                foreach (Feature feature in part.Features)
                {
                    if (feature is DatumAxisFeature datumAxisFeature && HasMatchingGuid(feature, AXIS_GUID_ATTR, guid))
                        return datumAxisFeature.DatumAxis;
                }

                // Search Datums collection (STEP imports)
                foreach (DisplayableObject datum in part.Datums)
                {
                    if (datum is DatumAxis axis && HasMatchingGuid(axis, AXIS_GUID_ATTR, guid))
                        return axis;
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"Error finding axis in part: {ex.Message}");
            }

            return null;
        }

        private static Point FindPointInPart(Part part, string guid)
        {
            if (part == null || string.IsNullOrEmpty(guid))
                return null;

            try
            {
                foreach (Point point in part.Points)
                {
                    if (HasMatchingGuid(point, POINT_GUID_ATTR, guid))
                        return point;
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"Error finding point in part: {ex.Message}");
            }

            return null;
        }

        #endregion

        #region Private Helpers

        private static NXObject GetPrototype(NXObject obj)
        {
            if (obj == null)
                return null;

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
            if (obj == null)
                return null;

            try
            {
                if (obj.IsOccurrence)
                    return obj.OwningComponent;
            }
            catch { }

            return null;
        }

        public static Feature GetOwningFeature(NXObject obj)
        {
            if (obj == null)
                return null;

            try
            {
                ufSession.Modl.AskObjectFeat(obj.Tag, out Tag featureTag);
                if (featureTag != Tag.Null)
                    return NXOpen.Utilities.NXObjectManager.Get(featureTag) as Feature;
            }
            catch { }

            return null;
        }

        private static string GetOrCreateGuidOnTarget(NXObject target, string attributeName)
        {
            if (target == null)
                return null;

            try
            {
                if (target.HasUserAttribute(attributeName, NXObject.AttributeType.String, -1))
                {
                    string existingGuid = target.GetStringUserAttribute(attributeName, -1);
                    if (!string.IsNullOrEmpty(existingGuid))
                        return existingGuid;
                }

                string newGuid = Guid.NewGuid().ToString();
                target.SetUserAttribute(attributeName, -1, newGuid, Update.Option.Now);
                logger.Debug($"GetOrCreateGuidOnTarget: Set GUID '{newGuid}' on {target.GetType().Name} tag={target.Tag}");

                return newGuid;
            }
            catch (Exception ex)
            {
                logger.Warning($"Failed to get/create GUID: {ex.Message}");
                return null;
            }
        }

        private static bool HasMatchingGuid(NXObject obj, string attributeName, string guid)
        {
            if (obj == null || string.IsNullOrEmpty(guid))
                return false;

            try
            {
                if (obj.HasUserAttribute(attributeName, NXObject.AttributeType.String, -1))
                {
                    string objGuid = obj.GetStringUserAttribute(attributeName, -1);
                    return string.Equals(objGuid, guid, StringComparison.OrdinalIgnoreCase);
                }
            }
            catch { }

            return false;
        }

        private static Body GetBodyFromFeature(Feature feature)
        {
            if (feature == null)
                return null;

            try
            {
                if (feature is BodyFeature bodyFeature)
                {
                    Body[] bodies = bodyFeature.GetBodies();
                    if (bodies?.Length > 0)
                        return bodies[0];
                }

                NXObject[] entities = feature.GetEntities();
                foreach (NXObject entity in entities ?? Array.Empty<NXObject>())
                {
                    if (entity is Body body)
                        return body;
                }
            }
            catch { }

            return null;
        }

        private static CartesianCoordinateSystem GetCsysFromFeature(DatumCsys datumCsys)
        {
            if (datumCsys == null)
                return null;

            try
            {
                NXObject[] entities = datumCsys.GetEntities();
                foreach (NXObject entity in entities ?? Array.Empty<NXObject>())
                {
                    if (entity is CartesianCoordinateSystem csys)
                        return csys;
                }
            }
            catch { }

            return null;
        }

        #endregion
    }
}

#endif
