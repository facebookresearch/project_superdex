/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;

using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

using NXOpen;
using NXOpen.Features;
using NXOpen.UF;

using CADRobotExporter.RobotDescription;
using CADRobotExporter.UI;
using CADRobotExporter.Utilities;

namespace CADRobotExporter.CAD.NX
{
    /// <summary>
    /// Class to serialize URDF trees to string so they can be saved to an NX Custom Feature
    /// in the part. The feature appears in the Part Navigator and persists with the part file.
    /// </summary>
    public static class NXConfigurationSerialization
    {
        private static readonly Serilog.ILogger logger = Logger.GetLogger();

        /// <summary>
        /// Current Serialization version
        /// </summary>
        public const double SerializationVersion = 1.0;

        /// <summary>
        /// The class name for our custom feature
        /// </summary>
        public const string CustomFeatureClassName = "NXOpen::CustomFeature::RobotExporterConfiguration";

        /// <summary>
        /// Default feature name prefix
        /// </summary>
        public const string DefaultFeatureName = "Robot Configuration";

        /// <summary>
        /// Attribute names for the custom feature data
        /// </summary>
        private const string AttrData = "ConfigurationData";
        private const string AttrRobotName = "RobotName";
        private const string AttrDate = "SaveDate";
        private const string AttrVersion = "ExporterVersion";
        private const string AttrExporterConfig = "ExporterConfiguration";
        private const string AttrTendonData = "TendonData";

        private static Session session => Session.GetSession();
        private static UFSession ufSession => UFSession.GetUFSession();

        /// <summary>
        /// Saves the robot configuration to a custom feature in the part.
        /// Creates a new feature if one doesn't exist, or updates the existing one.
        /// </summary>
        /// <param name="workPart">The NX Part to save to</param>
        /// <param name="baseLink">The root link of the robot tree</param>
        /// <param name="exportConfig">The exporter configuration</param>
        /// <param name="existingFeature">Optional existing feature to update</param>
        /// <param name="featureName">Optional custom feature name</param>
        /// <returns>The created or updated custom feature</returns>
        public static CustomFeature SaveConfiguration(
            Part workPart,
            Link baseLink,
            ExporterConfiguration exportConfig,
            CustomFeature existingFeature = null,
            string featureName = null,
            List<Tendon> tendons = null)
        {
            if (workPart == null)
                throw new ArgumentNullException(nameof(workPart));

            string serializedData = SerializeLinkToString(baseLink);
            if (baseLink != null && string.IsNullOrEmpty(serializedData))
            {
                logger.Error("Serialization failed - returning null");
                return null;
            }

            string exporterConfigJson = SerializeExporterConfig(exportConfig);
            string tendonDataXml = SerializeTendonsToString(tendons);
            string robotName = exportConfig?.robotName ?? "Robot";

            if (string.IsNullOrEmpty(featureName))
            {
                featureName = $"{DefaultFeatureName} ({robotName})";
            }

            featureName = GetUniqueFeatureName(workPart, featureName, existingFeature);

            CustomFeature feature = SaveDataToFeature(
                workPart,
                serializedData,
                robotName,
                exporterConfigJson,
                tendonDataXml,
                existingFeature,
                featureName);

            return feature;
        }

        /// <summary>
        /// Loads a robot configuration from the selected custom feature.
        /// </summary>
        /// <param name="feature">The custom feature containing configuration data</param>
        /// <param name="exportConfig">Output: the loaded exporter configuration</param>
        /// <returns>The deserialized root Link, or null if loading failed</returns>
        public static Link LoadConfiguration(CustomFeature feature, out ExporterConfiguration exportConfig)
        {
            return LoadConfiguration(feature, out exportConfig, out _);
        }

        public static Link LoadConfiguration(CustomFeature feature, out ExporterConfiguration exportConfig, out List<Tendon> tendons)
        {
            exportConfig = new ExporterConfiguration();
            tendons = new List<Tendon>();

            if (feature == null)
            {
                logger.Warning("No feature provided to LoadConfiguration");
                return null;
            }

            try
            {
                CustomFeatureData cfData = feature.FeatureData;
                if (cfData == null)
                {
                    logger.Warning("Feature has no CustomFeatureData");
                    return null;
                }

                // Get the version
                double version = GetDoubleAttribute(cfData, AttrVersion, 0.0);
                if (version > SerializationVersion)
                {
                    logger.Warning($"Configuration version {version} is newer than supported {SerializationVersion}");
                }

                // Get the serialized link data
                string serializedData = GetStringAttribute(cfData, AttrData, "");
                if (string.IsNullOrEmpty(serializedData))
                {
                    logger.Warning("No configuration data found in feature");
                    return null;
                }

                // Deserialize the link tree
                Link baseLink = DeserializeLinkFromString(serializedData);

                // Get exporter configuration
                string exporterConfigJson = GetStringAttribute(cfData, AttrExporterConfig, "");
                if (!string.IsNullOrEmpty(exporterConfigJson))
                {
                    // Migrate previous name
                    exporterConfigJson = exporterConfigJson.Replace(
                        "\"folderStructure\":\"Mochi\"",
                        "\"folderStructure\":\"SuperDex\"");
                    exportConfig = DeserializeExporterConfig(exporterConfigJson);
                }
                else
                {
                    exportConfig.robotName = GetStringAttribute(cfData, AttrRobotName, "Robot");
                }

                // Get tendon data
                string tendonDataXml = GetStringAttribute(cfData, AttrTendonData, "");
                if (!string.IsNullOrEmpty(tendonDataXml))
                {
                    tendons = DeserializeTendonsFromString(tendonDataXml);
                }

                return baseLink;
            }
            catch (Exception ex)
            {
                logger.Error($"Error loading configuration: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Finds all Robot Configuration features in the part.
        /// </summary>
        /// <param name="workPart">The part to search</param>
        /// <returns>List of custom features that are robot configurations</returns>
        public static List<CustomFeature> FindConfigurationFeatures(Part workPart)
        {
            var results = new List<CustomFeature>();

            if (workPart == null)
                return results;

            try
            {
                FeatureCollection features = workPart.Features;
                foreach (Feature feature in features)
                {
                    if (feature is CustomFeature customFeature)
                    {
                        // Check if this is one of our features by looking for our attributes
                        try
                        {
                            CustomFeatureData cfData = customFeature.FeatureData;
                            if (cfData != null)
                            {
                                // Try to get our version attribute - if it exists, this is our feature
                                CustomDoubleAttribute versionAttr = cfData.CustomDoubleAttributeByName(AttrVersion);
                                if (versionAttr != null)
                                {
                                    results.Add(customFeature);
                                }
                            }
                        }
                        catch
                        {
                            // Not our feature, continue
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Error($"Error finding configuration features: {ex.Message}");
            }

            return results;
        }

        /// <summary>
        /// Gets the robot name from a configuration feature without fully deserializing.
        /// </summary>
        public static string GetRobotNameFromFeature(CustomFeature feature)
        {
            if (feature == null)
                return null;

            try
            {
                CustomFeatureData cfData = feature.FeatureData;
                if (cfData != null)
                {
                    return GetStringAttribute(cfData, AttrRobotName, "Robot");
                }
            }
            catch { }

            return null;
        }

        private static string SerializeTendonsToString(List<Tendon> tendons)
        {
            if (tendons == null || tendons.Count == 0)
                return "";

            using (MemoryStream stream = new MemoryStream())
            {
                var ser = new DataContractSerializer(typeof(List<Tendon>));
                try
                {
                    ser.WriteObject(stream, tendons);
                    stream.Flush();
                    string result = Encoding.UTF8.GetString(stream.GetBuffer(), 0, (int)stream.Position);
                    logger.Information($"Serialized {tendons.Count} tendon(s), data length: {result.Length}");
                    return result;
                }
                catch (SerializationException e)
                {
                    logger.Error($"Tendon serialization failed: {e.Message}");
                }
            }
            return "";
        }

        private static List<Tendon> DeserializeTendonsFromString(string data)
        {
            if (string.IsNullOrWhiteSpace(data))
                return new List<Tendon>();

            logger.Information($"Deserializing tendon data, length: {data.Length}");

            using (MemoryStream stream = new MemoryStream(Encoding.UTF8.GetBytes(data)))
            {
                var ser = new DataContractSerializer(typeof(List<Tendon>));
                try
                {
                    var result = (List<Tendon>)ser.ReadObject(stream) ?? new List<Tendon>();
                    logger.Information($"Deserialized {result.Count} tendon(s)");
                    return result;
                }
                catch (SerializationException e)
                {
                    logger.Error($"Tendon deserialization failed: {e.Message}");
                }
            }
            return new List<Tendon>();
        }

        /// <summary>
        /// Serializes a Link tree to an XML string using DataContractSerializer.
        /// </summary>
        private static string SerializeLinkToString(Link link)
        {
            if (link == null)
                return "";

            string data = "";
            using (MemoryStream stream = new MemoryStream())
            {
                DataContractSerializer ser = new DataContractSerializer(typeof(Link));

                try
                {
                    ser.WriteObject(stream, link);
                    stream.Flush();
                    data = Encoding.UTF8.GetString(stream.GetBuffer(), 0, (int)stream.Position);
                }
                catch (SerializationException e)
                {
                    logger.Error($"Serialization failed: {e.Message}");
                }
            }
            return data;
        }

        /// <summary>
        /// Deserializes a Link tree from an XML string.
        /// </summary>
        private static Link DeserializeLinkFromString(string data)
        {
            if (string.IsNullOrWhiteSpace(data))
                return null;

            // Migrate old serialized data: URDFElement was renamed to RobotElement
            // The base class hierarchy changed so we need to update the XML element names
            string migratedData = MigrateSerializedData(data);

            using (MemoryStream stream = new MemoryStream(Encoding.UTF8.GetBytes(migratedData)))
            {
                DataContractSerializer ser = new DataContractSerializer(typeof(Link));

                try
                {
                    Link link = (Link)ser.ReadObject(stream);
                    // Clone to ensure all non-serialized properties are set up correctly
                    return link?.Clone();
                }
                catch (SerializationException e)
                {
                    logger.Error($"Deserialization failed: {e.Message}");
                    logger.Error(migratedData);
                }
            }
            return null;
        }

        /// <summary>
        /// Migrates old serialized XML data to be compatible with the current class hierarchy.
        /// Replaces URDFElement references with RobotElement for backward compatibility.
        /// </summary>
        /// <param name="data">The serialized XML string to migrate</param>
        /// <returns>Migrated XML string compatible with current serialization format</returns>
        private static string MigrateSerializedData(string data)
        {
            if (string.IsNullOrEmpty(data))
            {
                return data;
            }

            // Replace URDFElement with RobotElement in both opening and closing tags
            // This handles the inheritance change from URDFElement to RobotElement base class
            string migratedData = data
                .Replace("<URDFElement", "<RobotElement")
                .Replace("</URDFElement>", "</RobotElement>")
                .Replace(
                    "http://schemas.datacontract.org/2004/07/SW2URDF.UI",
                    "http://schemas.datacontract.org/2004/07/CADRobotExporter.Export")
                .Replace(
                    "http://schemas.datacontract.org/2004/07/SW2URDF.Export",
                    "http://schemas.datacontract.org/2004/07/CADRobotExporter.Export");

            return migratedData;
        }

        /// <summary>
        /// Serializes the ExporterConfiguration to JSON.
        /// </summary>
        private static string SerializeExporterConfig(ExporterConfiguration config)
        {
            if (config == null)
                return "";

            JsonSerializerSettings settings = new JsonSerializerSettings()
            {
                ObjectCreationHandling = ObjectCreationHandling.Replace,
                Converters = new List<JsonConverter> { new StringEnumConverter() },
            };

            return JsonConvert.SerializeObject(config, settings);
        }

        /// <summary>
        /// Deserializes the ExporterConfiguration from JSON.
        /// </summary>
        private static ExporterConfiguration DeserializeExporterConfig(string json)
        {
            if (string.IsNullOrEmpty(json))
                return new ExporterConfiguration();

            JsonSerializerSettings settings = new JsonSerializerSettings()
            {
                ObjectCreationHandling = ObjectCreationHandling.Replace,
                Converters = new List<JsonConverter> { new StringEnumConverter() },
            };

            try
            {
                return JsonConvert.DeserializeObject<ExporterConfiguration>(json, settings);
            }
            catch (JsonException)
            {
                logger.Error("Could not parse exporter configuration JSON");
                return new ExporterConfiguration();
            }
        }

        /// <summary>
        /// Saves data to a custom feature, creating or updating as needed.
        /// </summary>
        private static CustomFeature SaveDataToFeature(
            Part workPart,
            string serializedData,
            string robotName,
            string exporterConfigJson,
            string tendonDataXml,
            CustomFeature existingFeature,
            string featureName)
        {
            CustomFeatureBuilder builder = null;

            try
            {
                CustomFeatureClassManager mgr = session.CustomFeatureClassManager;

                // Get or create our custom feature class
                CustomFeatureClass cfClass = GetOrCreateFeatureClass(mgr);
                if (cfClass == null)
                {
                    logger.Error("Failed to get or create custom feature class");
                    return null;
                }

                // Create the builder
                builder = workPart.Features.CreateCustomFeatureBuilder(existingFeature);

                CustomFeatureData cfData;

                if (existingFeature == null)
                {
                    var attrs = CreateMissingFeatureAttributes(workPart, null);
                    cfData = workPart.Features.CustomFeatureDataCollection.CreateData(cfClass, attrs.ToArray());
                }
                else
                {
                    cfData = existingFeature.FeatureData;
                    var missing = CreateMissingFeatureAttributes(workPart, cfData);
                    if (missing.Count > 0)
                    {
                        cfData.AddCustomAttributes(missing.ToArray());
                        logger.Information($"Added {missing.Count} missing attribute(s) to existing feature");
                    }
                }

                // Populate attribute values
                SetStringAttribute(cfData, AttrData, serializedData);
                SetStringAttribute(cfData, AttrRobotName, robotName);
                SetStringAttribute(cfData, AttrDate, DateTime.Now.ToString("o"));
                SetDoubleAttribute(cfData, AttrVersion, SerializationVersion);
                SetStringAttribute(cfData, AttrExporterConfig, exporterConfigJson);
                SetStringAttribute(cfData, AttrTendonData, tendonDataXml ?? "");

                // Set the data and commit
                builder.FeatureData = cfData;
                Feature feature = builder.CommitFeature();

                // Set feature name
                if (feature != null && !string.IsNullOrEmpty(featureName))
                {
                    feature.SetName(featureName);
                }

                builder.Destroy();
                builder = null;

                return feature as CustomFeature;
            }
            catch (Exception ex)
            {
                logger.Error($"Error saving data to feature: {ex.Message}");
                if (builder != null)
                {
                    builder.Destroy();
                }
                return null;
            }
        }

        /// <summary>
        /// Gets the custom feature class, or creates it if it doesn't exist.
        /// </summary>
        private static CustomFeatureClass GetOrCreateFeatureClass(CustomFeatureClassManager mgr)
        {
            try
            {
                // Try to get existing class
                return mgr.GetClassFromName(CustomFeatureClassName);
            }
            catch
            {
                // Class doesn't exist - for now we'll return null and let the caller handle it
                // In a full implementation, you'd register the class via a .udf XML file
                // or create it programmatically if the API supports it
                logger.Warning($"Custom feature class '{CustomFeatureClassName}' not found. " +
                    "Check your $UGII_USER_DIR/application for a CustomFeatureConfiguration.xml file");
                return null;
            }
        }

        /// <summary>
        /// Creates feature attributes that don't already exist on the given feature data.
        /// If cfData is null (new feature), creates all attributes.
        /// </summary>
        private static List<CustomAttribute> CreateMissingFeatureAttributes(Part workPart, CustomFeatureData cfData)
        {
            var existingSet = new HashSet<string>();
            if (cfData != null)
            {
                cfData.GetAllCustomAttributeNameAndTypes(out string[] existingNames, out CustomAttribute.Type[] existingTypes);
                if (existingNames != null)
                {
                    foreach (var name in existingNames)
                        existingSet.Add(name);
                }
            }

            var attrs = new List<CustomAttribute>();
            CustomAttributeCollection attribCollection = workPart.Features.CustomAttributeCollection;
            var properties = new List<CustomAttribute.Property>();

            if (!existingSet.Contains(AttrData))
                attrs.Add(attribCollection.CreateCustomStringAttribute(AttrData, properties.ToArray()));
            if (!existingSet.Contains(AttrRobotName))
                attrs.Add(attribCollection.CreateCustomStringAttribute(AttrRobotName, properties.ToArray()));
            if (!existingSet.Contains(AttrDate))
                attrs.Add(attribCollection.CreateCustomStringAttribute(AttrDate, properties.ToArray()));
            if (!existingSet.Contains(AttrVersion))
                attrs.Add(attribCollection.CreateCustomDoubleAttribute(AttrVersion, properties.ToArray()));
            if (!existingSet.Contains(AttrExporterConfig))
                attrs.Add(attribCollection.CreateCustomStringAttribute(AttrExporterConfig, properties.ToArray()));
            if (!existingSet.Contains(AttrTendonData))
                attrs.Add(attribCollection.CreateCustomStringAttribute(AttrTendonData, properties.ToArray()));

            return attrs;
        }

        private static string GetStringAttribute(CustomFeatureData cfData, string name, string defaultValue)
        {
            try
            {
                CustomStringAttribute attr = cfData.CustomStringAttributeByName(name);
                return attr?.Value ?? defaultValue;
            }
            catch (NXOpen.NXException)
            {
                return defaultValue;
            }
        }

        private static void SetStringAttribute(CustomFeatureData cfData, string name, string value)
        {
            try
            {
                CustomStringAttribute attr = cfData.CustomStringAttributeByName(name);
                if (attr != null)
                {
                    attr.Value = value ?? "";
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"Could not set string attribute {name}: {ex.Message}");
            }
        }

        private static double GetDoubleAttribute(CustomFeatureData cfData, string name, double defaultValue)
        {
            try
            {
                CustomDoubleAttribute attr = cfData.CustomDoubleAttributeByName(name);
                return attr?.Value ?? defaultValue;
            }
            catch
            {
                return defaultValue;
            }
        }

        private static void SetDoubleAttribute(CustomFeatureData cfData, string name, double value)
        {
            try
            {
                CustomDoubleAttribute attr = cfData.CustomDoubleAttributeByName(name);
                if (attr != null)
                {
                    attr.Value = value;
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"Could not set double attribute {name}: {ex.Message}");
            }
        }

        /// <summary>
        /// Generates a unique feature name that doesn't conflict with existing features.
        /// </summary>
        public static string GetUniqueFeatureName(Part workPart, string baseName, CustomFeature currentFeature)
        {
            return GetUniqueName(baseName, name => DoesFeatureNameExist(workPart, currentFeature, name));
        }

        /// <summary>
        /// Gets the raw serialized string data without saving to a feature.
        /// Useful for exporting configuration to files.
        /// </summary>
        /// <returns>True if serialization succeeded, false otherwise</returns>
        public static bool GetRawStringData(
            Link baseLink,
            ExporterConfiguration exportConfig,
            List<Tendon> tendons,
            out string robotName,
            out string urdfConfiguration,
            out string exporterConfiguration,
            out string tendonData)
        {
            robotName = exportConfig?.robotName ?? "Robot";
            urdfConfiguration = "";
            exporterConfiguration = "";
            tendonData = "";

            try
            {
                urdfConfiguration = SerializeLinkToString(baseLink);
                if (baseLink != null && string.IsNullOrEmpty(urdfConfiguration))
                {
                    logger.Error("URDF serialization failed");
                    return false;
                }

                exporterConfiguration = SerializeExporterConfig(exportConfig);
                tendonData = SerializeTendonsToString(tendons);
                return true;
            }
            catch (Exception ex)
            {
                logger.Error($"GetRawStringData failed: {ex.Message}");
                return false;
            }
        }

        private static bool DoesFeatureNameExist(Part workPart, CustomFeature currentFeature, string featureName)
        {
            if (workPart == null)
                return false;

            try
            {
                foreach (Feature feature in workPart.Features)
                {
                    if (currentFeature != null && feature.Tag == currentFeature.Tag)
                        continue;

                    if (feature.Name == featureName)
                        return true;
                }
            }
            catch { }

            return false;
        }

        private static string GetUniqueName(string baseName, Func<string, bool> exists)
        {
            if (!exists(baseName))
                return baseName;

            // Check if the name already ends with a number pattern like " (1)"
            string coreName = baseName;
            int startNumber = 1;

            var match = System.Text.RegularExpressions.Regex.Match(baseName, @"^(.+) \((\d+)\)$");
            if (match.Success)
            {
                coreName = match.Groups[1].Value;
                startNumber = int.Parse(match.Groups[2].Value) + 1;
            }

            // Find the next available number
            int counter = startNumber;
            string newName;
            do
            {
                newName = $"{coreName} ({counter})";
                counter++;
            } while (exists(newName));

            return newName;
        }

        /// <summary>
        /// Gets the raw serialized string data from an existing custom feature.
        /// Useful for backup/export operations.
        /// </summary>
        /// <param name="feature">The custom feature containing configuration data</param>
        /// <param name="robotName">Output: the robot name</param>
        /// <param name="urdfConfiguration">Output: the serialized URDF configuration (XML)</param>
        /// <param name="exporterConfiguration">Output: the serialized exporter configuration (JSON)</param>
        /// <param name="tendonData">Output: the serialized tendon data (XML)</param>
        /// <returns>True if extraction succeeded, false otherwise</returns>
        public static bool GetRawStringDataFromFeature(
            CustomFeature feature,
            out string robotName,
            out string urdfConfiguration,
            out string exporterConfiguration,
            out string tendonData)
        {
            robotName = "";
            urdfConfiguration = "";
            exporterConfiguration = "";
            tendonData = "";

            if (feature == null)
            {
                logger.Warning("No feature provided to GetRawStringDataFromFeature");
                return false;
            }

            try
            {
                CustomFeatureData cfData = feature.FeatureData;
                if (cfData == null)
                {
                    logger.Warning("Feature has no CustomFeatureData");
                    return false;
                }

                robotName = GetStringAttribute(cfData, AttrRobotName, "Robot");
                urdfConfiguration = GetStringAttribute(cfData, AttrData, "");
                exporterConfiguration = GetStringAttribute(cfData, AttrExporterConfig, "");
                tendonData = GetStringAttribute(cfData, AttrTendonData, "");

                return true;
            }
            catch (Exception ex)
            {
                logger.Error($"Error extracting raw string data from feature: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Creates a new configuration custom feature from raw string data.
        /// Useful for import/duplicate operations.
        /// </summary>
        /// <param name="workPart">The NX Part to save to</param>
        /// <param name="robotName">The robot name</param>
        /// <param name="urdfConfiguration">The serialized URDF configuration (XML)</param>
        /// <param name="exporterConfiguration">The serialized exporter configuration (JSON)</param>
        /// <param name="tendonData">The serialized tendon data (XML)</param>
        /// <returns>The created custom feature, or null if creation failed</returns>
        public static CustomFeature CreateNewConfigurationFromRawData(
            Part workPart,
            string robotName,
            string urdfConfiguration,
            string exporterConfiguration,
            string tendonData = "")
        {
            if (workPart == null)
                throw new ArgumentNullException(nameof(workPart));

            if (string.IsNullOrEmpty(robotName))
            {
                robotName = "Robot";
            }

            string featureName = $"{DefaultFeatureName} ({robotName})";
            featureName = GetUniqueFeatureName(workPart, featureName, null);

            CustomFeature feature = SaveDataToFeature(
                workPart,
                urdfConfiguration,
                robotName,
                exporterConfiguration,
                tendonData ?? "",
                null,
                featureName);

            return feature;
        }
    }
}
