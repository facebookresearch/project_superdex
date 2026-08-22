/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if SOLIDWORKS

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Text;
using System.Windows.Forms;

using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;
using Attribute = SolidWorks.Interop.sldworks.Attribute;

using CADRobotExporter.UI;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.Utilities;

namespace CADRobotExporter.CAD
{
    /// <summary>
    /// Class to serialize URDF trees to string so they can be saved to an SW Attribute in the
    /// top-level assembly document.
    ///
    /// Any changes to the serialization scheme need to support backwards compatibility in some way.
    /// At least in regards to reading the old configuration. I'm also choosing to save any old xml
    /// formats to a second attribute in case they need to be reloaded.
    /// </summary>
    public static class ConfigurationSerialization
    {
        private static readonly Serilog.ILogger logger = Logger.GetLogger();

        /// <summary>
        /// Current Serialization version
        /// </summary>
        private const double SerializationVersion = 1.4;

        /// <summary>
        /// Previous versions of serialization were set at 1 in the SW Document. This is
        /// used to denote the version at which the serialization scheme was modified.
        /// </summary>
        private const double MinDataContractVersion = 1.3;

        /// <summary>
        /// The name given to the URDF configuration in the ModelDoc Feature tree. This is displayed to the
        /// user
        /// </summary>
        public const string UrdfConfigurationSwAttributeNameV14 = "URDF Export Configuration (v1.4)";

        public const string UrdfAttributeName = "pubURDFAttributeV1";
        public const string UrdfCongfigurationFeatureName = "Robot Exporter Configuration";

        public static AttributeDef urdfAttributeDef;
        public static AttributeDef previousUrdfAttribute;

        public static List<string> PREVIOUS_URDF_CONFIGURATION_NAMES = new List<string>() {
            "URDF Export Configuration (v1.3)",
            "URDF Export Configuration"
            };

        public const string UrdfConficgurationParamName = "urdfConfiguration";

        public static void RegisterUrdfAttribute(SldWorks swApp)
        {
            int options = 0;

            urdfAttributeDef = swApp.DefineAttribute(UrdfAttributeName);

            urdfAttributeDef.AddParameter(
                "data", (int)swParamType_e.swParamTypeString, 0, options);
            urdfAttributeDef.AddParameter(
                "name", (int)swParamType_e.swParamTypeString, 0, options);
            urdfAttributeDef.AddParameter(
                "date", (int)swParamType_e.swParamTypeString, 0, options);
            urdfAttributeDef.AddParameter(
                "exporterVersion", (int)swParamType_e.swParamTypeDouble, SerializationVersion, options);
            urdfAttributeDef.AddParameter(
                "exporterConfiguration", (int)swParamType_e.swParamTypeString, 0, options);
            urdfAttributeDef.AddParameter(
                "tendonData", (int)swParamType_e.swParamTypeString, 0, options);

            urdfAttributeDef.Register();

            // Previous attribute for migration

            previousUrdfAttribute = swApp.DefineAttribute(UrdfConfigurationSwAttributeNameV14);

            previousUrdfAttribute.AddParameter(
                "data", (int)swParamType_e.swParamTypeString, 0, options);
            previousUrdfAttribute.AddParameter(
                "name", (int)swParamType_e.swParamTypeString, 0, options);
            previousUrdfAttribute.AddParameter(
                "date", (int)swParamType_e.swParamTypeString, 0, options);
            previousUrdfAttribute.AddParameter(
                "exporterVersion", (int)swParamType_e.swParamTypeDouble, SerializationVersion, options);

            previousUrdfAttribute.Register();
        }

        public static LinkNode LoadBaseNodeFromSelection(ModelDoc2 model, out bool error, out Feature exporterFeature)
        {
            string data = GetConfigTreeDataFromSelection(model, out double configVersion, out exporterFeature);

            if (string.IsNullOrEmpty(data))
            {
                error = true;
                return null;
            }

            if (configVersion > SerializationVersion)
            {
                MessageBox.Show("The configuration saved in this model is newer than what this " +
                    "exporter supports " + string.Format("({0} > {1})", configVersion, SerializationVersion) +
                    ". Please update your exporter version");
                error = true;
                return null;
            }

            LinkNode basenode = null;

            if (configVersion >= MinDataContractVersion)
            {
                basenode = DeserializeFromString(data);
            }

            error = false;
            return basenode;
        }

        public static bool LoadRawStringDataFromSelection(SldWorks swApp, out string robotName, out string xmlConfiguration, out string exporterConfiguration)
        {
            return LoadRawStringDataFromSelection(swApp, out robotName, out xmlConfiguration, out exporterConfiguration, out _);
        }

        public static bool LoadRawStringDataFromSelection(SldWorks swApp, out string robotName, out string xmlConfiguration, out string exporterConfiguration, out string tendonData)
        {
            Feature exporterFeature;
            robotName = "";
            exporterConfiguration = "";
            tendonData = "";

            xmlConfiguration = GetConfigTreeDataFromSelection(swApp.ActiveDoc, out double configVersion, out exporterFeature);

            if (exporterFeature == null)
            {
                return false;
            }

            ExporterConfiguration exportConfig = LoadExporterConfiguration(exporterFeature);

            robotName = exportConfig.robotName;

            JsonSerializerSettings settings = new JsonSerializerSettings()
            {
                ObjectCreationHandling = ObjectCreationHandling.Replace,
                Converters = new List<JsonConverter> { new StringEnumConverter() },
            };

            exporterConfiguration = JsonConvert.SerializeObject(exportConfig, settings);

            List<Tendon> tendons = LoadTendonData(exporterFeature);
            tendonData = SerializeTendons(tendons);

            return true;
        }

        public static bool GetRawStringData(Link rootNode, ExporterConfiguration exportConfig, out string robotName, out string xmlConfiguration, out string exporterConfiguration)
        {
            robotName = "";
            exporterConfiguration = "";
            xmlConfiguration = "";

            if (rootNode == null)
            {
                return false;
            }

            xmlConfiguration = SerializeLinkToString(rootNode);
            robotName = exportConfig.robotName;

            JsonSerializerSettings settings = new JsonSerializerSettings()
            {
                ObjectCreationHandling = ObjectCreationHandling.Replace,
                Converters = new List<JsonConverter> { new StringEnumConverter() },
            };

            exporterConfiguration = JsonConvert.SerializeObject(exportConfig, settings);

            return true;
        }

        public static bool CreateNewExporterFeatureFromRawData(SldWorks swApp, string robotName, string xmlConfiguration, string exporterConfiguration)
        {
            return CreateNewExporterFeatureFromRawData(swApp, robotName, xmlConfiguration, exporterConfiguration, "");
        }

        public static bool CreateNewExporterFeatureFromRawData(SldWorks swApp, string robotName, string xmlConfiguration, string exporterConfiguration, string tendonData)
        {
            if (string.IsNullOrEmpty(xmlConfiguration))
            {
                return false;
            }

            Feature exporterFeature = CreateExporterFeature(swApp.ActiveDoc, $"Robot Configuration ({robotName})");

            if (exporterFeature == null)
            {
                return false;
            }

            int ConfigurationOptions = (int)swInConfigurationOpts_e.swAllConfiguration;

            Attribute exporterAttribute = exporterFeature.GetSpecificFeature2();
            Parameter param = exporterAttribute.GetParameter("data");
            param.SetStringValue2(xmlConfiguration, ConfigurationOptions, "");
            param = exporterAttribute.GetParameter("name");
            param.SetStringValue2(UrdfConficgurationParamName, ConfigurationOptions, "");
            param = exporterAttribute.GetParameter("date");
            param.SetStringValue2(DateTime.Now.ToString(), ConfigurationOptions, "");
            param = exporterAttribute.GetParameter("exporterVersion");
            param.SetDoubleValue2(SerializationVersion, ConfigurationOptions, "");
            param = exporterAttribute.GetParameter("exporterConfiguration");
            param.SetStringValue2(exporterConfiguration, ConfigurationOptions, "");
            param = exporterAttribute.GetParameter("tendonData");
            param.SetStringValue2(tendonData ?? "", ConfigurationOptions, "");

            return true;
        }

        public static Feature SaveConfigTreeXML(ModelDoc2 model, LinkNode BaseNode, bool warnUser, Feature exporterFeature, string featureName)
        {
            string oldData = GetConfigTreeData(exporterFeature, out double version);
            string newData = SerializeToString(BaseNode);

            if (BaseNode != null && string.IsNullOrEmpty(newData))
            {
                MessageBox.Show("Serializing this link failed. Please Contact your maintainer with your SW assembly.");
                return null;
            }
            if (oldData != newData)
            {
                if (!warnUser ||
                    (warnUser &&
                    MessageBox.Show("The Robot Configuration has changed, would you like to save?",
                    "Save Robot Configuration", MessageBoxButtons.YesNo) == DialogResult.Yes))
                {
                    exporterFeature = SaveDataToModelDoc(model, newData, exporterFeature, featureName);
                }
            }

            if (!string.IsNullOrEmpty(featureName) && exporterFeature != null)
            {
                featureName = GetUniqueFeatureName(model, featureName, exporterFeature);
                exporterFeature.Name = featureName;
            }

            return exporterFeature;
        }

        /// <summary>
        /// If someone updates the name of a LinkNode in the Treeview, it needs to be pushed down
        /// to the URDF Link itself.
        /// </summary>
        /// <param name="node">TreeView LinkNode to save properties of to its URDF Link</param>
        private static void SavePropertiesLinkNodeToLink(LinkNode node)
        {
            if (node.Link == null)
            {
                node.Link = new Link();
                return;
            }

            node.Link.Name = node.Name;

            foreach (LinkNode child in node.Nodes)
            {
                SavePropertiesLinkNodeToLink(child);
            }
        }

        /// <summary>
        /// Data Contract serialization. All members of an object need to be annotated with a
        /// [DataMember] attribute.
        /// </summary>
        /// <param name="node">TreeView LinkNode to serialize</param>
        /// <returns>A string serialized utilizing DataContract serialization XML scheme</returns>
        private static string SerializeToString(LinkNode node)
        {
            SavePropertiesLinkNodeToLink(node);
            Link link = node.UpdateLinkTree(null);
            string data = "";
            using (MemoryStream stream = new MemoryStream())
            {
                DataContractSerializer ser =
                    new DataContractSerializer(typeof(Link));

                try
                {
                    ser.WriteObject(stream, link);
                    stream.Flush();
                    data = Encoding.ASCII.GetString(stream.GetBuffer(), 0, (int)stream.Position);
                }
                catch (SerializationException e)
                {
                    logger.Error("Serialization failed with exception, returning empty string", e);
                }
            }
            return data;
        }

        private static string SerializeLinkToString(Link link)
        {
            string data = "";
            using (MemoryStream stream = new MemoryStream())
            {
                DataContractSerializer ser =
                    new DataContractSerializer(typeof(Link));

                try
                {
                    ser.WriteObject(stream, link);
                    stream.Flush();
                    data = Encoding.ASCII.GetString(stream.GetBuffer(), 0, (int)stream.Position);
                }
                catch (SerializationException e)
                {
                    logger.Error("Serialization failed with exception, returning empty string", e);
                }
            }
            return data;
        }

        /// <summary>
        /// Read a URDF Link from a serialized string
        /// </summary>
        /// <param name="data">Data string to read into a TreeView LinkNode</param>
        /// <returns>Deserialized LinkNode</returns>
        private static LinkNode DeserializeFromString(string data)
        {
            LinkNode baseNode = null;
            if (!string.IsNullOrWhiteSpace(data))
            {
                // Migrate old serialized data: URDFElement was renamed to RobotElement
                // The base class hierarchy changed so we need to update the XML element names
                string migratedData = MigrateSerializedData(data);

                using (MemoryStream stream = new MemoryStream(Encoding.ASCII.GetBytes(migratedData)))
                {
                    DataContractSerializer ser =
                        new DataContractSerializer(typeof(Link));

                    try
                    {
                        Link link = (Link)ser.ReadObject(stream);

                        // By copying this link, we can ensure that all non-serialized properties are setup correctly
                        Link copy = link.Clone();
                        baseNode = new LinkNode(copy);
                    }
                    catch (SerializationException e)
                    {
                        logger.Error("Deserialization failed with exception, returning empty LinkNode", e);
                        logger.Error(migratedData);
                    }
                }
            }
            return baseNode;
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

        private static string GetConfigTreeDataFromSelection(ModelDoc2 model, out double version, out Feature exporterFeature)
        {
            string data = "";
            version = 0.0;
            exporterFeature = null;
            int numSelected = 0;

            if (model.SelectionManager.GetSelectedObjectCount2(-1) == 0)
            {
                MessageBox.Show("Nothing selected.\n\nPlease select the Robot Configuration you want to edit/export.");
                return "";
            }

            exporterFeature = GetFeatureWithURDFAttributeFromSelection(model, out numSelected);

            if (exporterFeature == null)
            {
                if (numSelected == 0)
                {
                    MessageBox.Show("No Robot Configurations selected.\n\nPlease select the Robot Configuration you want to edit/export.");
                }
                else
                {
                    MessageBox.Show("Found more than one Robot Configuration in the selection.\n\nPlease select only one.");
                }
                return "";
            }

            if (exporterFeature.Name == UrdfConfigurationSwAttributeNameV14)
            {
                exporterFeature = MigrateAttribute(model, exporterFeature);
            }

            Attribute exporterAttribute = exporterFeature.GetSpecificFeature2();
            Parameter param = exporterAttribute.GetParameter("data");
            data = param.GetStringValue();
            logger.Information("Robot Configuration found\n" + data);

            param = exporterAttribute.GetParameter("exporterVersion");
            version = param.GetDoubleValue();

            return data;
        }

        private static string GetConfigTreeData(Feature exporterFeature, out double version)
        {
            string data = "";
            version = 0.0;

            if (exporterFeature != null)
            {
                Attribute exporterAttribute = exporterFeature.GetSpecificFeature2();
                Parameter param = exporterAttribute.GetParameter("data");
                data = param.GetStringValue();
                logger.Information("Robot Configuration found\n" + data);

                param = exporterAttribute.GetParameter("exporterVersion");
                version = param.GetDoubleValue();
            }

            return data;
        }

        private static Feature GetFeatureWithURDFAttributeFromSelection(ModelDoc2 model, out int numSelected)
        {
            SelectionMgr selectionManager = model.SelectionManager;

            List<Feature> features = new List<Feature>();

            for (int i = 0; i < selectionManager.GetSelectedObjectCount2(-1); i++)
            {
                swSelectType_e selectType = (swSelectType_e)selectionManager.GetSelectedObjectType3(i + 1, -1);
                if (selectType == swSelectType_e.swSelATTRIBUTES)
                {
                    Feature feature = selectionManager.GetSelectedObject6(i + 1, -1);
                    Attribute attribute = (Attribute)feature.GetSpecificFeature2();
                    Parameter parameter = attribute.IGetParameter("name");
                    if (parameter != null && (parameter.GetStringValue() == "config1" || parameter.GetStringValue() == UrdfConficgurationParamName))
                    {
                        features.Add(feature);
                    }
                }
            }

            numSelected = features.Count;

            if (features.Count > 1)
            {
                return null;
            }

            if (features.Count == 1)
            {
                return features[0];
            }

            return null;
        }

        private static Feature GetFeatureAttributeByName(ModelDoc2 model, string featName)
        {
            Object[] objects = model.FeatureManager.GetFeatures(true);
            foreach (Object obj in objects)
            {
                Feature feature = (Feature)obj;
                if (feature.GetTypeName2() == "Attribute")
                {
                    Attribute att = (Attribute)feature.GetSpecificFeature2();
                    if (att.GetName() == featName)
                    {
                        return feature;
                    }
                }
            }
            return null;
        }

        public static bool DoesFeatureNameExist(ModelDoc2 model, Feature currentFeature, string featureName)
        {
            Feature feature = GetFeatureAttributeByName(model, featureName);

            if (currentFeature != null && feature == currentFeature)
            {
                return false;
            }

            if (feature != null)
            {
                return true;
            }

            return false;
        }

        public static string GetUniqueName(string baseName, Func<string, bool> exists)
        {
            // If the name doesn't exist, return it as-is
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

        public static string GetUniqueFeatureName(ModelDoc2 model, string featureName, Feature currentFeature)
        {
            return GetUniqueName(featureName, name => DoesFeatureNameExist(model, currentFeature, name));
        }

        private static Feature CreateExporterFeature(ModelDoc2 model, string featureName)
        {
            if (string.IsNullOrEmpty(featureName))
            {
                featureName = GetUniqueFeatureName(model, UrdfCongfigurationFeatureName, null);
            }
            else
            {
                featureName = GetUniqueFeatureName(model, featureName, null);
            }

            int options = 0;
            int configurationOptions = (int)swInConfigurationOpts_e.swAllConfiguration;

            Attribute exporterAttribute = urdfAttributeDef.CreateInstance5(model, null, featureName, options, configurationOptions);
            Feature exporterFeature = GetFeatureAttributeByName(model, featureName);

            return exporterFeature;
        }

        private static Feature MigrateAttribute(ModelDoc2 model, Feature previousFeature)
        {
            Attribute previousAttribute = previousFeature.GetSpecificFeature2();
            Parameter param;
            param = previousAttribute.GetParameter("data");
            string data = param.GetStringValue();
            param = previousAttribute.GetParameter("name");
            string name = param.GetStringValue();
            param = previousAttribute.GetParameter("date");
            string date = param.GetStringValue();

            bool deleted = previousAttribute.Delete(true);
            logger.Information("Deleted: " + deleted);

            int options = 0;
            int configurationOptions = (int)swInConfigurationOpts_e.swAllConfiguration;

            string featureName = GetUniqueFeatureName(model, UrdfCongfigurationFeatureName, null);

            Attribute newAttribute = urdfAttributeDef.CreateInstance5(
                                                                    model,
                                                                    null,
                                                                    featureName,
                                                                    options,
                                                                    configurationOptions);

            param = newAttribute.GetParameter("data");
            param.SetStringValue2(data, configurationOptions, "");
            param = newAttribute.GetParameter("name");
            param.SetStringValue2(UrdfConficgurationParamName, configurationOptions, "");
            param = newAttribute.GetParameter("date");
            param.SetStringValue2(date, configurationOptions, "");

            Feature newFeature = GetFeatureAttributeByName(model, featureName);

            return newFeature;
        }

        private static Feature SaveDataToModelDoc(ModelDoc2 model, string data, Feature exporterFeature, string featureName)
        {
            int ConfigurationOptions = (int)swInConfigurationOpts_e.swAllConfiguration;

            if (exporterFeature == null)
            {
                exporterFeature = CreateExporterFeature(model, featureName);
            }

            Attribute exporterAttribute = exporterFeature.GetSpecificFeature2();
            Parameter param = exporterAttribute.GetParameter("data");
            param.SetStringValue2(data, ConfigurationOptions, "");
            param = exporterAttribute.GetParameter("name");
            param.SetStringValue2(UrdfConficgurationParamName, ConfigurationOptions, "");
            param = exporterAttribute.GetParameter("date");
            param.SetStringValue2(DateTime.Now.ToString(), ConfigurationOptions, "");
            param = exporterAttribute.GetParameter("exporterVersion");
            param.SetDoubleValue2(SerializationVersion, ConfigurationOptions, "");

            if (!string.IsNullOrEmpty(featureName))
            {
                featureName = GetUniqueFeatureName(model, featureName, exporterFeature);
                exporterFeature.Name = featureName;
            }

            return exporterFeature;
        }

        public static void SaveExporterConfiguration(ExporterConfiguration config, Feature exporterFeature)
        {
            JsonSerializerSettings settings = new JsonSerializerSettings()
            {
                ObjectCreationHandling = ObjectCreationHandling.Replace,
                Converters = new List<JsonConverter> { new StringEnumConverter() },
            };

            string jsonString = JsonConvert.SerializeObject(config, settings);
            logger.Information("Saving json:");
            logger.Information(jsonString);

            int ConfigurationOptions = (int)swInConfigurationOpts_e.swAllConfiguration;

            Attribute exporterAttribute = exporterFeature.GetSpecificFeature2();
            Parameter param = exporterAttribute.GetParameter("exporterConfiguration");
            param.SetStringValue2(jsonString, ConfigurationOptions, "");
        }

        public static ExporterConfiguration LoadExporterConfiguration(Feature exporterFeature)
        {
            string jsonString = "";
            ExporterConfiguration config = null;
            JsonSerializerSettings settings = new JsonSerializerSettings()
            {
                ObjectCreationHandling = ObjectCreationHandling.Replace,
                Converters = new List<JsonConverter> { new StringEnumConverter() },
            };

            if (exporterFeature != null)
            {
                Attribute exporterAttribute = exporterFeature.GetSpecificFeature2();
                Parameter param = exporterAttribute.GetParameter("exporterConfiguration");

                if (param != null)
                {
                    jsonString = param.GetStringValue();
                    logger.Information("Exporter configuration found\n" + jsonString);
                }
                else
                {
                    logger.Information("Exporter configuration not found, using defaults");
                    return new ExporterConfiguration();
                }
            }

            try
            {
               config = JsonConvert.DeserializeObject<ExporterConfiguration>(jsonString, settings);
            }
            catch (JsonException)
            {
                logger.Error("Could not parse json!");
                config = new ExporterConfiguration();
            }

            return config;
        }

        public static string SerializeTendons(List<Tendon> tendons)
        {
            if (tendons == null || tendons.Count == 0)
                return "";

            string data = "";
            using (MemoryStream stream = new MemoryStream())
            {
                DataContractSerializer ser = new DataContractSerializer(
                    typeof(List<Tendon>),
                    CADRobotExporter.Model.RobotElement.GetKnownTypes());
                try
                {
                    ser.WriteObject(stream, tendons);
                    stream.Flush();
                    data = Encoding.ASCII.GetString(stream.GetBuffer(), 0, (int)stream.Position);
                }
                catch (SerializationException e)
                {
                    logger.Error("Tendon serialization failed", e);
                }
            }
            return data;
        }

        public static List<Tendon> DeserializeTendons(string data)
        {
            if (string.IsNullOrWhiteSpace(data))
                return new List<Tendon>();

            try
            {
                using (MemoryStream stream = new MemoryStream(Encoding.ASCII.GetBytes(data)))
                {
                    DataContractSerializer ser = new DataContractSerializer(
                        typeof(List<Tendon>),
                        CADRobotExporter.Model.RobotElement.GetKnownTypes());
                    return (List<Tendon>)ser.ReadObject(stream);
                }
            }
            catch (SerializationException e)
            {
                logger.Error("Tendon deserialization failed", e);
                return new List<Tendon>();
            }
        }

        public static void SaveTendonData(List<Tendon> tendons, Feature exporterFeature)
        {
            if (exporterFeature == null) return;

            string data = SerializeTendons(tendons);
            int configurationOptions = (int)swInConfigurationOpts_e.swAllConfiguration;

            Attribute exporterAttribute = exporterFeature.GetSpecificFeature2();
            Parameter param = exporterAttribute.GetParameter("tendonData");
            if (param != null)
            {
                param.SetStringValue2(data, configurationOptions, "");
            }
        }

        public static List<Tendon> LoadTendonData(Feature exporterFeature)
        {
            if (exporterFeature == null)
                return new List<Tendon>();

            Attribute exporterAttribute = exporterFeature.GetSpecificFeature2();
            Parameter param = exporterAttribute.GetParameter("tendonData");

            if (param == null)
                return new List<Tendon>();

            string data = param.GetStringValue();
            return DeserializeTendons(data);
        }
    }
}

#endif
