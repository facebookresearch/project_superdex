/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;

using Serilog;

using CADRobotExporter.Utilities;

namespace CADRobotExporter.Model
{
    /// <summary>
    /// Base class for all robot model elements.
    /// Provides hierarchical structure with attributes and child elements.
    /// This is a format-agnostic base class that can be serialized to URDF, MJCF, or other formats.
    /// </summary>
    [DataContract(IsReference = true, Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    [KnownType("GetKnownTypes")]
    public class RobotElement : IExtensibleDataObject
    {
        protected static readonly ILogger logger = Logger.GetLogger();

        [DataMember]
        protected readonly List<RobotElement> ChildElements;

        [DataMember]
        protected readonly List<ElementAttribute> Attributes;

        [DataMember]
        protected readonly string ElementName;

        [DataMember]
        protected bool required;

        private ExtensionDataObject ExtensionDataValue;

        public virtual ExtensionDataObject ExtensionData
        {
            get => ExtensionDataValue;
            set => ExtensionDataValue = value;
        }

        public RobotElement(string elementName, bool required)
        {
            ElementName = elementName;
            this.required = required;
            ChildElements = new List<RobotElement>();
            Attributes = new List<ElementAttribute>();
        }

        public bool IsRequired()
        {
            return required;
        }

        public virtual void SetRequired(bool required)
        {
            this.required = required;
        }

        public virtual bool AreRequiredFieldsSatisfied()
        {
            foreach (ElementAttribute attribute in Attributes)
            {
                if (attribute.GetIsRequired() && attribute.Value == null)
                {
                    return false;
                }
            }
            foreach (RobotElement child in ChildElements)
            {
                if (!child.AreRequiredFieldsSatisfied() &&
                    (child.IsRequired() || child.ElementContainsData()))
                {
                    return false;
                }
            }
            return true;
        }

        public virtual bool ElementContainsData()
        {
            foreach (ElementAttribute attribute in Attributes)
            {
                if (attribute.Value != null)
                {
                    return true;
                }
            }

            foreach (RobotElement child in ChildElements)
            {
                if (child.ElementContainsData())
                {
                    return true;
                }
            }
            return false;
        }

        public void Unset()
        {
            foreach (ElementAttribute attribute in Attributes)
            {
                attribute.Value = null;
            }
            foreach (RobotElement child in ChildElements)
            {
                child.Unset();
            }
        }

        public virtual void SetElement(RobotElement externalElement)
        {
            if (externalElement.GetType() != GetType())
            {
                throw new Exception("RobotElements need to be the same type to set the internal values");
            }

            foreach (Tuple<ElementAttribute, ElementAttribute> pair in
                Enumerable.Zip(Attributes, externalElement.Attributes, Tuple.Create))
            {
                if (pair.Item2.Value != null && pair.Item2.Value.GetType() == typeof(double[]))
                {
                    double[] valueArray = (double[])pair.Item2.Value;
                    pair.Item1.Value = valueArray.Clone();
                }
                else
                {
                    pair.Item1.Value = pair.Item2.Value;
                }
            }

            foreach (RobotElement myChild in ChildElements)
            {
                string myChildTypeName = myChild.GetType().Name;
                RobotElement matchingExternalChild = externalElement.ChildElements
                    .FirstOrDefault(e => e.GetType().Name == myChildTypeName);

                if (matchingExternalChild != null)
                {
                    myChild.SetElement(matchingExternalChild);
                }
            }
        }

        public virtual void AppendToCSVDictionary(List<string> context, OrderedDictionary dictionary)
        {
            string typeName = GetType().Name;
            List<string> updatedContext = new List<string>(context) { typeName };

            foreach (ElementAttribute att in Attributes)
            {
                if (att.Value != null)
                {
                    att.AppendToCSVDictionary(updatedContext, dictionary);
                }
            }

            foreach (RobotElement child in ChildElements)
            {
                child.AppendToCSVDictionary(updatedContext, dictionary);
            }
        }

        public virtual void SetElementFromData(List<string> context, StringDictionary dictionary)
        {
            string typeName = GetType().Name;
            List<string> updatedContext = new List<string>(context) { typeName };

            foreach (ElementAttribute att in Attributes)
            {
                att.SetValueFromData(updatedContext, dictionary);
            }

            foreach (RobotElement child in ChildElements)
            {
                child.SetElementFromData(updatedContext, dictionary);
            }
        }

        public static Type[] GetKnownTypes()
        {
            return new List<Type>(
                Assembly.GetExecutingAssembly().GetTypes().Where(_ => _.IsSubclassOf(typeof(RobotElement))))
            {
                typeof(double[])
            }.ToArray();
        }
    }
}
