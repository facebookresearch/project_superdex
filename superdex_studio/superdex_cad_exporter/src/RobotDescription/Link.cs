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

using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Runtime.Serialization;

#if SOLIDWORKS
using SolidWorks.Interop.sldworks;
#endif

#if NX
using NXOpen;
#endif

using CADRobotExporter.Model;
using CADRobotExporter.Export;
using System;

namespace CADRobotExporter.RobotDescription
{
    //The link class, it contains many other elements not found in the URDF.
    [DataContract(IsReference = true, Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class Link : RobotElement, IExtensibleDataObject
    {
        public ExtensionDataObject ExtensionData { get; set; }

        [DataMember]
        public Link Parent;

        [DataMember]
        public List<Link> Children;

        [DataMember]
        private readonly ElementAttribute NameAttribute;

        public string Name
        {
            get => (string)NameAttribute.Value;
            set => NameAttribute.Value = value;
        }

        [DataMember]
        public Inertial Inertial;

        [DataMember]
        public Visual Visual;

        [DataMember]
        public Collision Collision;

        [DataMember]
        public Joint Joint;

        [DataMember]
        public bool STLQualityFine;

        [DataMember]
        public bool IsIncomplete;

        [DataMember]
        public bool isFixedFrame;

        [DataMember]
        public bool isSite;

        [DataMember]
        public bool shouldFlipAxis;

        [DataMember]
        public MeshingOptions visualMeshingOptions;

        [DataMember]
        public MeshingOptions collisionMeshingOptions;

        [DataMember]
        public bool exportCollision;

        [DataMember]
        public bool visualsOnly;

        [DataMember]
        public bool inertialsOnly;

        // Solidworks only components
        [DataMember]
        public List<byte[]> SWComponentPIDs;

        [DataMember]
        public List<byte[]> SWCollisionComponentPIDs;

        [DataMember]
        public List<byte[]> SWInertialComponentPIDs;

        // NX only components
        [DataMember]
        public List<string> NXVisualBodiesHandles;

        [DataMember]
        public List<string> NXCollisionBodiesHandles;

        [DataMember]
        public List<string> NXInertialBodiesHandles;

#if SOLIDWORKS
        public List<Component2> SWVisualComponents;
        public List<Component2> SWCollisionComponents;
        public List<Component2> SWInertialComponents;
#endif

        public enum ComponentType
        {
            Visual,
            Collision,
            Inertial
        }

        public Link() : base("link", true)
        {
            Parent = null;
            Children = new List<Link>();
            NameAttribute = new ElementAttribute("name", true, "");

            Inertial = new Inertial();
            Visual = new Visual();
            Collision = new Collision();
            Joint = new Joint();

            isFixedFrame = false;
            isSite = false;
            shouldFlipAxis = false;

            visualsOnly = false;
            inertialsOnly = false;

            visualMeshingOptions = new MeshingOptions
            {
                linearDeflection = 0.1,
                angularDeflection = 0.5,
                scale = 1.0
            };
            collisionMeshingOptions = new MeshingOptions
            {
                linearDeflection = 0.5,
                angularDeflection = 0.75,
                scale = 1.0
            };

            Attributes.Add(NameAttribute);
            ChildElements.Add(Inertial);
            ChildElements.Add(Visual);
            ChildElements.Add(Collision);
            ChildElements.Add(Joint);

            SWComponentPIDs = new List<byte[]>();
            SWCollisionComponentPIDs = new List<byte[]>();
            SWInertialComponentPIDs = new List<byte[]>();

#if SOLIDWORKS
            SWVisualComponents = new List<Component2>();
            SWCollisionComponents = new List<Component2>();
            SWInertialComponents = new List<Component2>();
#endif
            NXVisualBodiesHandles = new List<string>();
            NXCollisionBodiesHandles = new List<string>();
            NXInertialBodiesHandles = new List<string>();
        }

        public bool IsBaseLink { get; set; }

        public Link Clone()
        {
            Link cloned = new Link();
            cloned.SetElement(this);
            foreach (Link child in Children)
            {
                Link clonedChild = child.Clone();
                clonedChild.Parent = this;
                cloned.Children.Add(clonedChild);
            }
            return cloned;
        }

        public Link(Link parent) : base("link", true)
        {
            Parent = parent;
            Children = new List<Link>();
            NameAttribute = new ElementAttribute("name", true, "");

            Inertial = new Inertial();
            Visual = new Visual();
            Collision = new Collision();
            Joint = new Joint();

            isFixedFrame = false;
            isSite = false;
            shouldFlipAxis = false;

            visualsOnly = false;
            inertialsOnly = false;

            visualMeshingOptions = new MeshingOptions
            {
                linearDeflection = 0.1,
                angularDeflection = 0.5,
                scale = 1.0
            };
            collisionMeshingOptions = new MeshingOptions
            {
                linearDeflection = 0.5,
                angularDeflection = 0.75,
                scale = 1.0
            };

            Attributes.Add(NameAttribute);
            ChildElements.Add(Inertial);
            ChildElements.Add(Visual);
            ChildElements.Add(Collision);
            ChildElements.Add(Joint);

            SWComponentPIDs = new List<byte[]>();
            SWCollisionComponentPIDs = new List<byte[]>();
            SWInertialComponentPIDs = new List<byte[]>();

#if SOLIDWORKS
            SWVisualComponents = new List<Component2>();
            SWCollisionComponents = new List<Component2>();
            SWInertialComponents = new List<Component2>();
#endif

            NXVisualBodiesHandles = new List<string>();
            NXCollisionBodiesHandles = new List<string>();
            NXInertialBodiesHandles = new List<string>();
        }

        public void AddChild(Link child)
        {
            child.Parent = this;
            Children.Add(child);
        }

        public void RemoveChild(Link child)
        {
            child.Parent = null;
            Children.Remove(child);
        }

        public override void AppendToCSVDictionary(List<string> context, OrderedDictionary dictionary)
        {
            IEnumerable<string> componentNames = new List<string>();
#if SOLIDWORKS
            componentNames = SWVisualComponents.Select(component => component.Name2);
            string componentNamesStr = string.Join(";", componentNames);
            string componentsContext = "Link.SWComponents";
            dictionary.Add(componentsContext, componentNamesStr);
#endif

            base.AppendToCSVDictionary(context, dictionary);
        }

        public override void SetElement(RobotElement externalElement)
        {
            base.SetElement(externalElement);

            Link externalLink = (Link)externalElement;

            isFixedFrame = externalLink.isFixedFrame;
            isSite = externalLink.isSite;
            shouldFlipAxis = externalLink.shouldFlipAxis;

            visualsOnly = externalLink.visualsOnly;
            inertialsOnly = externalLink.inertialsOnly;

            visualMeshingOptions = externalLink.visualMeshingOptions;
            collisionMeshingOptions = externalLink.collisionMeshingOptions;

            if (visualMeshingOptions == null)
            {
                visualMeshingOptions = new MeshingOptions
                {
                    linearDeflection = 0.1,
                    angularDeflection = 0.5,
                    scale = 1.0
                };
            }

            if (collisionMeshingOptions == null)
            {
                collisionMeshingOptions = new MeshingOptions
                {
                    linearDeflection = 0.5,
                    angularDeflection = 0.75,
                    scale = 1.0
                };
            }

            // NX components
            // we don't conditionally compile here because these are just strings
            // and can be serialize in case we want to import back to Solidworks for some reason
            if (externalLink.NXVisualBodiesHandles != null)
            {
                NXVisualBodiesHandles = new List<string>(externalLink.NXVisualBodiesHandles);
            }
            if (externalLink.NXCollisionBodiesHandles != null)
            {
                NXCollisionBodiesHandles = new List<string>(externalLink.NXCollisionBodiesHandles);
            }
            if (externalLink.NXInertialBodiesHandles != null)
            {
                NXInertialBodiesHandles = new List<string>(externalLink.NXInertialBodiesHandles);
            }

            // Solidworks only function call
#if SOLIDWORKS
            SetSWComponents(externalLink);
#endif
        }

#if SOLIDWORKS
        public void SetSWComponents(Link externalLink)
        {
            if (externalLink.SWVisualComponents != null)
            {
                SWVisualComponents = new List<Component2>(externalLink.SWVisualComponents);
            }
            else
            {
                SWVisualComponents = new List<Component2>();
            }
            if (externalLink.SWComponentPIDs != null)
            {
                SWComponentPIDs = new List<byte[]>(externalLink.SWComponentPIDs);
            }
            else
            {
                SWComponentPIDs = new List<byte[]>();
            }
            if (externalLink.SWCollisionComponents != null)
            {
                SWCollisionComponents = new List<Component2>(externalLink.SWCollisionComponents);
            }
            else
            {
                SWCollisionComponents = new List<Component2>();
            }
            if (externalLink.SWCollisionComponentPIDs != null)
            {
                SWCollisionComponentPIDs = new List<byte[]>(externalLink.SWCollisionComponentPIDs);
            }
            else
            {
                SWCollisionComponentPIDs = new List<byte[]>();
            }
            if (externalLink.SWInertialComponents != null)
            {
                SWInertialComponents = new List<Component2>(externalLink.SWInertialComponents);
            }
            else
            {
                SWInertialComponents = new List<Component2>();
            }
            if (externalLink.SWInertialComponentPIDs != null)
            {
                SWInertialComponentPIDs = new List<byte[]>(externalLink.SWInertialComponentPIDs);
            }
            else
            {
                SWInertialComponentPIDs = new List<byte[]>();
            }
        }
#endif

        public string[] GetJointNames(bool includeFixed)
        {
            List<string> names = new List<string>();

            if (Joint != null && (includeFixed || Joint.Type != "fixed"))
            {
                names.Add(Joint.Name);
            }
            foreach (Link child in Children)
            {
                names.AddRange(child.GetJointNames(includeFixed));
            }

            return names.ToArray();
        }

        public override bool AreRequiredFieldsSatisfied()
        {
            if (!base.AreRequiredFieldsSatisfied())
            {
                return false;
            }

            foreach (Link child in Children)
            {
                if (!child.AreRequiredFieldsSatisfied())
                {
                    return false;
                }
            }

            return true;
        }

        public bool HasValidInertialProperties()
        {
            if (visualsOnly)
            {
                return true;
            }

            if (!HasVisualComponents() && !HasInertialComponents())
            {
                return true;
            }

            if (Inertial.Mass.Value <= 0.0)
            {
                return false;
            }

            if (Inertial.Inertia.Ixx == 0.0
                && Inertial.Inertia.Ixz == 0.0
                && Inertial.Inertia.Ixy == 0.0
                && Inertial.Inertia.Iyz == 0.0
                && Inertial.Inertia.Ixy == 0.0)
            {
                return false;
            }

            return true;
        }

#if SOLIDWORKS
        public List<Component2> GetComponents(ComponentType componentType)
        {
            switch (componentType)
            {
                case ComponentType.Visual:
                    return SWVisualComponents;
                case ComponentType.Collision:
                    return SWCollisionComponents;
                case ComponentType.Inertial:
                    return SWInertialComponents;
                default:
                    return null;
            }
        }
#endif
#if NX
        public List<string> GetComponents(ComponentType componentType)
        {
            switch (componentType)
            {
                case ComponentType.Visual:
                    return NXVisualBodiesHandles;
                case ComponentType.Collision:
                    return NXCollisionBodiesHandles;
                case ComponentType.Inertial:
                    return NXInertialBodiesHandles;
                default:
                    return null;
            }
        }
#endif

        public bool HasComponents(ComponentType componentType)
        {
#if SOLIDWORKS
            List<Component2> components = GetComponents(componentType);
#endif
#if NX
            List<string> components = GetComponents(componentType);
#endif
            return components != null && components.Count > 0;
        }

        public bool HasVisualComponents()
        {
            return HasComponents(ComponentType.Visual);
        }

        public bool HasCollisionComponents()
        {
            return HasComponents(ComponentType.Collision);
        }

        public bool HasInertialComponents()
        {
            return HasComponents(ComponentType.Inertial);
        }
    }
}
