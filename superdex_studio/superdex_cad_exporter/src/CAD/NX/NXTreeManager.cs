/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using NXOpen;
using NXOpen.BlockStyler;
using CADRobotExporter.RobotDescription;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace NXRobotExporter.CAD.NX
{
    /// <summary>
    /// Manages the NX TreeList control for the robot link hierarchy.
    /// The NX Tree is the single source of truth - Link objects are built on demand for export.
    /// </summary>
    public class NXTreeManager
    {
        private readonly Tree _tree;
        private readonly Dictionary<string, int> _linkCounters;

        public List<NXLinkNode> SelectedNodes { get; private set; }

        public event Action<List<NXLinkNode>> OnNodesSelected;
        public event Action<NXLinkNode> OnNodeCreated;

        public NXTreeManager(Tree treeControl)
        {
            _tree = treeControl ?? throw new ArgumentNullException(nameof(treeControl));
            _linkCounters = new Dictionary<string, int>();
        }

        #region Initialization

        /// <summary>
        /// Sets up the tree columns for link/joint display.
        /// </summary>
        public void InitializeColumns()
        {
            _tree.InsertColumn(NXLinkNode.ColumnLink, "Link", 200);
            _tree.InsertColumn(NXLinkNode.ColumnJointType, "Joint Type", 80);
            _tree.InsertColumn(NXLinkNode.ColumnCSYS, "CSYS", 40);
            _tree.InsertColumn(NXLinkNode.ColumnAxis, "Axis", 40);
            _tree.InsertColumn(NXLinkNode.ColumnInertial, "Ine", 40);
            _tree.InsertColumn(NXLinkNode.ColumnCollision, "Col", 40);
            _tree.InsertColumn(NXLinkNode.ColumnVisual, "Vis", 40);
        }

        /// <summary>
        /// Creates the root (base_link) node and initializes the tree.
        /// </summary>
        public NXLinkNode CreateRootNode(string name = "base_link")
        {
            var treeNode = _tree.CreateNode(name);
            _tree.InsertNode(treeNode, null, null, Tree.NodeInsertOption.Last);

            var node = new NXLinkNode(treeNode);
            node.InitializeDataContainer();  // Pre-initialize keys to avoid exception overhead
            node.LinkName = name;
            node.IsBaseLink = true;
            node.JointType = "fixed";
            node.UpdateAllColumns();  // Update display columns

            OnNodeCreated?.Invoke(node);
            return node;
        }

        #endregion

        #region Node Retrieval

        /// <summary>
        /// Gets the root node of the tree.
        /// </summary>
        public NXLinkNode GetRootNode()
        {
            var rootTreeNode = _tree.RootNode;
            return rootTreeNode != null ? new NXLinkNode(rootTreeNode) : null;
        }

        /// <summary>
        /// Wraps an NX Tree Node in an NXLinkNode.
        /// </summary>
        public NXLinkNode GetNode(Node treeNode)
        {
            return treeNode != null ? new NXLinkNode(treeNode) : null;
        }

        /// <summary>
        /// Gets the parent of a node.
        /// </summary>
        public NXLinkNode GetParent(NXLinkNode node)
        {
            var parentTreeNode = node?.TreeNode?.ParentNode;
            return parentTreeNode != null ? new NXLinkNode(parentTreeNode) : null;
        }

        /// <summary>
        /// Gets the children of a node.
        /// </summary>
        public static List<NXLinkNode> GetChildren(NXLinkNode node)
        {
            var children = new List<NXLinkNode>();
            if (node?.TreeNode == null) return children;

            var child = node.TreeNode.FirstChildNode;
            while (child != null)
            {
                children.Add(new NXLinkNode(child));
                child = child.NextSiblingNode;
            }
            return children;
        }

        public static List<Node> GetChildren(Node node)
        {
            var children = new List<Node>();

            var child = node.FirstChildNode;
            while (child != null)
            {
                children.Add(child);
                child = child.NextSiblingNode;
            }
            return children;
        }

        #endregion

        #region Node Creation

        /// <summary>
        /// Creates a new child node under the specified parent.
        /// </summary>
        public NXLinkNode CreateChildNode(NXLinkNode parent)
        {
            if (parent == null) throw new ArgumentNullException(nameof(parent));

            string linkName = GenerateLinkName(parent);
            string jointName = $"joint_{linkName}";

            var treeNode = _tree.CreateNode(linkName);
            _tree.InsertNode(treeNode, parent.TreeNode, null, Tree.NodeInsertOption.Last);

            var node = new NXLinkNode(treeNode);
            node.InitializeDataContainer();  // Pre-initialize keys to avoid exception overhead
            node.LinkName = linkName;
            node.JointName = jointName;
            node.JointType = "revolute";
            node.IsBaseLink = false;
            node.UpdateAllColumns();  // Update display columns

            OnNodeCreated?.Invoke(node);
            return node;
        }

        /// <summary>
        /// Creates a new site child node under the specified parent.
        /// Sites only need a CSYS - no joint name, axis, or bodies.
        /// </summary>
        public NXLinkNode CreateChildSiteNode(NXLinkNode parent)
        {
            if (parent == null) throw new ArgumentNullException(nameof(parent));

            int nextNumber = GetNextNumber("site");
            string linkName = $"site_{nextNumber}";

            var treeNode = _tree.CreateNode(linkName);
            _tree.InsertNode(treeNode, parent.TreeNode, null, Tree.NodeInsertOption.Last);

            var node = new NXLinkNode(treeNode);
            node.InitializeDataContainer();
            node.LinkName = linkName;
            node.JointName = "";
            node.JointType = "fixed";
            node.IsBaseLink = false;
            node.IsSite = true;
            node.UpdateAllColumns();

            OnNodeCreated?.Invoke(node);
            return node;
        }

        /// <summary>
        /// Creates multiple child nodes in a chain.
        /// </summary>
        public List<NXLinkNode> CreateSerialChain(NXLinkNode startNode, int count)
        {
            var nodes = new List<NXLinkNode>();
            var current = startNode;

            for (int i = 0; i < count; i++)
            {
                current = CreateChildNode(current);
                nodes.Add(current);
            }

            return nodes;
        }

        public NXLinkNode InsertRootNode(NXLinkNode node)
        {
            if (node == null || !node.IsRootNode)
            {
                return null;
            }

            if (node.JointName == "")
            {
                node.JointName = "joint_" + node.LinkName;
            }

            var newRootNode = CreateRootNode();
            MoveNodeUnder(node.TreeNode, newRootNode.TreeNode);

            return newRootNode;
        }

        /// <summary>
        /// Replaces the root node with an existing node from the tree.
        /// The promoted node becomes the new root, and the old root becomes its child.
        /// Children of the promoted node are moved to its former parent.
        /// </summary>
        public NXLinkNode ReplaceRootNode(NXLinkNode nodeToPromote, NXLinkNode currentRoot)
        {
            if (nodeToPromote == null || currentRoot == null) return null;
            if (!currentRoot.IsRootNode || nodeToPromote.IsRootNode) return null;

            // Remember the promoted node's parent (where its children will go)
            var promotedParentTreeNode = nodeToPromote.TreeNode.ParentNode;

            // Move the promoted node's children to its current parent
            var promotedChildren = GetChildren(nodeToPromote.TreeNode);
            var previousSibling = nodeToPromote.TreeNode.PreviousSiblingNode;
            foreach (var child in promotedChildren)
            {
                if (previousSibling == null)
                {
                    var copiedChild = _tree.CopyNode(child);
                    _tree.InsertNode(copiedChild, promotedParentTreeNode, null, Tree.NodeInsertOption.First);
                    MoveChildren(child, copiedChild);
                    UpdateColumnsRecursive(copiedChild);
                    previousSibling = copiedChild;
                    _tree.DeleteNode(child);
                }
                else
                {
                    var copiedChild = _tree.CopyNode(child);
                    _tree.InsertNode(copiedChild, promotedParentTreeNode, previousSibling, Tree.NodeInsertOption.Last);
                    MoveChildren(child, copiedChild);
                    UpdateColumnsRecursive(copiedChild);
                    previousSibling = copiedChild;
                    _tree.DeleteNode(child);
                }
            }

            // Ensure old root gets a joint name
            if (string.IsNullOrEmpty(currentRoot.JointName))
            {
                currentRoot.JointName = "joint_" + currentRoot.LinkName;
            }

            // Capture properties from the promoted node before deleting it
            string promotedLinkName = nodeToPromote.LinkName;
            string promotedCSYS = nodeToPromote.CoordinateSystemHandle;
            string[] promotedVisual = nodeToPromote.VisualBodiesHandles;
            string[] promotedCollision = nodeToPromote.CollisionBodiesHandles;
            string[] promotedInertial = nodeToPromote.InertialBodiesHandles;
            double promotedVisualMeshLinear = nodeToPromote.VisualMeshLinear;
            double promotedVisualMeshAngular = nodeToPromote.VisualMeshAngular;
            double promotedVisualMeshScale = nodeToPromote.VisualMeshScale;
            double promotedCollisionMeshLinear = nodeToPromote.CollisionMeshLinear;
            double promotedCollisionMeshAngular = nodeToPromote.CollisionMeshAngular;
            double promotedCollisionMeshScale = nodeToPromote.CollisionMeshScale;
            bool promotedPureVisual = nodeToPromote.PureVisual;
            bool promotedPureInertial = nodeToPromote.PureInertial;

            // Delete the promoted node from its original position
            _tree.DeleteNode(nodeToPromote.TreeNode);

            // Create new root with the promoted node's name and data
            var newRootNode = CreateRootNode(promotedLinkName);
            newRootNode.CoordinateSystemHandle = promotedCSYS;
            newRootNode.VisualBodiesHandles = promotedVisual;
            newRootNode.CollisionBodiesHandles = promotedCollision;
            newRootNode.InertialBodiesHandles = promotedInertial;
            newRootNode.VisualMeshLinear = promotedVisualMeshLinear;
            newRootNode.VisualMeshAngular = promotedVisualMeshAngular;
            newRootNode.VisualMeshScale = promotedVisualMeshScale;
            newRootNode.CollisionMeshLinear = promotedCollisionMeshLinear;
            newRootNode.CollisionMeshAngular = promotedCollisionMeshAngular;
            newRootNode.CollisionMeshScale = promotedCollisionMeshScale;
            newRootNode.PureVisual = promotedPureVisual;
            newRootNode.PureInertial = promotedPureInertial;

            // Move old root's children directly under the new root
            var oldRootChildren = GetChildren(currentRoot.TreeNode);
            foreach (var child in oldRootChildren)
            {
                MoveNodeUnder(child, newRootNode.TreeNode);
            }

            // Move old root (now childless) under new root as a sibling of its former children
            MoveNodeUnder(currentRoot.TreeNode, newRootNode.TreeNode);

            newRootNode.UpdateAllColumns();
            return newRootNode;
        }

        /// <summary>
        /// Inserts a new parent node between a node and its current parent.
        /// </summary>
        public NXLinkNode InsertParentNode(NXLinkNode node)
        {
            if (node == null || node.IsRootNode) return null;

            var oldParentTreeNode = node.TreeNode.ParentNode;
            if (oldParentTreeNode == null) return null;

            var oldParent = new NXLinkNode(oldParentTreeNode);

            // Generate name for new intermediate node
            string linkName = GenerateLinkName(oldParent);
            string jointName = $"joint_{linkName}";

            // Remember original node's position
            var previousSibling = node.TreeNode.PreviousSiblingNode;

            // Create the new parent node at original node's position
            var newParentTreeNode = _tree.CreateNode(linkName);
            if (previousSibling == null)
            {
                _tree.InsertNode(newParentTreeNode, oldParentTreeNode, null, Tree.NodeInsertOption.First);
            }
            else
            {
                _tree.InsertNode(newParentTreeNode, oldParentTreeNode, previousSibling, Tree.NodeInsertOption.Last);
            }

            var newParent = new NXLinkNode(newParentTreeNode);
            newParent.InitializeDataContainer();  // Pre-initialize keys to avoid exception overhead
            newParent.LinkName = linkName;
            newParent.JointName = jointName;
            newParent.JointType = "revolute";
            newParent.IsBaseLink = false;
            newParent.UpdateAllColumns();  // Update display columns

            // Move original node under new parent
            MoveNodeUnder(node.TreeNode, newParentTreeNode);

            OnNodeCreated?.Invoke(newParent);
            return newParent;
        }

        #endregion

        #region Node Movement (Drag & Drop)

        /// <summary>
        /// Reparents a node based on drop operation.
        /// Returns the new node after the move (original is deleted).
        /// </summary>
        public NXLinkNode ReparentNode(NXLinkNode node, NXLinkNode target, Node.DropType dropType)
        {
            if (node == null || target == null) return null;
            if (node.IsRootNode) return null;

            // Prevent dropping a node onto itself or its descendants
            if (IsDescendantOf(target.TreeNode, node.TreeNode)) return null;

            Node newTreeNode = null;
            switch (dropType)
            {
                case Node.DropType.On:
                    // Make node a child of target
                    newTreeNode = MoveNodeUnder(node.TreeNode, target.TreeNode);
                    break;

                case Node.DropType.Before:
                    // Make node a sibling before target
                    newTreeNode = MoveNodeBefore(node.TreeNode, target.TreeNode);
                    break;

                case Node.DropType.After:
                case Node.DropType.BeforeAndAfter:
                    // Make node a sibling after target
                    newTreeNode = MoveNodeAfter(node.TreeNode, target.TreeNode);
                    break;

                default:
                    return null;
            }

            if (newTreeNode != null)
            {
                ExpandNodesRecursively(newTreeNode);
            }

            return newTreeNode != null ? new NXLinkNode(newTreeNode) : null;
        }

        /// <summary>
        /// Moves a node to be a child of a new parent.
        /// Returns the new tree node.
        /// </summary>
        private Node MoveNodeUnder(Node nodeToMove, Node newParent)
        {
            // NX Tree requires copy/insert/delete pattern
            var copiedNode = _tree.CopyNode(nodeToMove);
            _tree.InsertNode(copiedNode, newParent, null, Tree.NodeInsertOption.Last);

            // Move all children recursively
            MoveChildren(nodeToMove, copiedNode);

            // Update columns for the moved node and all descendants
            UpdateColumnsRecursive(copiedNode);

            // Delete original
            _tree.DeleteNode(nodeToMove);

            return copiedNode;
        }

        /// <summary>
        /// Moves a node to be a sibling before the target.
        /// Returns the new tree node.
        /// </summary>
        private Node MoveNodeBefore(Node nodeToMove, Node target)
        {
            var targetParent = target.ParentNode;
            var previousSibling = target.PreviousSiblingNode;

            var copiedNode = _tree.CopyNode(nodeToMove);

            if (previousSibling == null)
            {
                _tree.InsertNode(copiedNode, targetParent, null, Tree.NodeInsertOption.First);
            }
            else
            {
                _tree.InsertNode(copiedNode, targetParent, previousSibling, Tree.NodeInsertOption.Last);
            }

            MoveChildren(nodeToMove, copiedNode);

            // Update columns for the moved node and all descendants
            UpdateColumnsRecursive(copiedNode);

            _tree.DeleteNode(nodeToMove);

            return copiedNode;
        }

        /// <summary>
        /// Moves a node to be a sibling after the target.
        /// Returns the new tree node.
        /// </summary>
        private Node MoveNodeAfter(Node nodeToMove, Node target)
        {
            var targetParent = target.ParentNode;

            var copiedNode = _tree.CopyNode(nodeToMove);
            _tree.InsertNode(copiedNode, targetParent, target, Tree.NodeInsertOption.Last);

            MoveChildren(nodeToMove, copiedNode);

            // Update columns for the moved node and all descendants
            UpdateColumnsRecursive(copiedNode);

            _tree.DeleteNode(nodeToMove);

            return copiedNode;
        }

        /// <summary>
        /// Recursively updates display columns for a node and all its descendants.
        /// </summary>
        private void UpdateColumnsRecursive(Node treeNode)
        {
            var node = new NXLinkNode(treeNode);
            node.UpdateAllColumns();

            var child = treeNode.FirstChildNode;
            while (child != null)
            {
                UpdateColumnsRecursive(child);
                child = child.NextSiblingNode;
            }
        }

        /// <summary>
        /// Recursively moves children from old node to new node.
        /// </summary>
        private void MoveChildren(Node oldParent, Node newParent)
        {
            // Collect children first (they'll be modified during iteration)
            var children = new List<Node>();
            var child = oldParent.FirstChildNode;
            while (child != null)
            {
                children.Add(child);
                child = child.NextSiblingNode;
            }

            // Move each child
            Node previousCopied = null;
            foreach (var childNode in children)
            {
                var copiedChild = _tree.CopyNode(childNode);

                if (previousCopied == null)
                {
                    _tree.InsertNode(copiedChild, newParent, null, Tree.NodeInsertOption.First);
                }
                else
                {
                    _tree.InsertNode(copiedChild, newParent, previousCopied, Tree.NodeInsertOption.Last);
                }

                // Recursively handle grandchildren
                MoveChildren(childNode, copiedChild);

                previousCopied = copiedChild;
            }
        }

        /// <summary>
        /// Checks if potentialDescendant is a descendant of potentialAncestor.
        /// </summary>
        private bool IsDescendantOf(Node potentialDescendant, Node potentialAncestor)
        {
            var current = potentialDescendant;
            while (current != null)
            {
                if (current == potentialAncestor) return true;
                current = current.ParentNode;
            }
            return false;
        }

        public static void ExpandNodesRecursively(NXLinkNode node)
        {
            if (node == null)
                return;

            node.TreeNode.Expand(Node.ExpandOption.Expand);

            foreach (var child in GetChildren(node))
            {
                ExpandNodesRecursively(child);
            }
        }

        public void ExpandNodesRecursively(Node node)
        {
            node.Expand(Node.ExpandOption.Expand);

            foreach (var child in GetChildren(node))
            {
                ExpandNodesRecursively(child);
            }
        }


        #endregion

        #region Selection

        /// <summary>
        /// Handles node selection.
        /// </summary>
        public void UpdateSelectedNodes()
        {
            var treeNodes = _tree.GetSelectedNodes();

            if (SelectedNodes == null)
            {
                SelectedNodes = new List<NXLinkNode>();
            }

            // Remove nodes that are no longer selected in the tree
            SelectedNodes.RemoveAll(node => !treeNodes.Contains(node.TreeNode));

            // Add newly selected tree nodes that aren't already tracked
            foreach (var treeNode in treeNodes)
            {
                if (!SelectedNodes.Any(node => node.TreeNode == treeNode))
                {
                    SelectedNodes.Add(new NXLinkNode(treeNode));
                }
            }

            OnNodesSelected?.Invoke(SelectedNodes);
        }

        #endregion

        #region Validation

        /// <summary>
        /// Finds the first incomplete node in the tree.
        /// </summary>
        public NXLinkNode FindFirstIncompleteNode()
        {
            var root = GetRootNode();
            return root != null ? FindIncompleteRecursive(root.TreeNode) : null;
        }

        private NXLinkNode FindIncompleteRecursive(Node treeNode)
        {
            var node = new NXLinkNode(treeNode);
            if (node.IsIncomplete) return node;

            var child = treeNode.FirstChildNode;
            while (child != null)
            {
                var result = FindIncompleteRecursive(child);
                if (result != null) return result;
                child = child.NextSiblingNode;
            }

            return null;
        }

        /// <summary>
        /// Gets all nodes in the tree.
        /// </summary>
        public List<NXLinkNode> CollectAllNodes()
        {
            var result = new List<NXLinkNode>();
            var root = GetRootNode();
            if (root != null)
            {
                CollectAllNodesRecursive(root.TreeNode, result);
            }
            return result;
        }

        private void CollectAllNodesRecursive(Node treeNode, List<NXLinkNode> result)
        {
            var linkNode = GetNode(treeNode);
            if (linkNode != null)
                result.Add(linkNode);
            Node child = treeNode.FirstChildNode;
            while (child != null)
            {
                CollectAllNodesRecursive(child, result);
                child = child.NextSiblingNode;
            }
        }

        /// <summary>
        /// Gets all incomplete nodes.
        /// </summary>
        public List<NXLinkNode> GetIncompleteNodes()
        {
            var result = new List<NXLinkNode>();
            var root = GetRootNode();
            if (root != null)
            {
                CollectIncompleteNodes(root.TreeNode, result);
            }
            return result;
        }

        /// <summary>
        /// Gets all site nodes that have children (invalid state).
        /// </summary>
        public List<NXLinkNode> GetSitesWithChildren()
        {
            var result = new List<NXLinkNode>();
            var root = GetRootNode();
            if (root != null)
            {
                CollectSitesWithChildren(root.TreeNode, result);
            }
            return result;
        }

        private void CollectSitesWithChildren(Node treeNode, List<NXLinkNode> result)
        {
            var node = new NXLinkNode(treeNode);
            if (node.IsSite && treeNode.FirstChildNode != null)
            {
                result.Add(node);
            }

            var child = treeNode.FirstChildNode;
            while (child != null)
            {
                CollectSitesWithChildren(child, result);
                child = child.NextSiblingNode;
            }
        }

        private void CollectIncompleteNodes(Node treeNode, List<NXLinkNode> result)
        {
            var node = new NXLinkNode(treeNode);
            if (node.IsIncomplete) result.Add(node);

            var child = treeNode.FirstChildNode;
            while (child != null)
            {
                CollectIncompleteNodes(child, result);
                child = child.NextSiblingNode;
            }
        }

        /// <summary>
        /// Checks for duplicate link names.
        /// </summary>
        public bool CheckLinkNamesUnique(out List<string> duplicates)
        {
            duplicates = new List<string>();
            var seen = new HashSet<string>();
            var root = GetRootNode();

            if (root != null)
            {
                CheckNamesRecursive(root.TreeNode, seen, duplicates, n => new NXLinkNode(n).LinkName);
            }

            return duplicates.Count == 0;
        }

        /// <summary>
        /// Checks for duplicate joint names.
        /// </summary>
        public bool CheckJointNamesUnique(out List<string> duplicates)
        {
            duplicates = new List<string>();
            var seen = new HashSet<string>();
            var root = GetRootNode();

            if (root != null)
            {
                CheckNamesRecursive(root.TreeNode, seen, duplicates, n =>
                {
                    var node = new NXLinkNode(n);
                    return node.IsBaseLink ? null : node.JointName;
                });
            }

            return duplicates.Count == 0;
        }

        private void CheckNamesRecursive(Node treeNode, HashSet<string> seen, List<string> duplicates, Func<Node, string> getName)
        {
            var name = getName(treeNode);
            if (!string.IsNullOrEmpty(name) && !seen.Add(name))
            {
                duplicates.Add(name);
            }

            var child = treeNode.FirstChildNode;
            while (child != null)
            {
                CheckNamesRecursive(child, seen, duplicates, getName);
                child = child.NextSiblingNode;
            }
        }

        #endregion

        #region Link Tree Building (for export)

        /// <summary>
        /// Builds a complete Link tree from the current NX Tree state.
        /// Call this when you need to export - this is the only time Link objects are created.
        /// </summary>
        public Link BuildLinkTree()
        {
            var rootTreeNode = _tree.RootNode;
            if (rootTreeNode == null) return null;

            return BuildLinkRecursive(rootTreeNode, null);
        }

        private Link BuildLinkRecursive(Node treeNode, Link parent)
        {
            var node = new NXLinkNode(treeNode);
            var link = node.ToLink();
            link.Parent = parent;

            var child = treeNode.FirstChildNode;
            while (child != null)
            {
                var childLink = BuildLinkRecursive(child, link);
                link.Children.Add(childLink);
                child = child.NextSiblingNode;
            }

            return link;
        }

        /// <summary>
        /// Populates the tree from an existing Link hierarchy.
        /// </summary>
        public NXLinkNode PopulateFromLink(Link rootLink)
        {
            if (rootLink == null) throw new ArgumentNullException(nameof(rootLink));

            _linkCounters.Clear();

            if (_tree.RootNode != null)
            {
                _tree.DeleteNode(_tree.RootNode);
            }

            var rootNode = CreateNodeFromLink(rootLink, null);
            SyncCountersFromTree();

            return rootNode;
        }

        private NXLinkNode CreateNodeFromLink(Link link, Node parentTreeNode)
        {
            var treeNode = _tree.CreateNode(link.Name);
            _tree.InsertNode(treeNode, parentTreeNode, null, Tree.NodeInsertOption.Last);

            var node = new NXLinkNode(treeNode);
            node.InitializeDataContainer();  // Pre-initialize keys to avoid exception overhead
            node.FromLink(link);
            node.UpdateAllColumns();  // Update display columns

            foreach (var childLink in link.Children)
            {
                CreateNodeFromLink(childLink, treeNode);
            }

            OnNodeCreated?.Invoke(node);
            return node;
        }

        #endregion

        #region Naming

        private string GenerateLinkName(NXLinkNode parent)
        {
            string prefix;

            var firstChild = parent.TreeNode.FirstChildNode;
            if (firstChild != null)
            {
                prefix = ExtractPrefix(new NXLinkNode(firstChild).LinkName);
            }
            else
            {
                prefix = parent.IsBaseLink ? "link" : ExtractPrefix(parent.LinkName);
            }

            int nextNumber = GetNextNumber(prefix);
            return $"{prefix}_{nextNumber}";
        }

        private static string ExtractPrefix(string name)
        {
            if (string.IsNullOrEmpty(name)) return "link";

            var parts = name.Split('_');
            if (parts.Length > 1 && int.TryParse(parts[parts.Length - 1], out _))
            {
                return string.Join("_", parts, 0, parts.Length - 1);
            }

            return name;
        }

        private int GetNextNumber(string prefix)
        {
            if (!_linkCounters.ContainsKey(prefix))
                _linkCounters[prefix] = 0;

            return ++_linkCounters[prefix];
        }

        /// <summary>
        /// Syncs the link counters with existing tree names.
        /// </summary>
        public void SyncCountersFromTree()
        {
            _linkCounters.Clear();
            var root = _tree.RootNode;
            if (root != null)
            {
                SyncCountersRecursive(root);
            }
        }

        private void SyncCountersRecursive(Node treeNode)
        {
            var name = new NXLinkNode(treeNode).LinkName;
            var parts = name.Split('_');

            if (parts.Length > 1 && int.TryParse(parts[parts.Length - 1], out int number))
            {
                string prefix = string.Join("_", parts, 0, parts.Length - 1);
                if (!_linkCounters.ContainsKey(prefix))
                    _linkCounters[prefix] = number;
                else
                    _linkCounters[prefix] = Math.Max(_linkCounters[prefix], number);
            }

            var child = treeNode.FirstChildNode;
            while (child != null)
            {
                SyncCountersRecursive(child);
                child = child.NextSiblingNode;
            }
        }

        #endregion
    }
}
