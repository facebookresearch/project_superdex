# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import apply_linear_map
from superdex.physics.viewer.backend import polyscope as ps
from superdex.physics.viewer.renderers.renderer import Renderer

########################################################################################


class CurveNetworkRenderer(Renderer):
    """
    Renderer for displaying 3D line segments using Polyscope's CurveNetwork. Provides
    functionality to manage and render line segment geometries, including node
    coordinates, radii, and colors. The CurveNetworkRenderer supports both global and
    per-node/per-edge radius and color specifications. The class follows the same
    patterns as other renderers in the SuperDex Physics viewer, including stack-based property
    management for easy overrides.
    """

    ####################################################################################
    # Members
    ####################################################################################

    # Curve network geometry.
    _nodes: npt.NDArray[float]
    _edges: npt.NDArray[int]

    # Render settings (stack-based for easy overrides).
    _color: list[npt.NDArray[float]]
    _transparency: list[float]

    # Backing Polyscope render structure.
    _render_struct: ps.CurveNetwork | None
    _nodes_dirty: bool

    # Per-node radii (optional).
    _node_radii: npt.NDArray[float] | None
    _node_radii_dirty: bool

    # Per-node colors (optional).
    _node_colors: npt.NDArray[float] | None
    _node_colors_dirty: bool

    # Per-edge radii (optional).
    _edge_radii: npt.NDArray[float] | None
    _edge_radii_dirty: bool

    # Per-edge colors (optional).
    _edge_colors: npt.NDArray[float] | None
    _edge_colors_dirty: bool

    ####################################################################################
    # Constants
    ####################################################################################

    _TAG: str = "[CurveNetwork]"
    """Tag used to identify CurveNetworkRenderer-related render structures."""

    _DEFAULT_RADIUS: float = 0.005
    """Default radius for all edges."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        name: str,
        nodes: npt.NDArray[float] | npt.ArrayLike,
        edges: npt.NDArray[int] | npt.ArrayLike,
        coordinate_transform: CoordinateTransform,
        radius: float | None = None,
        node_radii: npt.NDArray[float] | npt.ArrayLike | None = None,
        edge_radii: npt.NDArray[float] | npt.ArrayLike | None = None,
        color: npt.NDArray[float] | npt.ArrayLike | None = None,
        node_colors: npt.NDArray[float] | npt.ArrayLike | None = None,
        edge_colors: npt.NDArray[float] | npt.ArrayLike | None = None,
        transparency: float | None = None,
    ):
        """Initialize the CurveNetworkRenderer with given parameters.

        Args:
            name: Name identifier for this renderer.
            nodes: Nx3 array of node coordinates in global space.
            edges: Mx2 array of edge indices, where each row specifies the indices
                of the two nodes that form an edge.
            coordinate_transform: Converter for coordinate system transformations.
            radius: Global radius for all edges. If not provided, a default value
                will be used. Ignored if per-node/per-edge radii are specified.
            node_radii: Optional N-element array of per-node radii. If provided,
                overrides the global radius for nodes.
            edge_radii: Optional M-element array of per-edge radii. If provided,
                overrides the global radius for edges.
            color: RGB color for all elements as a 3-element array. If not provided,
                a default color will be used. Ignored if per-node/per-edge colors
                are specified.
            node_colors: Optional Nx3 array of per-node RGB colors. If provided,
                overrides the global color for nodes.
            edge_colors: Optional Mx3 array of per-edge RGB colors. If provided,
                overrides the global color for edges.
            transparency: Transparency level (0.0 = opaque, 1.0 = fully transparent).
        """

        # Initialize base class.
        super().__init__(name, coordinate_transform)

        # Initialize rendering structure.
        self._render_struct = None
        self._node_radii = None
        self._node_colors = None
        self._edge_radii = None
        self._edge_colors = None
        self._color = []
        self._transparency = []
        self.set_curve_network(
            nodes, edges, node_radii, edge_radii, node_colors, edge_colors
        )
        self._create_render_structure()
        self.update()

        # Apply provided settings or defaults.
        if color is not None:
            self.set_color(color)
        if transparency is not None:
            self.set_transparency(transparency)

        # Set global radius (only used if per-node/per-edge radii are None).
        if radius is None:
            radius = self._DEFAULT_RADIUS
        self.set_radius(radius)

    ####################################################################################
    # General Management Methods
    ####################################################################################

    def _create_render_structure(self):
        """Creates the backing Polyscope render structure. Called internally during
        initialization and whenever the curve network topology changes."""

        tagged_name = f"{self._TAG} {self._name}"

        # Convert coordinates to target coordinate system.
        source_to_ps = self._coordinate_transform.source_to_target
        nodes_ps = apply_linear_map(source_to_ps[:3, :3], self._nodes)

        # Generate curve network structure.
        self._render_struct = ps.register_curve_network(
            name=tagged_name,
            nodes=nodes_ps,
            edges=self._edges,
        )

    @override_from(Renderer)
    def update(self):
        """Update the renderer's visualization if any properties have changed."""

        # Check if render structure is initialized.
        assert self._render_struct is not None, "Render structure not initialized."

        # Retrieve the change of basis matrix.
        source_to_ps = self._coordinate_transform.source_to_target

        # Update node positions if dirty.
        if self._nodes_dirty and self._render_struct.is_enabled():
            if self._render_struct.n_nodes() != self._nodes.shape[0]:
                self._create_render_structure()
            else:
                nodes_ps = apply_linear_map(source_to_ps[:3, :3], self._nodes)
                self._render_struct.update_node_positions(nodes_ps)
            self._nodes_dirty = False

        # Update per-node radii if dirty.
        if self._node_radii_dirty and self._render_struct.is_enabled():
            if self._node_radii is not None:
                self._render_struct.add_scalar_quantity(
                    "NodeRadii", self._node_radii, defined_on="nodes", enabled=False
                )
                self._render_struct.set_node_radius_quantity(
                    "NodeRadii", autoscale=False
                )
            else:
                # Clear per-node radii if set to None.
                self._render_struct.remove_quantity("NodeRadii", error_if_absent=False)
                self._render_struct.clear_node_radius_quantity()
            self._node_radii_dirty = False

        # Update per-node colors if dirty.
        if self._node_colors_dirty and self._render_struct.is_enabled():
            if self._node_colors is not None:
                self._render_struct.add_color_quantity(
                    "NodeColors", self._node_colors, defined_on="nodes", enabled=True
                )
            else:
                self._render_struct.remove_quantity("NodeColors", error_if_absent=False)
            self._node_colors_dirty = False

        # Update per-edge radii if dirty.
        if self._edge_radii_dirty and self._render_struct.is_enabled():
            if self._edge_radii is not None:
                self._render_struct.add_scalar_quantity(
                    "EdgeRadii", self._edge_radii, defined_on="edges", enabled=False
                )
                self._render_struct.set_edge_radius_quantity(
                    "EdgeRadii", autoscale=False
                )
            else:
                # Clear per-edge radii if set to None.
                self._render_struct.remove_quantity("EdgeRadii", error_if_absent=False)
                self._render_struct.clear_edge_radius_quantity()
            self._edge_radii_dirty = False

        # Update per-edge colors if dirty.
        if self._edge_colors_dirty and self._render_struct.is_enabled():
            if self._edge_colors is not None:
                self._render_struct.add_color_quantity(
                    "EdgeColors", self._edge_colors, defined_on="edges", enabled=True
                )
            else:
                self._render_struct.remove_quantity("EdgeColors", error_if_absent=False)
            self._edge_colors_dirty = False

    @override_from(Renderer)
    def remove(self):
        """Remove the renderer and clean up resources."""
        if self._render_struct is not None:
            self._render_struct.remove()
            self._render_struct = None

    def is_dirty(self) -> bool:
        """Returns True if the render structure is marked as dirty."""
        return (
            self._nodes_dirty
            or self._node_radii_dirty
            or self._node_colors_dirty
            or self._edge_radii_dirty
            or self._edge_colors_dirty
        )

    ####################################################################################
    # Geometry Methods
    ####################################################################################

    def set_curve_network(  # noqa: C901
        self,
        nodes: npt.NDArray[float] | npt.ArrayLike,
        edges: npt.NDArray[int] | npt.ArrayLike,
        node_radii: npt.NDArray[float] | npt.ArrayLike | None = None,
        edge_radii: npt.NDArray[float] | npt.ArrayLike | None = None,
        node_colors: npt.NDArray[float] | npt.ArrayLike | None = None,
        edge_colors: npt.NDArray[float] | npt.ArrayLike | None = None,
    ):
        """Set the curve network geometry. The coordinates are expected to be in
        global space. The coordinate transform will be applied to convert to the
        rendering coordinate system.

        Args:
            nodes: Nx3 array of node coordinates in global space.
            edges: Mx2 array of edge indices, where each row specifies the indices
                of the two nodes that form an edge.
            node_radii: Optional N-element array of per-node radii. If None, the global
                radius will be used for all nodes.
            edge_radii: Optional M-element array of per-edge radii. If None, the global
                radius will be used for all edges.
            node_colors: Optional Nx3 array of per-node RGB colors. If None, the global
                color will be used for all nodes.
            edge_colors: Optional Mx3 array of per-edge RGB colors. If None, the global
                color will be used for all edges.
        """

        # Validate input data and generate copies to avoid modifying originals.
        nodes = np.array(nodes, dtype=np.float32)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("Invalid nodes. Expected Nx3 array.")

        edges = np.array(edges, dtype=np.int32)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Invalid edges. Expected Mx2 array.")

        num_nodes = nodes.shape[0]
        num_edges = edges.shape[0]

        # Validate per-node radii if provided.
        if node_radii is not None:
            node_radii = np.asarray(node_radii, dtype=np.float32)
            if node_radii.ndim != 1:
                raise ValueError("Invalid node_radii. Expected 1D array.")
            if node_radii.shape[0] != num_nodes:
                raise ValueError(
                    "Invalid node_radii. Expected same number of elements as nodes."
                )
            self._node_radii = node_radii.copy()
        else:
            self._node_radii = None

        # Validate per-edge radii if provided.
        if edge_radii is not None:
            edge_radii = np.asarray(edge_radii, dtype=np.float32)
            if edge_radii.ndim != 1:
                raise ValueError("Invalid edge_radii. Expected 1D array.")
            if edge_radii.shape[0] != num_edges:
                raise ValueError(
                    "Invalid edge_radii. Expected same number of elements as edges."
                )
            self._edge_radii = edge_radii.copy()
        else:
            self._edge_radii = None

        # Validate per-node colors if provided.
        if node_colors is not None:
            node_colors = np.asarray(node_colors, dtype=np.float32)
            if node_colors.ndim != 2 or node_colors.shape[1] != 3:
                raise ValueError("Invalid node_colors. Expected Nx3 array.")
            if node_colors.shape[0] != num_nodes:
                raise ValueError(
                    "Invalid node_colors. Expected same number of elements as nodes."
                )
            self._node_colors = node_colors.copy()
        else:
            self._node_colors = None

        # Validate per-edge colors if provided.
        if edge_colors is not None:
            edge_colors = np.asarray(edge_colors, dtype=np.float32)
            if edge_colors.ndim != 2 or edge_colors.shape[1] != 3:
                raise ValueError("Invalid edge_colors. Expected Mx3 array.")
            if edge_colors.shape[0] != num_edges:
                raise ValueError(
                    "Invalid edge_colors. Expected same number of elements as edges."
                )
            self._edge_colors = edge_colors.copy()
        else:
            self._edge_colors = None

        # Update members.
        self._nodes = nodes
        self._edges = edges
        self._nodes_dirty = True
        self._node_radii_dirty = True
        self._node_colors_dirty = True
        self._edge_radii_dirty = True
        self._edge_colors_dirty = True

    def set_nodes(
        self,
        nodes: npt.NDArray[float] | npt.ArrayLike,
        node_radii: npt.NDArray[float] | npt.ArrayLike | None = None,
        node_colors: npt.NDArray[float] | npt.ArrayLike | None = None,
    ):
        """Set only the node positions of the curve network. The edges remain unchanged.
        The coordinates are expected to be in global space.

        Args:
            nodes: Nx3 array of node coordinates in global space.
            node_radii: Optional N-element array of per-node radii. If None, the existing
                per-node radii (if any) will be cleared.
            node_colors: Optional Nx3 array of per-node RGB colors. If None, the existing
                per-node colors (if any) will be cleared.
        """

        # Validate input data.
        nodes = np.array(nodes, dtype=np.float32)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("Invalid nodes. Expected Nx3 array.")

        num_nodes = nodes.shape[0]

        # Validate per-node radii if provided.
        if node_radii is not None:
            node_radii = np.asarray(node_radii, dtype=np.float32)
            if node_radii.ndim != 1:
                raise ValueError("Invalid node_radii. Expected 1D array.")
            if node_radii.shape[0] != num_nodes:
                raise ValueError(
                    "Invalid node_radii. Expected same number of elements as nodes."
                )
            self._node_radii = node_radii.copy()
        else:
            self._node_radii = None

        # Validate per-node colors if provided.
        if node_colors is not None:
            node_colors = np.asarray(node_colors, dtype=np.float32)
            if node_colors.ndim != 2 or node_colors.shape[1] != 3:
                raise ValueError("Invalid node_colors. Expected Nx3 array.")
            if node_colors.shape[0] != num_nodes:
                raise ValueError(
                    "Invalid node_colors. Expected same number of elements as nodes."
                )
            self._node_colors = node_colors.copy()
        else:
            self._node_colors = None

        # Update members.
        self._nodes = nodes
        self._nodes_dirty = True
        self._node_radii_dirty = True
        self._node_colors_dirty = True

    def get_nodes(self) -> npt.NDArray[float]:
        """Returns the node coordinates of the curve network."""
        return self._nodes

    def get_edges(self) -> npt.NDArray[int]:
        """Returns the edge indices of the curve network."""
        return self._edges

    def get_num_nodes(self) -> int:
        """Returns the number of nodes in the curve network."""
        return self._nodes.shape[0]

    def get_num_edges(self) -> int:
        """Returns the number of edges in the curve network."""
        return self._edges.shape[0]

    ####################################################################################
    # Per-Node Appearance Properties
    ####################################################################################

    def get_node_radii(self) -> npt.NDArray[float] | None:
        """Returns the per-node radii array, or None if using global radius."""
        return self._node_radii

    def set_node_radii(self, radii: npt.NDArray[float] | npt.ArrayLike | None):
        """Sets per-node radii for the curve network.

        Args:
            radii: N-element array of per-node radii, or None to clear per-node
                radii and use the global radius.
        """
        if radii is not None:
            radii = np.asarray(radii, dtype=np.float32)
            if radii.ndim != 1:
                raise ValueError("Invalid radii. Expected 1D array.")
            if radii.shape[0] != self._nodes.shape[0]:
                raise ValueError(
                    "Invalid radii. Expected same number of elements as nodes. If you "
                    "want to change the number of nodes, call set_curve_network() instead."
                )
            self._node_radii = radii.copy()
        else:
            self._node_radii = None
        self._node_radii_dirty = True

    def get_node_colors(self) -> npt.NDArray[float] | None:
        """Returns the per-node colors array, or None if using global color."""
        return self._node_colors

    def set_node_colors(self, colors: npt.NDArray[float] | npt.ArrayLike | None):
        """Sets per-node colors for the curve network.

        Args:
            colors: Nx3 array of per-node RGB colors with values in [0, 1],
                or None to clear per-node colors and use the global color.
        """
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.ndim != 2 or colors.shape[1] != 3:
                raise ValueError("Invalid colors. Expected Nx3 array.")
            if colors.shape[0] != self._nodes.shape[0]:
                raise ValueError(
                    "Invalid colors. Expected same number of elements as nodes. If "
                    "you want to change the number of nodes, call set_curve_network() "
                    "instead."
                )
            self._node_colors = colors.copy()
        else:
            self._node_colors = None
        self._node_colors_dirty = True

    ####################################################################################
    # Per-Edge Appearance Properties
    ####################################################################################

    def get_edge_radii(self) -> npt.NDArray[float] | None:
        """Returns the per-edge radii array, or None if using global radius."""
        return self._edge_radii

    def set_edge_radii(self, radii: npt.NDArray[float] | npt.ArrayLike | None):
        """Sets per-edge radii for the curve network.

        Args:
            radii: M-element array of per-edge radii, or None to clear per-edge
                radii and use the global radius.
        """
        if radii is not None:
            radii = np.asarray(radii, dtype=np.float32)
            if radii.ndim != 1:
                raise ValueError("Invalid radii. Expected 1D array.")
            if radii.shape[0] != self._edges.shape[0]:
                raise ValueError(
                    "Invalid radii. Expected same number of elements as edges. If you "
                    "want to change the number of edges, call set_curve_network() instead."
                )
            self._edge_radii = radii.copy()
        else:
            self._edge_radii = None
        self._edge_radii_dirty = True

    def get_edge_colors(self) -> npt.NDArray[float] | None:
        """Returns the per-edge colors array, or None if using global color."""
        return self._edge_colors

    def set_edge_colors(self, colors: npt.NDArray[float] | npt.ArrayLike | None):
        """Sets per-edge colors for the curve network.

        Args:
            colors: Mx3 array of per-edge RGB colors with values in [0, 1],
                or None to clear per-edge colors and use the global color.
        """
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.ndim != 2 or colors.shape[1] != 3:
                raise ValueError("Invalid colors. Expected Mx3 array.")
            if colors.shape[0] != self._edges.shape[0]:
                raise ValueError(
                    "Invalid colors. Expected same number of elements as edges. If "
                    "you want to change the number of edges, call set_curve_network() "
                    "instead."
                )
            self._edge_colors = colors.copy()
        else:
            self._edge_colors = None
        self._edge_colors_dirty = True

    ####################################################################################
    # Appearance Properties
    ####################################################################################

    def get_render_structure(self) -> ps.CurveNetwork:
        """Returns the underlying Polyscope curve network structure."""
        assert self._render_struct is not None, "Render structure not initialized."
        return self._render_struct

    def is_enabled(self) -> bool:
        """Returns whether the curve network rendering is enabled."""
        return self.get_render_structure().is_enabled()

    def set_enabled(self, enabled: bool):
        """Enables or disables the curve network rendering."""
        self.get_render_structure().set_enabled(enabled)

    def get_radius(self) -> float:
        """Returns the current global edge radius."""
        return self.get_render_structure().get_radius()

    def set_radius(self, radius: float):
        """Sets a new global radius for all edges. This radius is used when per-node
        and per-edge radii are not specified.

        Args:
            radius: Global radius for all edges.
        """
        self.get_render_structure().set_radius(radius, relative=False)

    def get_color(self) -> npt.NDArray[float]:
        """Returns the current global color."""
        return np.asarray(self.get_render_structure().get_color(), dtype=np.float32)

    def set_color(self, color: npt.ArrayLike):
        """Sets a new color for all elements.

        Args:
            color: RGB color as a 3-element array with values in [0, 1].
        """
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self.get_render_structure().set_color(color)

    def push_color(self, color: npt.ArrayLike):
        """Pushes a new color onto the stack.

        Args:
            color: RGB color as a 3-element array with values in [0, 1].
        """
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._color.append(self.get_color())
        self.get_render_structure().set_color(color)

    def pop_color(self):
        """Pops the current color from the stack."""
        self.get_render_structure().set_color(self._color[-1])
        self._color.pop()

    def get_transparency(self) -> float:
        """Returns the current transparency level."""
        return self.get_render_structure().get_transparency()

    def set_transparency(self, transparency: float):
        """Sets a new transparency level.

        Args:
            transparency: Transparency level (0.0 = opaque, 1.0 = fully transparent).
        """
        self.get_render_structure().set_transparency(transparency)

    def push_transparency(self, transparency: float):
        """Pushes a new transparency level onto the stack.

        Args:
            transparency: Transparency level (0.0 = opaque, 1.0 = fully transparent).
        """
        self._transparency.append(self.get_transparency())
        self.get_render_structure().set_transparency(transparency)

    def pop_transparency(self):
        """Pops the current transparency level from the stack."""
        self._render_struct.set_transparency(self._transparency[-1])
        self._transparency.pop()
