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

import enum

import numpy as np
import numpy.typing as npt
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import (
    apply_affine_transform,
    apply_linear_map,
)
from superdex.physics.viewer.backend import polyscope as ps
from superdex.physics.viewer.renderers.axes_renderer import AxesRenderer
from superdex.physics.viewer.renderers.renderer import Renderer
from superdex.physics.viewer.utils.aabb import AABB

########################################################################################


class MeshRenderer(Renderer):
    """
    Class responsible for the rendering of 3D mesh objects using Polyscope. Provides
    functionalities to manage and render 3D mesh geometries, including vertices, edges,
    and faces. It supports various rendering settings such as color, transparency, and
    transformation. The MeshRenderer class also allows for stack-based management of
    rendering properties, enabling easy overrides and reversion to previous states.
    """

    ####################################################################################
    # Constants
    ####################################################################################

    class BackFacePolicy(enum.Enum):
        """Back face rendering policy."""

        IDENTICAL = "identical"
        """Render back faces with the same color as front faces."""
        DIFFERENT = "different"
        """Render back faces with a different color than front faces."""
        CUSTOM = "custom"
        """Render back faces with a custom color."""
        CULL = "cull"
        """Cull back faces."""

    class Material(enum.Enum):
        """Material type for surface rendering."""

        CLAY = "clay"
        """Clay material."""
        WAX = "wax"
        """Wax material."""
        CANDY = "candy"
        """Candy material."""
        FLAT = "flat"
        """Flat material."""
        MUD = "mud"
        """Mud material."""
        CERAMIC = "ceramic"
        """Ceramic material."""
        JADE = "jade"
        """Jade material."""
        NORMAL = "normal"
        """Normal material."""

    class TextureOrigin(enum.Enum):
        """Texture origin."""

        UPPER_LEFT = "upper_left"
        """Upper-left texture origin."""
        LOWER_LEFT = "lower_left"
        """Lower-left texture origin."""

    class TextureFilter(enum.Enum):
        """Texture filtering mode."""

        LINEAR = "linear"
        """Linear texture filtering."""
        NEAREST = "nearest"
        """Nearest neighbor texture filtering."""

    class Colormap(enum.Enum):
        """Colormap for scalar textures."""

        VIRIDIS = "viridis"
        """Viridis colormap."""
        MAGMA = "magma"
        """Magma colormap."""
        INFERNO = "inferno"
        """Inferno colormap."""
        PLASMA = "plasma"
        """Plasma colormap."""
        GRAY = "gray"
        """Gray colormap."""
        BLUES = "blues"
        """Blues colormap."""
        REDS = "reds"
        """Reds colormap."""
        COOL_WARM = "coolwarm"
        """Cool-warm diverging colormap."""
        PURPLE_GREEN = "purple-green"
        """Purple-green diverging colormap."""
        SPECTRAL = "spectral"
        """Spectral colormap."""
        RAINBOW = "rainbow"
        """Rainbow colormap."""
        JET = "jet"
        """Jet colormap."""
        TURBO = "turbo"
        """Turbo colormap."""
        PHASE = "phase"
        """Phase colormap."""

    ####################################################################################
    # Members
    ####################################################################################

    # Mesh geometry and topology.
    _transform: npt.NDArray[float]
    _coordinates: npt.NDArray[float]
    _faces: npt.NDArray[int]
    _edges: npt.NDArray[int]
    _local_aabb: AABB  # Same as _aabb if soft.
    _aabb: AABB  # Approximate if rigid, exact if soft.
    _texture_coordinates: npt.NDArray[float] | None  # Optional UV coordinates (Nx2).

    # Texture data and settings.
    _texture: npt.NDArray[float] | None  # Texture data (H,W) or (H,W,3).
    _texture_dirty: bool  # Flag to track if texture needs updating.
    _texture_filter: MeshRenderer.TextureFilter  # Texture filtering mode.
    _texture_origin: MeshRenderer.TextureOrigin  # Texture origin.
    _texture_colormap: MeshRenderer.Colormap  # Colormap for scalar textures.
    _texture_colormap_range: tuple[float, float] | None  # Range for colormapping.

    # Render settings.
    # Use a stack-based approach to allow for overrides.
    _front_face_color: list[npt.NDArray[float]]
    _back_face_color: list[npt.NDArray[float]]
    _back_face_policy: list[MeshRenderer.BackFacePolicy]
    _transparency: list[float]
    _edge_color: list[npt.NDArray[float]]
    _edge_width: list[float]
    _edge_radius: list[float]
    _node_color: list[npt.NDArray[float]]
    _node_radius: list[float]
    _material: list[MeshRenderer.Material]
    _smooth_shading: list[bool]

    # Backing Polyscope render structures.
    _group: ps.Group | None
    _surface_struct: ps.SurfaceMesh | None
    _surface_dirty: bool
    _edges_struct: ps.CurveNetwork | None
    _edges_dirty: bool
    _nodes_struct: ps.PointCloud | None
    _node_dirty: bool
    _axes_renderer: AxesRenderer | None
    _transform_dirty: bool

    ####################################################################################
    # Constants
    ####################################################################################

    _TAG: str = "[Mesh]"
    """Tag used to identify MeshRenderer-related render structure."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(  # noqa: C901
        self,
        name: str,
        coordinates: npt.NDArray[float],
        coordinate_transform: CoordinateTransform,
        faces: npt.NDArray[int],
        transform: npt.NDArray[float] | None = None,
        texture_coordinates: npt.NDArray[float] | None = None,
        front_face_color: npt.NDArray[float] | npt.ArrayLike | None = None,
        back_face_color: npt.NDArray[float] | npt.ArrayLike | None = None,
        back_face_policy: MeshRenderer.BackFacePolicy | None = None,
        transparency: float | None = None,
        edge_color: npt.NDArray[float] | npt.ArrayLike | None = None,
        edge_width: float | None = None,
        edge_radius: float | None = None,
        node_color: npt.NDArray[float] | npt.ArrayLike | None = None,
        node_radius: float | None = None,
        material: MeshRenderer.Material | None = None,
        smooth_shading: bool | None = None,
    ):
        """Initializes the MeshRenderer with given parameters."""

        # Initialize base class
        super().__init__(name, coordinate_transform)

        # Fall back to identity if no transform is provided.
        if transform is None:
            transform = np.eye(4, dtype=np.float32)

        # Initialize rendering structure.
        self._transform = np.eye(4, dtype=np.float32)
        self._local_aabb = AABB.empty()
        self._aabb = AABB.empty()
        self._texture_coordinates = None
        self._group = None
        self._surface_struct = None
        self._surface_dirty = False
        self._edges_struct = None
        self._edges_dirty = False
        self._nodes_struct = None
        self._node_dirty = False
        self._axes_renderer = None
        self._transform_dirty = False
        self.replace_geometry(coordinates, faces, transform, texture_coordinates)

        # Initialize texture settings.
        self._texture = None
        self._texture_dirty = False
        self._texture_filter = MeshRenderer.TextureFilter.LINEAR
        self._texture_origin = MeshRenderer.TextureOrigin.UPPER_LEFT
        self._texture_colormap = MeshRenderer.Colormap.VIRIDIS
        self._texture_colormap_range = None

        # Initialize render settings.
        self._front_face_color = []
        self._back_face_color = []
        self._back_face_policy = []
        self._transparency = []
        self._edge_color = []
        self._edge_width = []
        self._edge_radius = []
        self._node_color = []
        self._node_radius = []
        self._material = []
        self._smooth_shading = []

        if front_face_color is not None:
            self.set_front_face_color(front_face_color)
        else:
            front_face_color = self.get_front_face_color()
        if back_face_color is not None:
            self.set_back_face_color(back_face_color)
        if back_face_policy is not None:
            self.set_back_face_policy(back_face_policy)
        if transparency is not None:
            self.set_transparency(transparency)
        if edge_color is None:
            edge_color = 0.25 * np.asarray(front_face_color)
        if edge_width is None:
            edge_width = 0.01
        if edge_radius is None:
            edge_radius = 0.00025
        if node_color is None:
            node_color = 0.125 * np.asarray(front_face_color)
        if node_radius is None:
            node_radius = 0.00125
        self.set_edge_color(np.asarray(edge_color))
        self.set_edge_width(edge_width)
        self.set_edge_radius(edge_radius)
        self.set_node_color(np.asarray(node_color))
        self.set_node_radius(node_radius)
        if material is not None:
            self.set_material(material)
        if smooth_shading is not None:
            self.set_smooth_shading(smooth_shading)

        # Hide edges, nodes and axes by default.
        self.set_enable_edges(False)
        self.set_enable_nodes(False)
        self.set_enable_axes(False)

    ####################################################################################
    # General Management Methods
    ####################################################################################

    def _create_render_structures(self) -> None:
        """Resets the renderer, creating the associated render structures. Called
        interally during initialization and whenever the mesh topology changes."""

        tagged_name = f"{self._TAG} {self._name}"

        # Generate group (if not already created).
        group = self._group
        if group is None:
            group = ps.create_group(tagged_name)
            group.set_hide_descendants_from_structure_lists(True)
            self._group = group

        # Convert coordinates to target coordinate system.
        source_to_ps = self._coordinate_transform.source_to_target
        coordinates_ps = apply_linear_map(source_to_ps[:3, :3], self._coordinates)

        # Flip faces if the coordinate system is left-handed to ensure the appropriate
        # winding order.
        flip_faces = self._coordinate_transform.encodes_reflection
        faces = self._faces[:, ::-1] if flip_faces else self._faces

        # Generate surface structure.
        self._surface_dirty = False
        self._surface_struct = ps.register_surface_mesh(
            name=f"{tagged_name} (Surface)",
            vertices=coordinates_ps,
            faces=faces,
        )
        group.add_child_structure(self._surface_struct)

        # Add UV parametrization based on texture coordinates, if provided.
        if self._texture_coordinates is not None:
            self.get_surface_structure().add_parameterization_quantity(
                "texture_coordinates",
                self._texture_coordinates,
                defined_on="vertices",
                coords_type="unit",
            )

        # Generate edges structure.
        self._edges_dirty = False
        self._edges_struct = ps.register_curve_network(
            name=f"{tagged_name} (Edges)",
            nodes=coordinates_ps,
            edges=self._edges,
            material="flat",
        )
        group.add_child_structure(self._edges_struct)

        # Generate nodes structure.
        self._node_dirty = False
        self._nodes_struct = ps.register_point_cloud(
            name=f"{tagged_name} (Nodes)",
            points=coordinates_ps,
        )
        group.add_child_structure(self._nodes_struct)

        # Generate axes structure.
        self._axes_renderer = AxesRenderer(self._name, self._coordinate_transform)
        group.add_child_structure(self._axes_renderer.get_structure())

    @override_from(Renderer)
    def update(self) -> None:
        """Updates the render structures if they are marked as dirty."""
        # Check if render structures are initialized
        is_valid = self._surface_struct and self._edges_struct and self._nodes_struct
        assert is_valid, "Render structure not initialized."

        # Retrieve the change of basis matrix.
        source_to_ps = self._coordinate_transform.source_to_target
        ps_to_source = self._coordinate_transform.target_to_source

        # Determine if we need to update any of the render structures.
        if (
            (self._surface_dirty and self.get_surface_structure().is_enabled())
            or (self._edges_dirty and self.get_edges_structure().is_enabled())
            or (self._node_dirty and self.get_nodes_structure().is_enabled())
        ):
            coordinates_ps = apply_linear_map(source_to_ps[:3, :3], self._coordinates)

            # Update only visible render structures.
            if self._surface_dirty and self.get_surface_structure().is_enabled():
                self.get_surface_structure().update_vertex_positions(coordinates_ps)

                # Update or remove UV parametrization based on texture coordinates
                if self._texture_coordinates is not None:
                    self.get_surface_structure().add_parameterization_quantity(
                        "texture_coordinates",
                        self._texture_coordinates,
                        defined_on="vertices",
                        coords_type="unit",
                    )
                else:
                    self.get_surface_structure().remove_quantity("texture_coordinates")

                self._surface_dirty = False
            if self._edges_dirty and self.get_edges_structure().is_enabled():
                self.get_edges_structure().update_node_positions(coordinates_ps)
                self._edges_dirty = False
            if self._node_dirty and self.get_nodes_structure().is_enabled():
                self.get_nodes_structure().update_point_positions(coordinates_ps)
                self._node_dirty = False

        # Update texture if dirty and surface is visible
        if self._texture_dirty and self.get_surface_structure().is_enabled():
            if self._texture is None:
                # Remove texture quantities if texture is None
                self.get_surface_structure().remove_quantity("texture_scalar")
                self.get_surface_structure().remove_quantity("texture_color")
            elif self._texture.ndim == 2:
                # Remove color texture if it exists
                self.get_surface_structure().remove_quantity("texture_color")

                # Scalar texture (H, W)
                self.get_surface_structure().add_scalar_quantity(
                    "texture_scalar",
                    self._texture,
                    defined_on="texture",
                    param_name="texture_coordinates",
                    cmap=self._texture_colormap.value,
                    image_origin=self._texture_origin.value,
                    filter_mode=self._texture_filter.value,
                    vminmax=self._texture_colormap_range,
                    enabled=True,
                )
            else:
                # Remove scalar texture if it exists
                self.get_surface_structure().remove_quantity("texture_scalar")

                # RGB color texture (H, W, 3)
                self.get_surface_structure().add_color_quantity(
                    "texture_color",
                    self._texture,
                    defined_on="texture",
                    param_name="texture_coordinates",
                    image_origin=self._texture_origin.value,
                    filter_mode=self._texture_filter.value,
                    enabled=True,
                )
            self._texture_dirty = False

        # Update transforms.
        if self._transform_dirty:
            # Build conjugate transform to account for the change in coordinate system.
            transform_ps = source_to_ps @ self._transform @ ps_to_source
            self.get_surface_structure().set_transform(transform_ps)
            self.get_edges_structure().set_transform(transform_ps)
            self.get_nodes_structure().set_transform(transform_ps)
            self.get_axes_renderer().set_transform(self._transform)  # NOT transform_ps!
            self.get_axes_renderer().update()
            self._transform_dirty = False

    @override_from(Renderer)
    def remove(self) -> None:
        """Removes the render structures."""
        # Remove surface structure if it exists
        if self._surface_struct is not None:
            self._surface_struct.remove()
            self._surface_struct = None
        # Remove edges structure if it exists
        if self._edges_struct is not None:
            self._edges_struct.remove()
            self._edges_struct = None
        # Remove nodes structure if it exists
        if self._nodes_struct is not None:
            self._nodes_struct.remove()
            self._nodes_struct = None
        # Remove axes structure if it exists
        if self._axes_renderer is not None:
            self._axes_renderer.remove()
            self._axes_renderer = None
        # Remove the group if it exists
        if self._group is not None:
            ps.remove_group(self._group)
            self._group = None

    def is_dirty(self) -> bool:
        """Returns True if the render structures are marked as dirty."""
        return (
            self._surface_dirty
            or self._edges_dirty
            or self._node_dirty
            or self._transform_dirty
            or self._texture_dirty
        )

    ####################################################################################
    # Geometry and topology
    ####################################################################################

    def replace_geometry(
        self,
        coordinates: npt.NDArray[float],
        faces: npt.NDArray[int],
        transform: npt.NDArray[float],
        texture_coordinates: npt.NDArray[float] | None = None,
    ):
        """Replaces the geometry and topology of the mesh. The vertex coordinates are
        expected to be in local space, with the transformation matrix indicating the
        transformation from local to world space. Note this method triggers an update
        of the render structures."""

        # Validate the input data.
        # Generate copy of the input data to avoid modifying the original arrays.
        coordinates = np.array(coordinates, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Invalid coordinates. Expected Nx3 array.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Invalid faces. Expected Mx3 array.")

        # Validate texture coordinates if provided.
        if texture_coordinates is not None:
            texture_coordinates = np.array(texture_coordinates, dtype=np.float32)
            if texture_coordinates.ndim != 2 or texture_coordinates.shape[1] != 2:
                raise ValueError("Invalid texture coordinates. Expected Nx2 array.")
            if texture_coordinates.shape[0] != coordinates.shape[0]:
                raise ValueError(
                    "Texture coordinates must have the same number of entries as vertices."
                )
            self._texture_coordinates = texture_coordinates
        else:
            self._texture_coordinates = None

        # Compute edge connectivity.
        edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        # Update members.
        self._coordinates = coordinates
        self._faces = faces
        self._edges = edges
        self._local_aabb.compute_from_points(self._coordinates)
        self._create_render_structures()
        self.set_transform(np.asarray(transform))

    def get_transform(self) -> npt.NDArray[float]:
        """Returns the current transformation matrix."""
        return self._transform

    def set_transform(self, transform: npt.NDArray[float]) -> None:
        """Sets a new transformation matrix and updates the AABB."""
        transform = np.asarray(transform, dtype=np.float32)
        if transform.shape != (4, 4):
            raise ValueError("Invalid transform. Expected 4x4 array.")
        self._transform[:] = transform
        self._aabb.compute_from_transformed_aabb(self._local_aabb, self._transform)
        self._transform_dirty = True

    def get_local_coordinates(self) -> npt.NDArray[float]:
        """Returns the local coordinates of the mesh."""
        return self._coordinates

    def set_local_coordinates(self, coordinates: npt.NDArray[float]) -> None:
        """Sets new local coordinates and marks the render structures as dirty."""
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Invalid coordinates. Expected Nx3 array.")
        if coordinates.shape[0] != self._coordinates.shape[0]:
            raise ValueError("Invalid coordinates. Expected same number of vertices.")
        self._coordinates[:] = coordinates
        self._local_aabb.compute_from_points(self._coordinates)
        self._aabb.compute_from_transformed_aabb(self._local_aabb, self._transform)
        # Defer updating the render structure until the next update call.
        self._surface_dirty = True
        self._edges_dirty = True
        self._node_dirty = True

    def get_world_coordinates(self) -> npt.NDArray[float]:
        """Returns the world coordinates of the mesh."""
        return apply_affine_transform(self._transform, self._coordinates)

    def get_faces(self) -> npt.NDArray[int]:
        """Returns the face indices of the mesh."""
        return self._faces

    def get_edges(self) -> npt.NDArray[int]:
        """Returns the edge indices of the mesh."""
        return self._edges

    def get_local_aabb(self) -> AABB:
        """Returns the local axis-aligned bounding box."""
        return self._local_aabb

    def get_aabb(self) -> AABB:
        """Returns the transformed axis-aligned bounding box."""
        return self._aabb

    def get_texture_coordinates(self) -> npt.NDArray[float] | None:
        """Returns the texture coordinates (UV) of the mesh, or None if not set."""
        return self._texture_coordinates

    def set_texture_coordinates(
        self, texture_coordinates: npt.NDArray[float] | npt.ArrayLike | None
    ) -> None:
        """Sets texture coordinates (UV) for the mesh vertices. If None, removes
        existing texture coordinates.

        Args:
            texture_coordinates: Nx2 array of UV coordinates, where N is the number
                of vertices, or None to remove texture coordinates.

        Raises:
            ValueError: If texture coordinates don't have shape Nx2 or don't match
                the number of vertices.
        """
        if texture_coordinates is not None:
            texture_coordinates = np.array(texture_coordinates, dtype=np.float32)
            if texture_coordinates.ndim != 2 or texture_coordinates.shape[1] != 2:
                raise ValueError("Invalid texture coordinates. Expected Nx2 array.")
            if texture_coordinates.shape[0] != self._coordinates.shape[0]:
                raise ValueError(
                    "Texture coordinates must have the same number of entries as vertices."
                )
            self._texture_coordinates = texture_coordinates
        else:
            self._texture_coordinates = None

        # Mark surface as dirty to trigger update/removal of UV parametrization.
        self._surface_dirty = True

    def get_texture(self) -> npt.NDArray[float] | None:
        """Returns the current texture data, or None if not set.

        Returns:
            Texture array of shape (H,W) for scalar textures or (H,W,3) for RGB
            color textures, or None if no texture is set.
        """
        return self._texture

    def set_texture(self, texture: npt.NDArray[float] | npt.ArrayLike | None) -> None:
        """Sets or updates the texture data for the mesh. If None, disables texture
        rendering. The texture will be applied using the mesh's texture coordinates.

        The method is optimized to minimize reallocation - if the new texture has the
        same dimensions as the current one, it will copy the data into the existing
        array rather than creating a new one.

        Args:
            texture: Texture data as (H,W) array for scalar textures or (H,W,3) array
                for RGB color textures, or None to disable texture rendering.

        Raises:
            ValueError: If texture dimensions are invalid.
        """

        # Disable texture rendering if texture is None.
        if texture is None:
            self._texture = None
            self._texture_dirty = True
            return

        # Validate texture dimensions
        texture = np.asarray(texture, dtype=np.float32)
        ndim, shape = texture.ndim, texture.shape
        if ndim != 2 and ndim != 3 or (ndim == 3 and shape[2] != 3):
            raise ValueError(
                "Invalid texture. Expected (H,W) for scalar or (H,W,3) for RGB color."
            )

        # Update backing texture array without reallocating if possible.
        # TODO: There might be a way to make the update more efficient by updating the
        # underlying quantity managed buffer, but this API is not clearly documented.
        # Investigate if this becomes a bottleneck.
        if self._texture is not None and self._texture.shape == shape:
            self._texture[:] = texture
        else:
            self._texture = texture.copy()

        self._texture_dirty = True

    def get_texture_origin(self) -> MeshRenderer.TextureOrigin:
        """Returns the current texture origin.

        Returns:
            TextureOrigin: Current texture origin (UPPER_LEFT or LOWER_LEFT).
        """
        return self._texture_origin

    def set_texture_origin(self, texture_origin: MeshRenderer.TextureOrigin) -> None:
        """Sets the texture origin.

        Args:
            texture_origin: Texture origin (UPPER_LEFT or LOWER_LEFT).
        """
        texture_origin = MeshRenderer.TextureOrigin(texture_origin)
        if self._texture_origin != texture_origin:
            self._texture_origin = texture_origin
            self._texture_dirty = True

    def get_texture_filter(self) -> MeshRenderer.TextureFilter:
        """Returns the current texture filtering mode.

        Returns:
            TextureFilter: Current texture filtering mode (LINEAR or NEAREST).
        """
        return self._texture_filter

    def set_texture_filter(self, mode: MeshRenderer.TextureFilter) -> None:
        """Sets the texture filtering mode.

        Args:
            mode: Texture filtering mode (LINEAR or NEAREST).
        """
        mode = MeshRenderer.TextureFilter(mode)
        if self._texture_filter != mode:
            self._texture_filter = mode
            self._texture_dirty = True

    def get_texture_colormap(self) -> MeshRenderer.Colormap:
        """Returns the current colormap for scalar textures.

        Returns:
            Colormap: Current colormap. Only applicable to scalar (H,W) textures.
        """
        return self._texture_colormap

    def set_texture_colormap(self, colormap: MeshRenderer.Colormap) -> None:
        """Sets the colormap for scalar textures. Only applicable to scalar (H,W)
        textures, not RGB (H,W,3) textures.

        Args:
            colormap: Colormap to use for scalar textures.
        """
        colormap = MeshRenderer.Colormap(colormap)
        if self._texture_colormap != colormap:
            self._texture_colormap = colormap
            self._texture_dirty = True

    def get_texture_colormap_range(self) -> tuple[float, float] | None:
        """Returns the current mapping range for scalar textures.

        Returns:
            Tuple of (vmin, vmax) for the texture mapping range, or None if using
            automatic range.
        """
        return self._texture_colormap_range

    def set_texture_colormap_range(
        self, colormap_range: tuple[float, float] | None
    ) -> None:
        """Sets the mapping range for scalar textures. Only applicable to scalar
        (H,W) textures, not RGB (H,W,3) textures.

        Args:
            colormap_range: Tuple of (vmin, vmax) specifying the mapping range, or None
                to use automatic range based on texture data.

        Raises:
            ValueError: If range is not a 2-element tuple or vmin >= vmax.
        """
        if colormap_range is not None:
            if len(colormap_range) != 2:
                raise ValueError("Range must be a 2-element tuple (vmin, vmax).")
            vmin, vmax = colormap_range
            if vmin >= vmax:
                raise ValueError("Range must have vmin < vmax.")
            colormap_range = (float(vmin), float(vmax))
        if self._texture_colormap_range != colormap_range:
            self._texture_colormap_range = colormap_range
            self._texture_dirty = True

    ####################################################################################
    # Surface rendering
    ####################################################################################

    def is_surface_enabled(self) -> bool:
        """Checks if the surface rendering is enabled."""
        return self.get_surface_structure().is_enabled()

    def set_enable_surface(self, enabled: bool) -> None:
        """Enables or disables the surface rendering."""
        enabled = bool(enabled)
        self.get_surface_structure().set_enabled(enabled)

    def get_surface_structure(self) -> ps.SurfaceMesh:
        """Returns the surface mesh render structure."""
        assert self._surface_struct is not None
        return self._surface_struct

    def get_front_face_color(self) -> npt.NDArray[float]:
        """Returns the current front face color."""
        return np.asarray(self.get_surface_structure().get_color(), dtype=np.float32)

    def set_front_face_color(self, color: npt.ArrayLike) -> None:
        """Sets a new front face color."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self.get_surface_structure().set_color(color)

    def push_front_face_color(self, color: npt.ArrayLike) -> None:
        """Pushes a new front face color onto the stack."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._front_face_color.append(self.get_front_face_color())
        self.get_surface_structure().set_color(color)

    def pop_front_face_color(self) -> None:
        """Pops the current front face color from the stack."""
        self.get_surface_structure().set_color(self._front_face_color[-1])
        self._front_face_color.pop()

    def get_back_face_color(self) -> npt.NDArray[float]:
        """Returns the current back face color."""
        return np.asarray(
            self.get_surface_structure().get_back_face_color(), dtype=np.float32
        )

    def set_back_face_color(self, color: npt.ArrayLike) -> None:
        """Sets a new back face color."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self.get_surface_structure().set_back_face_color(color)

    def push_back_face_color(self, color: npt.ArrayLike) -> None:
        """Pushes a new back face color onto the stack."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._back_face_color.append(self.get_back_face_color())
        self.get_surface_structure().set_back_face_color(color)

    def pop_back_face_color(self) -> None:
        """Pops the current back face color from the stack."""
        self.get_surface_structure().set_back_face_color(self._back_face_color[-1])
        self._back_face_color.pop()

    def get_back_face_policy(self) -> MeshRenderer.BackFacePolicy:
        """Returns the current back face policy."""
        return MeshRenderer.BackFacePolicy(
            self.get_surface_structure().get_back_face_policy()
        )

    def set_back_face_policy(self, policy: MeshRenderer.BackFacePolicy) -> None:
        """Sets a new back face policy."""
        policy = MeshRenderer.BackFacePolicy(policy)
        self.get_surface_structure().set_back_face_policy(policy.value)

    def push_back_face_policy(self, policy: MeshRenderer.BackFacePolicy) -> None:
        """Pushes a new back face policy onto the stack."""
        policy = MeshRenderer.BackFacePolicy(policy)
        self._back_face_policy.append(self.get_back_face_policy())
        self.get_surface_structure().set_back_face_policy(policy.value)

    def pop_back_face_policy(self) -> None:
        """Pops the current back face policy from the stack."""
        self.get_surface_structure().set_back_face_policy(
            self._back_face_policy[-1].value
        )
        self._back_face_policy.pop()

    ####################################################################################
    # Edges rendering
    ####################################################################################

    def are_edges_enabled(self) -> bool:
        """Checks if the edge rendering is enabled."""
        return self.get_edges_structure().is_enabled()

    def set_enable_edges(self, enabled: bool) -> None:
        """Enables or disables the edge rendering."""
        enabled = bool(enabled)
        self.get_edges_structure().set_enabled(enabled)

    def get_edges_structure(self) -> ps.CurveNetwork:
        """Returns the edge network render structure."""
        assert self._edges_struct is not None
        return self._edges_struct

    def get_edge_color(self) -> npt.NDArray[float]:
        """Returns the current edge color."""
        return np.asarray(self.get_edges_structure().get_color(), dtype=np.float32)

    def set_edge_color(self, color: npt.ArrayLike) -> None:
        """Sets a new edge color."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self.get_surface_structure().set_edge_color(color)
        self.get_edges_structure().set_color(color)

    def push_edge_color(self, color: npt.ArrayLike) -> None:
        """Pushes a new edge color onto the stack."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._edge_color.append(self.get_edges_structure().get_color())
        self.get_surface_structure().set_edge_color(color)
        self.get_edges_structure().set_color(color)

    def pop_edge_color(self) -> None:
        """Pops the current edge color from the stack."""
        self.get_surface_structure().set_edge_color(self._edge_color[-1])
        self.get_edges_structure().set_color(self._edge_color[-1])
        self._edge_color.pop()

    def get_edge_width(self) -> float:
        """Returns the current edge width."""
        return self.get_surface_structure().get_edge_width()

    def set_edge_width(self, width: float) -> None:
        """Sets a new edge width."""
        width = float(width)
        self.get_surface_structure().set_edge_width(width)

    def push_edge_width(self, width: float) -> None:
        """Pushes a new edge width onto the stack."""
        width = float(width)
        self._edge_width.append(self.get_edge_width())
        self.get_surface_structure().set_edge_width(width)

    def pop_edge_width(self) -> None:
        """Pops the current edge width from the stack."""
        self.get_surface_structure().set_edge_width(self._edge_width[-1])
        self._edge_width.pop()

    def get_edge_radius(self) -> float:
        """Returns the current edge radius."""
        return self.get_edges_structure().get_radius()

    def set_edge_radius(self, radius: float) -> None:
        """Sets a new edge radius."""
        radius = float(radius)
        self.get_edges_structure().set_radius(radius, relative=False)

    def push_edge_radius(self, radius: float) -> None:
        """Pushes a new edge radius onto the stack."""
        radius = float(radius)
        self._edge_radius.append(self.get_edge_radius())
        self.get_edges_structure().set_radius(radius, relative=False)

    def pop_edge_radius(self) -> None:
        """Pops the current edge radius from the stack."""
        self.get_edges_structure().set_radius(self._edge_radius[-1], relative=False)
        self._edge_radius.pop()

    ####################################################################################
    # Nodes rendering
    ####################################################################################

    def are_nodes_enabled(self) -> bool:
        """Checks if the point rendering is enabled."""
        return self.get_nodes_structure().is_enabled()

    def set_enable_nodes(self, enabled: bool) -> None:
        """Enables or disables the point rendering."""
        enabled = bool(enabled)
        self.get_nodes_structure().set_enabled(enabled)

    def get_nodes_structure(self) -> ps.PointCloud:
        """Returns the point cloud render structure."""
        assert self._nodes_struct is not None
        return self._nodes_struct

    def get_node_color(self) -> npt.NDArray[float]:
        """Returns the current node color."""
        return np.asarray(self.get_nodes_structure().get_color(), dtype=np.float32)

    def set_node_color(self, color: npt.ArrayLike) -> None:
        """Sets a new node color."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self.get_nodes_structure().set_color(color)

    def push_node_color(self, color: npt.ArrayLike) -> None:
        """Pushes a new node color onto the stack."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._node_color.append(self.get_node_color())
        self.get_nodes_structure().set_color(color)

    def pop_node_color(self) -> None:
        """Pops the current node color from the stack."""
        self.get_nodes_structure().set_color(self._node_color[-1])
        self._node_color.pop()

    def get_node_radius(self) -> float:
        """Returns the current node radius."""
        return self.get_nodes_structure().get_radius()

    def set_node_radius(self, radius: float) -> None:
        """Sets a new node radius."""
        radius = float(radius)
        self.get_nodes_structure().set_radius(radius, relative=False)

    def push_node_radius(self, radius: float) -> None:
        """Pushes a new node radius onto the stack."""
        radius = float(radius)
        self._node_radius.append(self.get_node_radius())
        self.get_nodes_structure().set_radius(radius, relative=False)

    def pop_node_radius(self) -> None:
        """Pops the current node radius from the stack."""
        self.get_nodes_structure().set_radius(self._node_radius[-1], relative=False)
        self._node_radius.pop()

    ####################################################################################
    # Axes
    ####################################################################################

    def are_axes_enabled(self) -> bool:
        """Checks if the axes rendering is enabled."""
        return self.get_axes_renderer().is_enabled()

    def set_enable_axes(self, enabled: bool) -> None:
        """Enables or disables the axes rendering."""
        self.get_axes_renderer().set_enabled(enabled)

    def get_axes_renderer(self) -> AxesRenderer:
        """Gets the renderer displaying the mesh's frame of reference."""
        assert self._axes_renderer is not None
        return self._axes_renderer

    ####################################################################################
    # Global appearance
    ####################################################################################

    def set_enabled(self, enabled: bool) -> None:
        """Sets the enabled state of the renderer."""
        assert self._group is not None
        self._group.set_enabled(enabled)

    def get_transparency(self) -> float:
        """Returns the current transparency level."""
        return self.get_surface_structure().get_transparency()

    def set_transparency(self, transparency: float) -> None:
        """Sets a new transparency level."""
        transparency = float(transparency)
        self.get_surface_structure().set_transparency(transparency)
        self.get_edges_structure().set_transparency(transparency)
        self.get_nodes_structure().set_transparency(transparency)

    def push_transparency(self, transparency: float) -> None:
        """Pushes a new transparency level onto the stack."""
        transparency = float(transparency)
        self._transparency.append(self.get_transparency())
        self.get_surface_structure().set_transparency(transparency)
        self.get_edges_structure().set_transparency(transparency)
        self.get_nodes_structure().set_transparency(transparency)

    def pop_transparency(self) -> None:
        """Pops the current transparency level from the stack."""
        self.get_surface_structure().set_transparency(self._transparency[-1])
        self.get_edges_structure().set_transparency(self._transparency[-1])
        self.get_nodes_structure().set_transparency(self._transparency[-1])
        self._transparency.pop()

    def get_material(self) -> MeshRenderer.Material:
        """Returns the current material."""
        material_str = self.get_surface_structure().get_material()
        return MeshRenderer.Material(material_str)

    def set_material(self, material: MeshRenderer.Material) -> None:
        """Sets a new material for the surface."""
        material = MeshRenderer.Material(material)
        self.get_surface_structure().set_material(material.value)

    def push_material(self, material: MeshRenderer.Material) -> None:
        """Pushes a new material onto the stack."""
        material = MeshRenderer.Material(material)
        self._material.append(self.get_material())
        self.get_surface_structure().set_material(material.value)

    def pop_material(self) -> None:
        """Pops the current material from the stack."""
        material = self._material[-1]
        self.get_surface_structure().set_material(material.value)
        self._material.pop()

    def get_smooth_shading(self) -> bool:
        """Return whether the surface is smooth shaded."""
        return self.get_surface_structure().get_smooth_shade()

    def set_smooth_shading(self, smooth: bool) -> None:
        """Sets the shading style for the surface."""
        smooth = bool(smooth)
        self.get_surface_structure().set_smooth_shade(smooth)

    def push_smooth_shading(self, smooth: bool) -> None:
        """Pushes a new shading style onto the stack."""
        smooth = bool(smooth)
        self._smooth_shading.append(self.get_smooth_shading())
        self.get_surface_structure().set_smooth_shade(smooth)

    def pop_smooth_shading(self) -> None:
        """Pops the current shading style from the stack."""
        smooth = self._smooth_shading[-1]
        self.get_surface_structure().set_smooth_shade(smooth)
        self._smooth_shading.pop()
