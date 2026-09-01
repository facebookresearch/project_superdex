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

from abc import ABC, abstractmethod

from superdex.physics.utils.coordinate_systems import CoordinateTransform

########################################################################################


class Renderer(ABC):
    """
    Base class for all renderer objects in the SuperDex Physics viewer.

    This abstract base class provides common functionality for managing renderer
    properties such as name and coordinate system transformations. Subclasses must
    implement the update() and remove() methods to handle their specific rendering logic.
    """

    ####################################################################################
    # Members
    ####################################################################################

    _name: str
    """Name identifier for this renderer."""
    _coordinate_transform: CoordinateTransform
    """Coordinate system converter for transforming between coordinate systems."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        name: str,
        coordinate_transform: CoordinateTransform,
    ):
        """Initialize the base renderer.

        Args:
            name: Name identifier for this renderer.
            coordinate_transform: Converter for coordinate system transformations.
        """
        self._name = name
        self._coordinate_transform = coordinate_transform

    ####################################################################################
    # Abstract Methods
    ####################################################################################

    @abstractmethod
    def update(self):
        """Update the renderer's visualization. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def remove(self):
        """Remove the renderer and clean up resources. Must be implemented by subclasses."""
        pass

    ####################################################################################
    # Common Properties
    ####################################################################################

    def get_name(self) -> str:
        """Returns the name of the renderer."""
        return self._name

    def get_coordinate_transform(self) -> CoordinateTransform:
        """Returns the current coordinate system transform."""
        return self._coordinate_transform
