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

"""
Base updater class for UnrealCV-based scene synchronization.

This class provides the abstract interface for all UnrealCV updaters,
following the same patterns as the Polyscope viewer's Renderer base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from superdex.physics.utils.coordinate_systems import CoordinateTransform

if TYPE_CHECKING:
    from ..unrealcv_client import UnrealCVClient

########################################################################################


class UnrealCVUpdater(ABC):
    """
    Base class for all updater objects in the UnrealCV viewer.

    This abstract base class provides common functionality for managing updater
    properties and UnrealCV communication. Subclasses must implement the remove()
    method to handle their specific update logic.
    """

    ####################################################################################
    # Members
    ####################################################################################

    _name: str
    """Name identifier for this updater (mochi actor name)."""

    _ue_actor_name: str
    """Name of the corresponding actor in Unreal Engine."""

    _client: "UnrealCVClient"
    """Reference to the UnrealCV client for sending commands."""

    _coordinate_transform: CoordinateTransform
    """Coordinate system converter for transforming between coordinate systems."""

    _is_visible: bool
    """Whether this updater's object is currently visible in UE."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        name: str,
        ue_actor_name: str,
        client: "UnrealCVClient",
        coordinate_transform: CoordinateTransform,
    ):
        """
        Initialize the base updater.

        Args:
            name: Name identifier for this updater (mochi actor name).
            ue_actor_name: Name of the corresponding actor in Unreal Engine.
            client: Reference to the UnrealCV client.
            coordinate_transform: Converter for coordinate system transformations.
        """
        self._name = name
        self._ue_actor_name = ue_actor_name
        self._client = client
        self._coordinate_transform = coordinate_transform
        self._is_visible = True

    ####################################################################################
    # Abstract Methods
    ####################################################################################

    @abstractmethod
    def remove(self):
        """Remove/hide the updater in Unreal Engine. Must be implemented by subclasses."""
        pass

    ####################################################################################
    # Common Properties and Methods
    ####################################################################################

    def get_name(self) -> str:
        """Returns the name of the updater (mochi actor name)."""
        return self._name

    def get_ue_actor_name(self) -> str:
        """Returns the name of the actor in Unreal Engine."""
        return self._ue_actor_name

    def get_coordinate_transform(self) -> CoordinateTransform:
        """Returns the current coordinate system transform."""
        return self._coordinate_transform

    def is_visible(self) -> bool:
        """Returns whether the updater's object is currently visible."""
        return self._is_visible

    def set_visible(self, visible: bool):
        """
        Sets the visibility of the updater's object in UE.

        Args:
            visible: Whether the object should be visible.
        """
        if visible != self._is_visible:
            self._client.set_object_visibility(self._ue_actor_name, visible)
            self._is_visible = visible
