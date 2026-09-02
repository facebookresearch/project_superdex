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


class Logger(ABC):
    """
    Base class for all logger objects in the Mochi rerun integration.

    Loggers are responsible for converting Mochi data structures to rerun
    entities and logging them to the active recording. This abstract base class
    provides common functionality for managing logger properties such as entity
    path and coordinate system transformations. Subclasses must implement the
    log() and clear() methods to handle their specific logging logic.
    """

    ####################################################################################
    # Members
    ####################################################################################

    _entity_path: str
    """Entity path for this logger in the rerun hierarchy."""
    _coordinate_transform: CoordinateTransform
    """Coordinate system converter for transforming between coordinate systems."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        entity_path: str,
        coordinate_transform: CoordinateTransform,
    ):
        """Initialize the base logger.

        Args:
            entity_path: Entity path for this logger in the rerun hierarchy.
            coordinate_transform: Converter for coordinate system transformations.
        """
        self._entity_path = entity_path
        self._coordinate_transform = coordinate_transform

    ####################################################################################
    # Abstract Methods
    ####################################################################################

    @abstractmethod
    def log(self, static: bool = False):
        """Log the current state to rerun.

        Args:
            static: If True, log as static (timeless) data.
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear/remove this entity from rerun."""
        pass

    ####################################################################################
    # Common Properties
    ####################################################################################

    def get_entity_path(self) -> str:
        """Returns the entity path of the logger."""
        return self._entity_path

    def get_coordinate_transform(self) -> CoordinateTransform:
        """Returns the current coordinate system transform."""
        return self._coordinate_transform
