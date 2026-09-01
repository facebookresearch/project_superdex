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

import enum
from contextlib import ExitStack
from typing import Callable, cast, TypeVar

import numpy as np
import numpy.typing as npt
from superdex.physics.viewer.backend import polyscope_imgui as psim
from superdex.physics.viewer.ui import styling

########################################################################################


def ndarray_inspector(  # noqa: C901
    name: str,
    values: npt.NDArray[float],
    lower_limits: npt.NDArray[float] | None = None,
    upper_limits: npt.NDArray[float] | None = None,
    read_only: bool = False,
    fmt: str = "%f",
) -> None:
    """
    Builds a UI for ndarray values. Each value in the array will be shown as a separate
    input field, denoted by the name of the array and the index of the value. If limits
    are provided, finite-ranged values will be shown as sliders. The rest of values will
    be shown as input fields. Supported dtypes are integers and floats.
    """

    # For simplicity, use multi-index iterator.
    ITERATOR_FLAGS: list[str] = ["multi_index"]

    FLAGS = psim.ImGuiInputTextFlags_ReadOnly if read_only else 0

    # Determine, based on the values dtype, which ImGui functions to use.
    if np.issubdtype(values.dtype, np.integer):

        def InputFn(label: str, v: float) -> tuple[bool, float]:
            return psim.InputInt(label, int(v), flags=FLAGS)

        def SliderFn(
            label: str, v: float, v_min: float, v_max: float
        ) -> tuple[bool, float]:
            return psim.SliderInt(label, int(v), v_min=v_min, v_max=v_max, format=fmt)
    elif np.issubdtype(values.dtype, np.floating):

        def InputFn(label: str, v: float) -> tuple[bool, float]:
            return psim.InputFloat(label, v, format=fmt, flags=FLAGS)

        def SliderFn(
            label: str, v: float, v_min: float, v_max: float
        ) -> tuple[bool, float]:
            return psim.SliderFloat(label, v, v_min=v_min, v_max=v_max, format=fmt)
    else:
        raise TypeError("Unsupported dtype. Only integer and float are supported.")

    # Build the ndarray inspector UI.
    psim.PushID(name)

    # Simplified path when no limits are provided, or all limits are set to infinity.
    # This will show the values as input fields.
    if (
        lower_limits is None
        or upper_limits is None
        or not np.any(np.isfinite(lower_limits))
        or not np.any(np.isfinite(upper_limits))
    ):
        with np.nditer(values, flags=ITERATOR_FLAGS) as it:
            for value in it:
                label = f"{name}{it.multi_index}"
                changed, new_value = InputFn(label, float(value))
                if changed and not read_only:
                    # Type-checking thinks this is a tuple, but it's actually a ndarray
                    value = cast(npt.NDArray[float], value)
                    value[...] = new_value

    # More complex path when limits are provided. Generate iterators for all values
    # and their corresponding lower and upper limits. Show values as sliders to delimit
    # how far they are from their limits. Note that this only works if its bounded from
    # both sides. If this is not the case, resort to an input field.
    else:
        with ExitStack() as ex:
            it = np.nditer(values, flags=ITERATOR_FLAGS)
            lower_it = np.nditer(lower_limits, flags=ITERATOR_FLAGS)
            upper_it = np.nditer(upper_limits, flags=ITERATOR_FLAGS)
            ex.enter_context(it)
            ex.enter_context(lower_it)
            ex.enter_context(upper_it)
            for value, lo, up in zip(it, lower_it, upper_it):
                label = f"{name}{it.multi_index}"
                if np.isfinite(lo) and np.isfinite(up):
                    changed, new_value = SliderFn(
                        label, float(value), float(lo), float(up)
                    )
                else:
                    changed, new_value = InputFn(label, float(value))
                if changed and not read_only:
                    # Type-checking thinks this is a tuple, but it's actually a ndarray
                    value = cast(npt.NDArray[float], value)
                    value[...] = new_value

    # Done!
    psim.PopID()


def dict_inspector(
    values: dict[str, float], fmt: str = "%f", read_only: bool = False
) -> None:
    """
    Builds a UI for dictionary values. Each key in the dictionary will be shown as a
    separate input field. Supported values are strings, numbers, booleans and numpy
    arrays. Other types will be shown as their string representation.
    """

    # Determine input flags.
    FLAGS = psim.ImGuiInputTextFlags_ReadOnly if read_only else 0

    # Build the dictionary inspector UI.
    # NOTE: Ordering is important here. E.g. `np.bool` is a subclass of `np.integer`.
    for key, value in values.items():
        # Boolean values.
        if isinstance(value, (bool, np.bool_)):
            changed, new_value = psim.Checkbox(key, value)
            if changed and not read_only:
                values[key] = new_value
        # Integer values.
        elif isinstance(value, (int, np.integer)):
            changed, new_value = psim.InputInt(key, value, flags=FLAGS)
            if changed and not read_only:
                psim.InputInt(key, value)
        # Floating point values.
        elif isinstance(value, (float, np.floating)):
            changed, new_value = psim.InputFloat(key, value, format=fmt, flags=FLAGS)
            if changed and not read_only:
                values[key] = new_value
        # NDArrays values.
        elif isinstance(value, np.ndarray):
            ndarray_inspector(key, value, fmt=fmt, read_only=read_only)
        # Strings.
        elif isinstance(value, str):
            changed, new_value = psim.InputText(key, value, flags=FLAGS)
            if changed and not read_only:
                values[key] = new_value
        # Anything else, try get string representation and use that.
        # Note that these will be read-only by default.
        else:
            psim.InputText(key, str(value))


def vertical_block_spacing() -> None:
    """Builds an (invisible) vertical spacing element."""
    psim.Dummy((0, styling.BLOCK_SPACING))


########################################################################################


def bool_property(
    name: str, getter: Callable[[], bool], setter: Callable[[bool], None]
) -> None:
    """
    Helper function to build a boolean property UI.
    """

    changed, new_value = psim.Checkbox(name, getter())
    if changed:
        setter(new_value)


def scalar_property(
    name: str,
    getter: Callable[[], float],
    setter: Callable[[float], None],
    *args,
    **kwargs,
) -> None:
    """
    Helper function to build a scalar property UI.
    """

    changed, new_value = psim.InputFloat(name, getter(), *args, **kwargs)
    if changed:
        setter(new_value)


def slider_property(
    name: str,
    getter: Callable[[], float],
    setter: Callable[[float], None],
    *args,
    **kwargs,
) -> None:
    """
    Helper function to build a scalar property UI with a slider.
    """

    changed, new_value = psim.SliderFloat(name, getter(), *args, **kwargs)
    if changed:
        setter(new_value)


def rgb_color_property(
    name: str,
    getter: Callable[[], npt.NDArray[float]],
    setter: Callable[[npt.NDArray[float]], None],
    *args,
    **kwargs,
) -> None:
    """
    Helper function to build a RGB color property UI.
    """

    changed, new_color = psim.ColorEdit3(name, [*getter()], *args, **kwargs)
    if changed:
        setter(np.asarray(new_color))


E = TypeVar("E", bound=enum.Enum)


def enum_property(
    name: str,
    getter: Callable[[], E],
    setter: Callable[[E], None],
    enum_type: type[E],
    *args,
    **kwargs,
) -> None:
    """
    Helper function to build an enum property UI.
    """

    enum_values = list(enum_type)
    enum_labels = [e.name for e in enum_type]
    current_value = getter()
    current_index = enum_values.index(current_value)
    changed, new_index = psim.Combo(name, current_index, enum_labels, *args, **kwargs)
    if changed:
        setter(enum_values[new_index])


########################################################################################


def tooltip_on_hover(text: str) -> None:
    """
    Helper function to show a tooltip on hover.
    """

    if psim.IsItemHovered():
        psim.SetTooltip(text)
