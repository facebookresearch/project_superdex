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
A collection of helper classes and functions for Mochi data.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import superdex.physics as sdp
import superdex.robotics as sdr
from superdex.physics.environment import get_env_var_value
from superdex.physics.paths import get_assets_root
from superdex.physics.utils import render_model_registry
from superdex.physics.utils.scene_helpers import (
    create_ground_plane,
    create_scene_from_prefab,
    find_actor,
)

########################################################################################
# Classes for scene initialization
#######################################################################################


class PrefabParams(sdp.prefab.PrefabParams):
    root_dir: str
    """
    Your prefab file can reference other files like models and nested prefabs. It can use
    absolute paths or relative paths. Usually relative paths refer to files in the assets
    directory which is provided by Mochi. Optionally override this field to load relative
    paths from somewhere else.
    """
    agent_actor_name: str
    """
    MochiEnv expects a scene with one special actor called the "agent". If the scene
    contains an articulated actor, then it will be labeled as the "agent" automatically.
    You can override this field if you want to select a different actor. Note that actor
    names may contain forward slashes if they come from nested prefabs.
    E.g. "my_prefab/my_nested_prefab/my_actor".
    """
    add_ground_plane: bool
    """
    If true, a ground plane will be added to the scene at height = 0 and pointing up
    the +Y axis.
    """
    ground_normal: sdp.Real3
    """
    If add_ground_plane is true, this is the normal of the ground plane.
    """
    ground_offset: float
    """
    If add_ground_plane is true, this is the height of the ground plane.
    """
    ground_contact_params: sdp.ContactParams | None
    """
    Contact params used to configure the ground plane. If not provided, default values
    will be used.
    """

    def __init__(self):
        super().__init__()
        self.root_dir = ""
        self.agent_actor_name = ""
        self.add_ground_plane = True
        self.ground_normal = sdp.Real3(0.0, 1.0, 0.0)
        self.ground_offset = 0.0
        self.ground_contact_params = None


########################################################################################
# Functions for scene initialization
#######################################################################################


def init_prefab_scene(
    prefab: str | sdp.prefab.ScenePrefab, params: PrefabParams
) -> [sdp.Scene, sdp.Actor]:
    """
    Initialize a scene using a ScenePrefab object or prefab file path. Any referenced model
    files or nested prefabs will be loaded as necessary. Relative files paths will be resolved
    relative to the assets root unless absolute paths were used in the prefab.
    """
    scene = create_scene_from_prefab(
        prefab=prefab,
        root_dir=params.root_dir,
        params=params,
    )

    # Add the ground plane if the user requested it.
    if params.add_ground_plane:
        create_ground_plane(
            scene,
            params.ground_normal,
            params.ground_offset,
            params.ground_contact_params,
        )

    # Select the "agent" actor.
    # If user specified an actor name pattern, fall back to the regular find_actor.
    # Otherwise, try to auto-select the agent actor from the first articulation.
    if isinstance(params.agent_actor_name, str) and len(params.agent_actor_name) > 0:
        agent = find_actor(scene, params.agent_actor_name)
    else:
        agent = None

        def check_if_agent_actor(actor: sdp.Actor):
            nonlocal agent
            if actor.get_type() == sdp.ActorType.ARTICULATED:
                if agent is not None:
                    raise RuntimeError(
                        "Failed to auto-select the agent actor because the scene "
                        "contains more than one articulation."
                    )
                agent = actor

        scene.for_each_actor(check_if_agent_actor)

    return scene, agent


########################################################################################
# Functions for actor management. Wrappers of the Scene class.
#######################################################################################


def get_actors(scene: sdp.Scene) -> list[sdp.Actor]:
    """Returns pointers to all the actors in the scene."""
    if scene is None:
        raise ValueError("There is no scene.")

    actors = []

    def gather_actor(actor: sdp.Actor):
        actors.append(actor)

    scene.for_each_actor(gather_actor)
    return actors


########################################################################################
# Functions for resolving bot assets.
#######################################################################################

BOTS_ASSETS_PATH_ENV_VAR = "SUPERDEX_BOTS_ASSETS_PATH"
"""Environment variable overriding the directory that contains ``bots/``."""
LEGACY_BOTS_ASSETS_PATH_ENV_VAR = "MOCHI_BOTS_ASSETS_PATH"
"""Deprecated environment variable alias for the bot assets directory."""


def _bot_assets_roots() -> Iterator[Path]:
    """Yields the candidate roots that may contain a ``bots/`` tree, in priority order.

    Deliberately independent of the current working directory, so a bot resolves the same
    way however the process was launched (test runner, notebook, service, wrapper script
    that chdirs).
    """
    # 1. Explicit override: what the build system sets, and the escape hatch for anyone
    #    who relocates the assets.
    env_value = get_env_var_value(
        BOTS_ASSETS_PATH_ENV_VAR, LEGACY_BOTS_ASSETS_PATH_ENV_VAR
    )
    if env_value:
        yield Path(env_value).expanduser()
    # 2. A source checkout: bot assets ship in the top-level Superdex assets directory.
    #    Anchored at __file__, never at the cwd.
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "assets"
        if candidate.is_dir():
            yield candidate
    # 3. A packaged install, where bot assets are laid down under the standard
    #    SuperDex assets root alongside the physics assets.
    try:
        yield get_assets_root()
    except FileNotFoundError:
        pass


def resolve_bot_asset(relative_path: str | Path) -> Path:
    """Resolves a bot asset path relative to the bot assets root.

    Bot assets live in Superdex's top-level ``assets`` tree, outside the physics tree that
    :func:`superdex.physics.paths.resolve_asset` searches, so they get their own
    resolution order (see :func:`_bot_assets_roots`).

    Args:
        relative_path: Path below the bot assets root, e.g.
            ``"bots/hands/dg5f_short/right/dg5f_short_right.superdex_bot"``.

    Returns:
        The absolute path to the asset.

    Raises:
        FileNotFoundError: If the asset is not found under any candidate root.
    """
    relative_path = Path(relative_path)
    searched = []
    for root in _bot_assets_roots():
        candidate = root / relative_path
        if candidate.exists():
            return candidate.resolve()
        searched.append(str(root))
    raise FileNotFoundError(
        f"Could not resolve bot asset '{relative_path}'. Searched: "
        f"{', '.join(searched) if searched else '<no candidate roots>'}. Set "
        f"{BOTS_ASSETS_PATH_ENV_VAR} to the directory containing 'bots/'."
    )


########################################################################################
# Functions for bot lifecycle. Wrappers of superdex.robotics that also register each
# visual render model (.glb) so the default viewer draws the visual meshes instead of
# the collision hulls.
#######################################################################################


def create_bot(
    scene: sdp.Scene,
    bot_prefab: sdr.BotPrefab,
    bots_context: sdr.RoboticsContext | None = None,
) -> sdr.Bot:
    """Creates a bot in the scene and registers each link's visual render model.

    Any environment that creates its agent through this helper automatically gets GLB
    visuals in the default viewer, with no per-env code: the viewer consults the shared
    render-model registry and swaps in a GLB renderer for the registered link actors.

    Args:
        scene: Scene to create the bot in.
        bot_prefab: Bot prefab, typically from
            ``superdex.robotics.load_bot_prefab_from_file``
            (which resolves each link's ``render_model_file`` to an absolute path).
        bots_context: Bots context to use. If None, the process-global context is used.
            ``superdex.robotics.create_context()`` is idempotent: it returns a
            non-owning reference to a single process-global native ``RoboticsContext``,
            creating it on first call, and that context is torn down by the physics
            module at shutdown. Passing None therefore shares the same context rather
            than leaking one per bot.

    Returns:
        The created bot.
    """
    if bots_context is None:
        bots_context = sdr.create_context()
    bot = sdr.create_bot(scene, bot_prefab, bots_context)

    # Register each link's GLB against its link actor. The bot's own (post-sort) prefab
    # links correspond by index to the articulated actor's nested link actors.
    agent = bot.get_articulated_actor()
    if agent is not None:
        link_actors = agent.get_nested_link_actors()
        links = bot.get_bot_prefab().links
        assert len(link_actors) == len(links), (
            f"Bot link count mismatch: {len(link_actors)} link actors vs "
            f"{len(links)} prefab links."
        )
        for link_actor_handle, link in zip(link_actors, links):
            if not link.render_model_file:
                continue
            local_transform = sdp.TransformRT(
                link.render_model_rotation, link.render_model_translation
            )
            render_model_registry.register(
                scene,
                link_actor_handle,
                link.render_model_file,
                local_transform,
                link.render_model_scale,
            )

    return bot


def destroy_bot(scene: sdp.Scene, bot: sdr.Bot) -> None:
    """Unregisters the bot's visual render models, then destroys the bot."""
    agent = bot.get_articulated_actor()
    if agent is not None:
        render_model_registry.unregister_actors(scene, agent.get_nested_link_actors())
    sdr.destroy_bot(scene, bot)


########################################################################################
# Functions for actor data queries. Wrappers of the Actor class.
#######################################################################################


def get_contact_force_and_torque_world(
    actor: sdp.Actor,
) -> npt.NDArray[float]:
    """Gets the contact force and torque on a rigid actor, as a 6D array."""
    return np.concatenate(
        (
            actor.get_contact_force_world(),
            actor.get_contact_torque_world(),
        )
    )


def get_articulated_pose(actor: sdp.Actor) -> npt.NDArray[float]:
    """Gets the pose of an articulated actor."""
    buffer = sdp.DynamicArrayReal(actor.get_num_dofs())
    actor.get_articulated_pose(buffer)
    return np.array(buffer)


def get_articulated_joint_velocities(actor: sdp.Actor) -> npt.NDArray[float]:
    """Gets the velocity of an articulated actor."""
    buffer = sdp.DynamicArrayReal(actor.get_num_dofs())
    actor.get_articulated_joint_velocities(buffer)
    return np.array(buffer)


def get_articulated_dof_limits(actor: sdp.Actor) -> npt.NDArray[float]:
    """Gets the joint limits of an articulated actor."""
    buffer = sdp.DynamicArrayReal2(actor.get_num_dofs())
    actor.get_articulated_dof_limits(buffer)
    return np.array(buffer)


########################################################################################
# Other utility functions
#######################################################################################


def TransformRT_to_numpy(transform: sdp.TransformRT) -> npt.NDArray[float]:
    """Convert a TransformRT to numpy array."""
    return np.stack((transform.translation, transform.rotation.to_rotation_vector()))
