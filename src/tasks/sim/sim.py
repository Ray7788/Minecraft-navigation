import uuid
from copy import deepcopy
from typing import Union, Optional, List, Dict, Tuple, Literal, Any

import minerl
import cv2
import gym
import numpy as np
from lxml import etree

from .handlers.agents.observations.TrueFlatInventoryObservation import EquipmentObservation
from .handlers.agents.observations.pov import POVObservation
from .handlers.agents.observations.voxel_lidar import RichLidarObservation, VoxelObservation

from .handlers.agents.observations.TrueFlatInventoryObservation import EquipmentObservation, TrueFlatInventoryObservation
from .handlers.agents.observations.damage_source import ObservationFromDamageSource
from .handlers.agents.observations.lifestats import ObservationFromLifeStats
from .handlers.agents.observations.location_stats import ObservationFromCurrentLocation
from .handlers.agents.observations.nearby_tools import NearbyToolsObservation
from .handlers.agents.observations import ObserveFromFullStats
from .handlers.agents.observations.pov import POVObservation
from .handlers.agents.observations.voxel_lidar import RichLidarObservation, VoxelObservation
# from .handlers.agents.observations import *


from .handlers.agents.actions import CameraAction
from .handlers.agents.actions import SmeltAction
from .handlers.agents.actions import PlaceBlock
from .handlers.agents.actions import CraftAction
from .handlers.agents.actions import CraftWithTableAction
from .handlers.agents.actions import EquipAction
from .handlers.agents.actions import KeybasedCommandAction
from .handlers.agents.actions import SwapSlotAction
# from .handlers.agents.actions import *

from . import handlers

from minerl.herobraine.hero import mc
from .bridge import BridgeEnv
from .cmd_executor import CMDExecutor
from .config_sim_spec import SimSpec
from .inventory import InventoryItem, parse_inventory_item


class MineDojoSim(gym.Env):
    """An environment wrapper for MineDojo simulation.

    Args:
        allow_mob_spawn: If ``True``, allow mobs (animals and hostiles) to spawn.
                Default: ``True``.

        allow_time_passage: Time flows if ``True``.
                Default: ``True``.

        break_speed_multiplier: Controls the speed of breaking blocks. A value larger than 1.0 accelerates the breaking.
                Default: ``1.0``.

        drawing_str: Draws shapes (e.g. spheres, cuboids) in the minecraft world.
                Default: ``None``.

        event_level_control: If ``True``, the agent is able to perform high-level controls including place and equip.
                If ``False``, then is keyboard-mouse level control.
                Default: ``True``.

        flat_world_seed_string: A string that specifies how we want the world layers to be created
                if ``generate_world_type`` is "flat".
                One can use the `tool <https://minecraft.tools/en/flat.php?biome=1&bloc_1_nb=1&bloc_1_id=2&bloc_2_nb=2&bloc_2_id=3%2F00&bloc_3_nb=1&bloc_3_id=7&village_size=1&village_distance=32&mineshaft_chance=1&stronghold_count=3&stronghold_distance=32&stronghold_spread=3&oceanmonument_spacing=32&oceanmonument_separation=5&biome_1_distance=32&valid=Create+the+Preset#seed>`_ to generate.
                Default: ``None``.

        generate_world_type: A string that specifies the type of the minecraft world.
                One of ``"default"``, ``"flat"``, ``"from_file"``, ``"specified_biome"``.
                Default: ``"default"``.

        image_size: The size of image observations.

        initial_inventory: If not ``None``, specifies initial items in the agent's inventory.
                Use class ``InventoryItem`` to specify items.
                Default: ``None``.

        initial_weather: If not ``None``, specifies the initial weather.
                Can be one of ``"clear"``, ``"normal"``, ``"rain"``, ``"thunder"``.
                Default: ``None``.

        lidar_rays: Defines the directions and maximum distances of the lidar rays if ``use_lidar`` is ``True``.
                If supplied, should be a list of tuple(pitch, yaw, distance).
                Pitch and yaw are in radians and relative to agent looking vector.
                Default: ``None``.

        raise_error_on_invalid_cmds: If ``True``, the cmd executor will raise error when a command is invalid.
                If ``False``, the executor will just skip instead.
                Default: ``False``.

        regenerate_world_after_reset: If ``True``, the minecraft world will be re-generated when resetting.
                Default: ``False``.

        seed: The seed for an instance's internal random number generator.
                Default: ``None``.

        sim_name: Name of a simulation instance.
                Default: ``"MineDojoSim"``.

        spawn_in_village: If ``True``, the agent will spawn in a village.
                Default: ``False``.

        specified_biome: If not ``None``, specifies the biome of the minecraft world by a string or an integer.
                Default: ``None``.

        start_food: If not ``None``, specifies initial food of the agent.
                Default: ``None``.

        start_health: If not ``None``, specifies initial health of the agent.
                Default: ``None``.

        start_position: If not ``None``, specifies the agent's initial location and orientation.
                If provided, should be a dict with keys ``x``, ``y``, ``z``, ``yaw``, ``pitch``.
                ``yaw`` and ``pitch`` are in degrees.
                Default: ``None``.

        start_time: If not ``None``, specifies the time when the agent spawns.
                If supplied, should be an int between 0 and 24000.
                See `here <https://minecraft.fandom.com/wiki/Daylight_cycle>`_ for more information.
                Default: ``None``.

        use_depth: If ``True``, includes depth map in observations.
                Default: ``False``.

        use_lidar: If ``True``, includes lidar in observations.
                Default: ``False``.

        use_voxel: If ``True``, includes voxel in observations.
                Default: ``False``.

        voxel_size: Defines the voxel's range in each axis if ``use_voxel`` is ``True``.
                If supplied, should be a dict with keys ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``.
                Each value specifies the voxel size relative to the agent.
                Default: ``None``.

        world_file_path: The path to the world file if ``generate_world_type`` is ``"from_file"``.
                Default: ``None``.

        world_seed: Seed for deterministic world generation
                if ``generate_world_type`` is ``"default"`` or ``"specified_biome"``.
                See `here <https://minecraft.fandom.com/wiki/Seed_(level_generation)>`_ for more details.
                Default: ``None``.
    """

    def __init__(
        self,
        *,
        # ------ initial conditions ------
        initial_inventory: Optional[List[InventoryItem]] = None,
        start_position: Optional[Dict[str, Union[float, int]]] = None,
        start_health: Optional[float] = None,
        start_food: Optional[int] = None,
        start_time: Optional[int] = None,
        initial_weather: Optional[Literal["normal", "clear", "rain", "thunder"]] = None,
        spawn_in_village: bool = False, # https://minecraft.fandom.com/wiki/Spawn   Java Edition
        drawing_str: Optional[str] = None,
        # ------ global conditions ------
        break_speed_multiplier: float = 1.0,
        allow_time_passage: bool = True,    # time flows if True
        allow_mob_spawn: bool = True,    # allow mobs (animals and hostiles) to spawn if True
        # ------ world generation ------
        generate_world_type: Literal[
            "default", "from_file", "flat", "specified_biome"
        ] = "default",
        regenerate_world_after_reset: bool = False,   # regenerate world when reset if True
        world_seed: Optional[Union[str, int]] = None,
        world_file_path: Optional[str] = None,
        flat_world_seed_string: Optional[str] = None,
        specified_biome: Optional[Union[int, str]] = None,
        # ------ observation ------
        image_size: Union[int, Tuple[int, int]],
        use_voxel: bool = False,
        voxel_size: Optional[Dict[str, int]] = None,
        use_lidar: bool = False,
        lidar_rays: Optional[List[Tuple[float, float, float]]] = None,
        use_depth: bool = False,    # if True, includes depth map in observations
        # ------ control ------
        event_level_control: bool = True,   # if True, the agent is able to perform high-level controls including place and equip???????
        # ------ randomness ------
        seed: Optional[int] = None,
        # ------ misc ------
        sim_name: str = "MineDojoSim",
        raise_error_on_invalid_cmds: bool = False,
    ):
        self._sim_name = sim_name
        self._rng = np.random.default_rng(seed) # random number generator
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if use_voxel:
            voxel_size = voxel_size or dict(
                xmin=-1, ymin=-1, zmin=-1, xmax=1, ymax=1, zmax=1
            )
        if use_lidar:
            lidar_rays = lidar_rays or [(0.0, 0.0, 10.0)]
        assert not use_depth, "TODO: fix depth obs bug, can not use depth obs now"
        start_health = start_health or 20.0 # 20.0 is the max health https://minecraft.fandom.com/wiki/Health
        start_food = start_food or 20

# 
        # world type generation --------------------
        assert generate_world_type in {
            "default",
            "from_file",
            "flat",
            "specified_biome",
        }, f"invalid generate world type {generate_world_type}"
        if generate_world_type == "default":
            world_seed = world_seed or ""
        elif generate_world_type == "from_file":
            assert (
                world_file_path is not None
            ), f'A world file path must be provided when `generate_world_type = "from_file"`'
        elif generate_world_type == "flat":
            flat_world_seed_string = flat_world_seed_string or ""
        elif generate_world_type == "specified_biome":
            assert (
                specified_biome is not None
            ), f"must provide a biome when generate_world_type = 'specified_biome'!"
            world_seed = world_seed or ""
        else:
            raise ValueError(f"Unknown world generation type {generate_world_type}")
        
        # Check biome's validity    -------------------- TODO: shorten lines
        if specified_biome is not None:
            if isinstance(specified_biome, str):    # string biome name, transfer to integer biome id
                assert (
                    specified_biome in mc.BIOMES_MAP
                ), f"Unknown biome name {specified_biome}"
                specified_biome = mc.BIOMES_MAP[specified_biome]
            elif isinstance(specified_biome, int):  # integer biome id
                assert (
                    specified_biome in mc.BIOMES_MAP.values()
                ), f"Invalid biome id {specified_biome}"
            else:
                raise ValueError(f"invalid biome type {specified_biome}")

        # configure observation handlers---------------------------------------------------------------------------
        # create a list of observation handlers
        obs_handlers = [
            POVObservation(image_size, False),
            TrueFlatInventoryObservation(),
            ObservationFromCurrentLocation(),
            EquipmentObservation(),
            ObservationFromLifeStats(),            
            ObserveFromFullStats(),
            NearbyToolsObservation(),
            ObservationFromDamageSource(),
        ]

        if use_voxel:   
            voxel_size = (
                (voxel_size["xmin"], voxel_size["xmax"]),
                (voxel_size["ymin"], voxel_size["ymax"]),
                (voxel_size["zmin"], voxel_size["zmax"]),
            )
            obs_handlers.append(VoxelObservation(voxel_size))
        if use_lidar:
            obs_handlers.append(RichLidarObservation(lidar_rays))

# 
        # configure action handlers
        common_actions = [  # TODO: Replace with minerl action space in hero.mc (not in v0.4.4)
            "forward",
            "back",
            "left",
            "right",
            "jump",
            "sneak",
            "sprint",
            "use",
            "attack",
            "drop", # new  action
        ]
        action_handlers = [
            CameraAction(),
            SmeltAction(
                ["none"] + mc.ALL_SMELTING_ITEMS, _other="none", _default="none"    # TODO!!
            ),
            CraftAction(
                ["none"] + mc.ALL_PERSONAL_CRAFTING_ITEMS,
                _other="none",
                _default="none",
            ),
            CraftWithTableAction(
                ["none"] + mc.ALL_CRAFTING_TABLE_ITEMS, _other="none", _default="none"
            ),
        ]
        action_handlers.extend(
            [
                KeybasedCommandAction(k, mc.INVERSE_KEYMAP[k])
                for k in common_actions
            ]
        )
        
        # level control
        if event_level_control:
            action_handlers.extend(
                [
                    EquipAction(
                        ["none"] + mc.ALL_ITEMS, _other="none", _default="none"
                    ),
                    PlaceBlock(
                        ["none"] + mc.ALL_ITEMS, _other="none", _default="none"
                    ),
                ]
            )
        else:
            action_handlers.append(SwapSlotAction())
            action_handlers.append(
                KeybasedCommandAction(
                    "pickItem", mc.INVERSE_KEYMAP["pickItem"]
                )
            )
            action_handlers.extend(
                [
                    KeybasedCommandAction(
                        f"hotbar.{i}", mc.INVERSE_KEYMAP[str(i)]
                    )
                    for i in range(1, 10)
                ]
            )

# ----------------------------------------------------------------------------------------------
        # configure agent handlers
        agent_handlers = []
        # configure agent start handlers, e.g., initial inventory
        self.start_health, self.start_food = start_health, start_food
        agent_start_handlers = [
            handlers.LowLevelInputsAgentStart(),
            handlers.AgentStartBreakSpeedMultiplier(break_speed_multiplier),
            handlers.StartingHealthAgentStart(health=start_health),
            handlers.StartingFoodAgentStart(food=start_food),
        ]

        # configure initial inventory handlers 
        self.initial_inventory = initial_inventory
        if initial_inventory is not None:
            initial_inventory = [
                parse_inventory_item(item) for item in initial_inventory
            ]
            agent_start_handlers.append(
                handlers.InventoryAgentStart(
                    # Create a dict mapping slot to item metadata
                    {
                        inventory_item[0]: inventory_item[1]
                        for inventory_item in initial_inventory
                    }
                )
            )
            
        # configure agent start position handlers
        self.start_position = start_position
        if start_position is not None:
            agent_start_handlers.append(
                handlers.AgentStartPlacement(
                    x=start_position["x"],
                    y=start_position["y"],
                    z=start_position["z"],
                    yaw=start_position["yaw"],
                    pitch=start_position["pitch"],
                )
            )

        # configure server initial conditions handlers
        self.start_time = start_time
        server_initial_conditions_handlers = [
            handlers.TimeInitialCondition(
                allow_passage_of_time=allow_time_passage, start_time=start_time
            ),
            handlers.SpawningInitialCondition(allow_mob_spawn),
        ]
        self.initial_weather = initial_weather
        if initial_weather is not None:
            server_initial_conditions_handlers.append(
                handlers.WeatherInitialCondition(initial_weather)
            )

        # configure world generator handlers
        world_generator_handlers = []
        if generate_world_type == "default":
            world_generator_handlers.append(
                handlers.DefaultWorldGenerator(regenerate_world_after_reset, world_seed)
            )
        elif generate_world_type == "from_file":
            world_generator_handlers.append(
                handlers.FileWorldGenerator(
                    world_file_path,
                    destroy_after_use=False,
                )
            )
        elif generate_world_type == "flat":
            world_generator_handlers.append(
                handlers.FlatWorldGenerator(
                    force_reset=True, generatorString=flat_world_seed_string
                )
            )
        elif generate_world_type == "specified_biome":
            world_generator_handlers.append(
                handlers.BiomeGenerator(
                    specified_biome, force_reset=True, world_seed=world_seed
                )
            )
        else:
            raise ValueError()
        if drawing_str is not None:
            world_generator_handlers.append(handlers.DrawingDecorator(drawing_str))

        # configure server decorator handlers
        server_decorator_handlers = []
        if spawn_in_village:
            server_decorator_handlers.append(handlers.VillageSpawnDecorator())

        # configure server quit handlers
        server_quit_handlers = []

        # configure sim spec
        self._sim_spec = SimSpec(
            sim_name=sim_name,
            agent_count=1,
            obs_handlers=obs_handlers,
            action_handlers=action_handlers,
            agent_handlers=agent_handlers,
            agent_start_handlers=agent_start_handlers,
            server_initial_conditions_handlers=server_initial_conditions_handlers,
            world_generator_handlers=world_generator_handlers,
            server_decorator_handlers=server_decorator_handlers,
            server_quit_handlers=server_quit_handlers,
            seed=self.new_seed,
        )

        self._prev_obs = None
        self._prev_action = None
        self._prev_info = None

# TODO: check these two lines' dependency
        self._bridge_env = BridgeEnv(is_fault_tolerant=True, seed=self.new_seed)
        self._cmd_executor = CMDExecutor(self, raise_error_on_invalid_cmds)

    @property
    def observation_space(self):
        return self._sim_spec.observation_space

    @property
    def action_space(self):
        return self._sim_spec.action_space

    @property
    def new_seed(self):
        return self._rng.integers(low=0, high=2**31 - 1).item()

    def seed(self, seed: int = None):
        """Sets the seed for this env's random number generator.

        Args:
            seed: The seed for the number generator
        """
        self._rng = np.random.default_rng(seed)

    def reset(self):
        """Resets the environment to an initial state and returns an initial observation.

        Return:
            Agent’s initial observation.
        """
        episode_id = str(uuid.uuid4())  # generate a random uuid

        xml = etree.fromstring(self._sim_spec.to_xml(episode_id))
        raw_obs = self._bridge_env.reset(episode_id, [xml])[0]
        obs, info = self._process_raw_obs(raw_obs)
        self._prev_obs, self._prev_info = deepcopy(obs), deepcopy(info)
        return obs

    def step(self, action: dict):
        """Run one timestep of the environment's dynamics. Accepts an action and returns next_obs, reward, done, info.

        Args:
            action: The action of the agent in current step.

        Return:
            A tuple (obs, reward, done, info)
            - ``dict`` - Agent’s next observation.
            - ``float`` - Amount of reward returned after executing previous action.
            - ``bool`` - Whether the episode has ended.
            - ``dict`` - Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        self._prev_action = deepcopy(action)
        action_xml = self._action_obj_to_xml(action)
        step_tuple = self._bridge_env.step([action_xml])
        step_success, raw_obs = step_tuple.step_success, step_tuple.raw_obs
        if not step_success:
            # when step failed, return prev obs, no reward, end of episode, and prev info
            return self._prev_obs, 0, True, self._prev_info
        else:
            obs, info = self._process_raw_obs(raw_obs[0])
            self._prev_obs, self._prev_info = deepcopy(obs), deepcopy(info)
            return obs, 0, self.is_terminated, info

    def execute_cmd(self, cmd: str, action: Optional[dict] = None):
        """Execute a given string command.

        Args:
            cmd: The string command accepted by the Minecraft client.
            action: An action that will be simultaneously executed with the command.

        Return:
            A tuple (obs, reward, done, info) representing the step result after executing the command.       
        """
        return self._cmd_executor.execute_cmd(cmd, action)

    def spawn_mobs(
        self,
        mobs: Union[str, List[str]],
        rel_positions: Union[np.ndarray, list],
        action: Optional[dict] = None,
    ):
        """Spawn mobs in the world.

        Args:
            mobs: The names of the mobs to spawn
            rel_positions: The mobs' positions relative to the agent
            action: An action that will be simultaneously executed with the spawning
        Return:
            A tuple (obs, reward, done, info)
        """
        return self._cmd_executor.spawn_mobs(mobs, rel_positions, action)

    def set_block(
        self,
        blocks: Union[str, List[str]],
        rel_positions: Union[np.ndarray, list],
        action: Optional[dict] = None,
    ):
        """Set blocks in the world.

        Args:
            blocks: The names of the blocks to set
            rel_positions: The blocks' positions relative to the agent
            action: An action that will be simultaneously executed with the setting
        Return:
            A tuple (obs, reward, done, info)
        """
        return self._cmd_executor.set_block(blocks, rel_positions, action)

    def clear_inventory(self, action: Optional[dict] = None):
        """Remove all items in the agent's inventory.

        Args:
            action: An action that will be simultaneously executed
        Return:
            A tuple (obs, reward, done, info)
            - ``dict`` - Agent’s observation of the current environment.
            - ``float`` - Amount of reward returned after previous action.
            - ``bool`` - Whether the episode has ended.
            - ``dict`` - Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        return self._cmd_executor.clear_inventory(action)

    def set_inventory(
        self, inventory_list: List[InventoryItem], action: Optional[dict] = None
    ):
        """Set items to the agent's inventory.

        Args:
            inventory_list: List of ``InventoryItem`` to change the inventory status
            action: An action that will be simultaneously executed
        Return:
            A tuple (obs, reward, done, info)
        """
        return self._cmd_executor.set_inventory(inventory_list, action)

    def teleport_agent(self, x, y, z, yaw, pitch, action: Optional[dict] = None):
        """Teleport the agent to a given position.

        Args:
            x: x coordinate of the destination
            y: y coordinate of the destination
            z: z coordinate of the destination
            yaw: yaw of the targeted orientation
            pitch: pitch of the targeted orientation
            action: An action that will be simultaneously executed with the teleporting
        Return:
            A tuple (obs, reward, done, info)
            - ``dict`` - Agent’s observation of the current environment.
            - ``float`` - Amount of reward returned after previous action.
            - ``bool`` - Whether the episode has ended.
            - ``dict`` - Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        return self._cmd_executor.teleport_agent(x, y, z, yaw, pitch, action)

    def kill_agent(self, action: Optional[dict] = None):
        """Kill the agent.

        Args:
            action: An action that will be simultaneously executed
        Return:
            A tuple (obs, reward, done, info)
            - ``dict`` - Agent’s observation of the current environment.
            - ``float`` - Amount of reward returned after previous action.
            - ``bool`` - Whether the episode has ended.
            - ``dict`` - Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        return self._cmd_executor.kill_agent(action)

    def set_time(self, time: int, action: Optional[dict] = None):
        """Set the world with the given time.

        Args:
            time: The target time
            action: An action that will be simultaneously executed
        Return:
            A tuple (obs, reward, done, info)
            - ``dict`` - Agent’s observation of the current environment.
            - ``float`` - Amount of reward returned after previous action.
            - ``bool`` - Whether the episode has ended.
            - ``dict`` - Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        return self._cmd_executor.set_time(time, action)

    def set_weather(self, weather: str, action: Optional[dict] = None):
        """Set the world with the given weather.

        Args:
            weather: The target weather
            action: An action that will be simultaneously executed
        Return:
            A tuple (obs, reward, done, info)
            - ``dict`` - Agent’s observation of the current environment.
            - ``float`` - Amount of reward returned after previous action.
            - ``bool`` - Whether the episode has ended.
            - ``dict`` - Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        return self._cmd_executor.set_weather(weather, action)

    def random_teleport(self, max_range: int, action: Optional[dict] = None):
        """Teleport the agent randomly.

        Args:
            max_range: The maximum distance on each horizontal axis from the center of the area to spread targets
                       (thus, the area is square, not circular)
            action: An action that will be simultaneously executed
        Return:
            A tuple (obs, reward, done, info)
            - ``dict`` - Agent’s observation of the current environment.
            - ``float`` - Amount of reward returned after previous action.
            - ``bool`` - Whether the episode has ended.
            - ``dict`` - Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        return self._cmd_executor.random_teleport(max_range, action)

    def close(self):
        """Environments will automatically close() themselves when garbage collected or when the program exits."""
        self._bridge_env.close()

    def render(self, mode: str = "human"):
        """Renders the environment.

        Args:
            mode: The mode to render with.
        """
        img = self._prev_obs["rgb"]
        img = img.transpose((1, 2, 0))
        img = img[:, :, ::-1]
        cv2.imshow(f"{self._sim_name}", img)
        cv2.waitKey(1)  # wait for 1ms

    @property
    def prev_obs(self):
        return self._prev_obs

    @property
    def prev_info(self):
        return self._prev_info

    @property
    def prev_action(self):
        return self._prev_action

    @property
    def is_terminated(self):
        return self._bridge_env.is_terminated

    def _process_raw_obs(self, raw_obs: dict):
        """
        Process raw obs from the bridge env to a dict of obs and info.
        """
        info = deepcopy(raw_obs)
        if "pov" in info:
            info.pop("pov")
        obs_dict = {
            h.to_string(): h.from_hero(raw_obs) for h in self._sim_spec.observables
        }
        return obs_dict, info

    def _action_obj_to_xml(self, action):
        parsed_action = [f'chat {action["chat"]}'] if "chat" in action else []
        parsed_action.extend(
            [h.to_hero(action[h.to_string()]) for h in self._sim_spec.actionables]
        )
        return "\n".join(parsed_action)