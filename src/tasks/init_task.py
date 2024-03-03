from __future__ import annotations
import copy
import re
from typing import Union
import importlib_resources
from itertools import product
from omegaconf import OmegaConf
from .utils.all_vars import _ALL_VARS

from .meta.base import MetaTaskBase
from .sim.inventory import InventoryItem
from .sim.sim import MineDojoSim
from .sim.wrappers import FastResetWrapper, ARNNWrapper
from .meta import (
    HarvestMeta,
    CombatMeta,
    TechTreeMeta,
    Playthrough,
    SurvivalMeta,
    CreativeMeta,
)
SUBGOAL_DISTANCE = 10


def _resource_file_path(file_name) -> str:
    """Retrieves the absolute file path of a specified resource file within the package."""
    with importlib_resources.path("minedojo.tasks.description_files", file_name) as p:
        return str(p)


_MetaTaskName2Class = {
    "Open-Ended": MineDojoSim,
    "Harvest": HarvestMeta,
    "Combat": CombatMeta,
    "TechTree": TechTreeMeta,
    "Playthrough": Playthrough,
    "Survival": SurvivalMeta,
    "Creative": CreativeMeta,
}
# Transform all keys to lower case
MetaTaskName2Class = {k.lower(): v for k, v in _MetaTaskName2Class.items()}


def _meta_task_make(meta_task: str, *args, **kwargs) -> Union[MetaTaskBase, FastResetWrapper]:
    """
    Gym-style making tasks from names.

    Args:
        meta_task: Name of the meta task. Can be one of (and their lower-cased equivalents)
        ``"Open-Ended"``, ``"Harvest"``, ``"Combat"``, ``"TechTree"``,``"Playthrough"``, ``"Survival"``, ``"Creative"``.

        *args and **kwargs: See corresponding task's docstring for more info.
    Returns:
        A task object.
    """
    meta_task = meta_task.lower()
    assert meta_task in MetaTaskName2Class, f"Invalid meta task name provided {meta_task}"  # check existence

    # 忽略open-ended
    if meta_task == "open-ended" and kwargs.get("fast_reset"):
        # 删除相关的参数
        fast_reset = kwargs.pop("fast_reset")
        fast_reset_random_teleport_range = kwargs.pop(
            "fast_reset_random_teleport_range", None
        )
        fast_reset_random_teleport_range_high = kwargs.pop(
            "fast_reset_random_teleport_range_high", None
        )
        fast_reset_random_teleport_range_low = kwargs.pop(
            "fast_reset_random_teleport_range_low", None
        )
        # 上面前2个暂时不用，可省略
        if fast_reset is True:
            return FastResetWrapper(
                MineDojoSim(*args, **kwargs), fast_reset_random_teleport_range_high, fast_reset_random_teleport_range_low
            )

    return MetaTaskName2Class[meta_task](*args, **kwargs)


def product_dict(**kwargs):
    """
    Generate all possible combinations of key-value pairs from the input dictionary.

    Yields:
    dict: A dictionary containing one combination of key-value pairs.
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

_ALL_TASKS_SPECS_UNFILLED = OmegaConf.load(
    _resource_file_path("tasks_specs.yaml"))
# check no duplicates
assert len(set(_ALL_TASKS_SPECS_UNFILLED.keys())) == len(_ALL_TASKS_SPECS_UNFILLED)


ALL_TASKS_SPECS = {}
for task_id, task_specs in _ALL_TASKS_SPECS_UNFILLED.items():
    unfilled_vars = re.findall(r"\{(.*?)\}", task_id)

    def _recursive_find_unfilled_vars(x):
        if OmegaConf.is_dict(x):
            return {k: _recursive_find_unfilled_vars(v) for k, v in x.items()}
        elif isinstance(x, str):
            unfilled_vars.extend(re.findall(r"\{(.*?)\}", x))
        return x

    _recursive_find_unfilled_vars(task_specs)

    # deduplicate unfilled vars
    unfilled_vars = list(set(unfilled_vars))
    if len(unfilled_vars) == 0:
        # no unfilled vars, just make the task
        ALL_TASKS_SPECS[task_id] = task_specs
    else:
        unfilled_vars_values = {var: _ALL_VARS[var] for var in unfilled_vars}
        for var_dict in product_dict(**unfilled_vars_values):
            filled_task_id = task_id.format(**var_dict)

            def _recursive_replace_var(x):
                if OmegaConf.is_dict(x):
                    return {k: _recursive_replace_var(v) for k, v in x.items()}
                elif isinstance(x, str):
                    return x.format(**var_dict)
                return x

            task_specs_filled = _recursive_replace_var(task_specs)

            for k, v in task_specs_filled.items():
                if k == "target_quantities":
                    task_specs_filled[k] = int(v)
            task_specs_filled["prompt"] = task_specs_filled["prompt"].replace(
                "_", " ")
            task_specs_filled["prompt"] = task_specs_filled["prompt"].replace(
                " 1", "")

            ALL_TASKS_SPECS[filled_task_id] = task_specs_filled

# check no duplicates
assert len(set(ALL_TASKS_SPECS.keys())) == len(ALL_TASKS_SPECS)

# load prompts and guidance for creative tasks and check no duplicates
C_TASKS_PROMPTS_GUIDANCE = OmegaConf.load(
    _resource_file_path("creative_tasks.yaml"))
assert len(set(C_TASKS_PROMPTS_GUIDANCE.keys())
           ) == len(C_TASKS_PROMPTS_GUIDANCE)
# load prompts and guidance for programmatic tasks and check no duplicates
P_TASKS_PROMPTS_GUIDANCE = OmegaConf.load(
    _resource_file_path("programmatic_tasks.yaml"))
assert len(set(P_TASKS_PROMPTS_GUIDANCE.keys())
           ) == len(P_TASKS_PROMPTS_GUIDANCE)
# load prompt and guidance for Playthrough task and check only one playthrough task
PLAYTHROUGH_PROMPT_GUIDANCE = OmegaConf.load(
    _resource_file_path("playthrough_task.yaml"))
assert len(PLAYTHROUGH_PROMPT_GUIDANCE.keys()) == 1

ALL_PROGRAMMATIC_TASK_IDS = list(P_TASKS_PROMPTS_GUIDANCE.keys())
ALL_PROGRAMMATIC_TASK_INSTRUCTIONS = {
    task_id: (
        P_TASKS_PROMPTS_GUIDANCE[task_id]["prompt"],
        P_TASKS_PROMPTS_GUIDANCE[task_id]["guidance"],
    )
    for task_id in ALL_PROGRAMMATIC_TASK_IDS
}
ALL_CREATIVE_TASK_IDS = list(C_TASKS_PROMPTS_GUIDANCE.keys())
ALL_CREATIVE_TASK_INSTRUCTIONS = {
    task_id: (
        C_TASKS_PROMPTS_GUIDANCE[task_id]["prompt"],
        C_TASKS_PROMPTS_GUIDANCE[task_id]["guidance"],
    )
    for task_id in ALL_CREATIVE_TASK_IDS
}
PLAYTHROUGH_TASK_ID = list(PLAYTHROUGH_PROMPT_GUIDANCE.keys())[0]
PLAYTHROUGH_TASK_INSTRUCTION = (
    PLAYTHROUGH_PROMPT_GUIDANCE[PLAYTHROUGH_TASK_ID]["prompt"],
    PLAYTHROUGH_PROMPT_GUIDANCE[PLAYTHROUGH_TASK_ID]["guidance"],
)
ALL_TASK_IDS = ALL_PROGRAMMATIC_TASK_IDS + ALL_CREATIVE_TASK_IDS + [PLAYTHROUGH_TASK_ID]
ALL_TASK_INSTRUCTIONS = {
    **ALL_PROGRAMMATIC_TASK_INSTRUCTIONS,
    **ALL_CREATIVE_TASK_INSTRUCTIONS,
    PLAYTHROUGH_TASK_ID: PLAYTHROUGH_TASK_INSTRUCTION,
}


def _parse_inventory_dict(inv_dict: dict[str, dict]) -> list[InventoryItem]:
    """
    Parses a dictionary representing inventory data and returns a list of InventoryItem objects.

    Parameters:
    - inv_dict (dict[str, dict]): A dictionary where keys are strings and values are dictionaries.
      Represents inventory data with slot information and corresponding attributes.

    Returns:
    - list[InventoryItem]: A list of InventoryItem objects created from the input dictionary.
    """
    return [InventoryItem(slot=k, **v) for k, v in inv_dict.items()]


def _specific_task_make(task_id: str, *args, **kwargs):
    """
    For Programmatic tasks and playthrough task.
    """
    assert task_id in ALL_TASKS_SPECS, f"Invalid task id provided {task_id}"
    # COMPLETED copy to avoid modifying the original task specs using deepcopy
    task_specs = copy.deepcopy(ALL_TASKS_SPECS[task_id])
    # handle list of inventory items
    if "initial_inventory" in task_specs:
        # allow specify initial inventory in kwargs
        if not "initial_inventory" in kwargs:
            kwargs["initial_inventory"] = _parse_inventory_dict(
                task_specs["initial_inventory"]
            )
            print('warning: the default initial_inventory is modified.')
        task_specs.pop("initial_inventory")

    # pop prompt from task specs because it is set from programmatic yaml
    task_specs.pop("prompt")    # delete prompt and get its value from yaml

    # meta task
    meta_task_cls = task_specs.pop("__cls__")
    # print('{}\n{}\n{}\n{}'.format(meta_task_cls, args, task_specs, kwargs))
    for key in kwargs:
        if key in task_specs:
            task_specs.pop(key)
            print(
                'warning: the default programmatic task argument {} is modified'.format(key))
            
    # print(task_specs, kwargs)
    task_obj = _meta_task_make(meta_task_cls, *args, **task_specs, **kwargs)
    return task_obj


def make(task_id: str, *args, cam_interval: Union[int, float] = 15, **kwargs):
    """
    Make a task. task_id can be one of the following:
    1. a task id for Programmatic tasks, e.g., "combat_bat_extreme_hills_barehand"
    2. format "creative:{idx}" for the idx-th Creative task
    3. "playthrough" or "open-ended" for these two special tasks
    4. one of "harvest", "combat", "techtree", and "survival" to creative meta task
    """
    if task_id.startswith("creative:"):
        creative_idx = int(task_id.split(":")[1])
        assert len(C_TASKS_PROMPTS_GUIDANCE) > creative_idx >= 0
        info = C_TASKS_PROMPTS_GUIDANCE[task_id]
        env_obj = _meta_task_make("creative", *args, **kwargs)
        env_obj.specify_prompt(info["prompt"])
        env_obj.specify_guidance(info["guidance"])
        env_obj.collection = info["collection"]
        env_obj.source = info["source"]
    elif task_id.lower() == PLAYTHROUGH_TASK_ID.lower():
        info = PLAYTHROUGH_PROMPT_GUIDANCE[PLAYTHROUGH_TASK_ID]
        env_obj = _specific_task_make(task_id, *args, **kwargs)
        env_obj.specify_prompt(info["prompt"])
        env_obj.specify_guidance(info["guidance"])
    # main tasks
    elif task_id in P_TASKS_PROMPTS_GUIDANCE:
        info = P_TASKS_PROMPTS_GUIDANCE[task_id]
        env_obj = _specific_task_make(task_id, *args, **kwargs)
        env_obj.specify_prompt(info["prompt"])
        env_obj.specify_guidance(info["guidance"])
    elif task_id.lower() in [
        "open-ended",
        "harvest",
        "combat",
        "techtree",
        "survival",
    ]:
        env_obj = _meta_task_make(meta_task=task_id, *args, **kwargs)
    else:
        raise ValueError(f"Invalid task id provided {task_id}")

    return ARNNWrapper(env_obj, cam_interval=cam_interval)
