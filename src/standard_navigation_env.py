"""MLG Water Bucket Gym"""
__ref__ = "form https://github.com/minerllabs/minerl/blob/v0.4.4/examples/mlg_wb_specs.py"
__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

from minerl.herobraine.env_specs.navigate_specs import Navigate
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero.handlers.agent.actions.chat import ChatAction
from typing import List

MLGWB_DOC = """
In MLG Water Bucket, an agent must perform an "MLG Water Bucket" jump onto a gold block. 
Then the agent mines the block to terminate the episode.
See the Custom Environments Tutorial for more information on this environment.
"""

MLGWB_LENGTH = 8000


class Navigation(Navigate):
    def __init__(
        self, dense, extreme, block_range=64, compass_radius=8, *args, **kwargs
    ):
        self.compass_radius = compass_radius
        self.block_range = block_range
        super().__init__(dense, extreme, *args, **kwargs)
        self.name = "navigation-v0"

    def create_observables(self) -> List[Handler]:
        return [
            handlers.POVObservation(self.resolution),
            handlers.CompassObservation(angle=True, distance=True),
            handlers.FlatInventoryObservation(["dirt"]),
            handlers.ObservationFromCurrentLocation(),
        ]

    def create_server_decorators(self) -> List[Handler]:
        return [
            handlers.NavigationDecorator(
                max_randomized_radius=self.block_range,
                min_randomized_radius=self.block_range,
                block="diamond_block",
                placement="surface",
                max_radius=self.compass_radius,
                min_radius=0,
                max_randomized_distance=self.compass_radius,
                min_randomized_distance=0,
                randomize_compass_location=True,
            )
        ]

    def create_actionables(self) -> List[Handler]:
        return super().create_actionables() + [
            # ChatAction()
        ]

    def is_from_folder(self, folder: str) -> bool:
        return folder == "stdN"

    def get_docstring(self):
        return MLGWB_DOC