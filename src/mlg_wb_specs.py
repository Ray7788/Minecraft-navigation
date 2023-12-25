"""MLG Water Bucket Gym"""
__ref__ = "form https://github.com/minerllabs/minerl/blob/v0.4.4/examples/mlg_wb_specs.py"
__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

# specialised for navigation
from minerl.herobraine.env_specs.navigate_specs import Navigate
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing import List
# Add chat module
from minerl.herobraine.hero.handlers.agent.actions.chat import ChatAction


MLGWB_DOC = """
In MLG Water Bucket, an agent must perform an "MLG Water Bucket" jump onto a gold block. 
Then the agent mines the block to terminate the episode.
See the Custom Environments Tutorial for more information on this environment.
"""

# Specifies how many time steps the environment can last until termination.
MLGWB_LENGTH = 8000


class MLGWB(Navigate):
    def __init__(
        self, dense, extreme, block_range=64, compass_radius=8, *args, **kwargs
    ):
        self.compass_radius = compass_radius
        self.block_range = block_range
        super().__init__(dense, extreme, *args, **kwargs)
        self.name = "MLGWB-v0"

    def create_server_world_generators(self) -> List[Handler]:
        """
        Create the server world generators.
        Seed for Minecraft 1.4 :
        https://minecraft.tools/en/flat.php?biome=1&bloc_1_nb=1&bloc_1_id=2&bloc_2_nb=2&bloc_2_id=3%2F00&bloc_3_nb=1&bloc_3_id=7&village_size=1&village_distance=32&mineshaft_chance=1&stronghold_count=3&stronghold_distance=32&stronghold_spread=3&oceanmonument_spacing=32&oceanmonument_separation=5&biome_1_distance=32&valid=Create+the+Preset#seed
        """

        # 1 layer of grass blocks above 2 layers of dirt above 1 layer of bedrock
        return [handlers.FlatWorldGenerator(generatorString="1;7,2x3,2;1")]

    # TODO: Add agent start inventory based on the task
    # def create_agent_start(self) -> List[Handler]:
    #     return [
    #         # make the agent start with these items
    #         handlers.SimpleInventoryAgentStart([
    #             dict(type="water_bucket", quantity=1),
    #             dict(type="diamond_pickaxe", quantity=1)
    #         ]),
    #         # make the agent start 90 blocks high in the air
    #         handlers.AgentStartPlacement(0, 90, 0, 0, 0)
    #     ]

    # TODO: Add rewardables based on the task: agent receives reward for getting to a gold block
    # def create_rewardables(self) -> List[Handler]:
    #     return [
    #         # reward the agent for touching a gold block (but only once)
    #         handlers.RewardForTouchingBlockType([
    #             {'type': 'gold_block', 'behaviour': 'onceOnly', 'reward': '50'},
    #         ]),
    #         # also reward on mission end
    #         handlers.RewardForMissionEnd(50)
    #     ]

# TODO: Add agent handlers based on the task:  terminate when the agent obtains a gold block
    # def create_agent_handlers(self) -> List[Handler]:
    #     return [
    #         # make the agent quit when it gets a gold block in its inventory
    #         handlers.AgentQuitFromPossessingItem([
    #             dict(type="gold_block", amount=1)
    #         ])
    #     ]

# TODO keep the actions which SimpleEmbodimentEnvSpec does provide by default (like movement, jumping)
    # def create_actionables(self) -> List[Handler]:----------------------
    #     return super().create_actionables() + [
    #         # allow agent to place water
    #         handlers.KeybasedCommandAction("use"),
    #         # also allow it to equip the pickaxe
    #         handlers.EquipAction(["diamond_pickaxe"])
    #     ]

    def create_observables(self) -> List[Handler]:
        return [
            handlers.POVObservation(self.resolution),
            handlers.CompassObservation(angle=True, distance=True),
            handlers.FlatInventoryObservation(["dirt"]),
            handlers.ObservationFromCurrentLocation(),
        ]

    # def create_server_initial_conditions(self) -> List[Handler]:
    #     return [
    #         # Sets time to morning and stops passing of time
    #         handlers.TimeInitialCondition(False, 23000)
    #     ]

    # def create_server_quit_producers(self):
    #     return []

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
        return folder == "mlgwb"

    def get_docstring(self):
        return MLGWB_DOC
