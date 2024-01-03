import argparse
from envs import envs_manager

class ArgParser:
    @staticmethod
    def create_parser() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="parses the arguments")
        # minedojo
        parser.add_argument(
            '--task',
            type=str,
            help="Specify the task to run",
            default='harvest_milk_with_crafting_table_and_iron_ingot'
        )

        # mineRL envs
        parser.add_argument(
            "--env_name",
            type=str,
            help="the class of environment to run in the visualisation.",
            choices=tuple(envs_manager.EnvsManager.envs.keys()),
            default="Navigation",
        )

        parser.add_argument(
            "-e",
            "--extreme",
            help="Use extrame environment for Navigation task",
            action="store_true",
            default=False,
        )

        parser.add_argument(
            "-d",
            "--dense",
            help="Use dense environment for Navigation task",
            action="store_true",
            default=False,
        )

    # path for pretrained model and config
        parser.add_argument(
            "--input_file",
            help="The location of the path file",
            default="src/paths/path0.txt",
        )

        parser.add_argument(
            "--agent_path",
            help="The path to the parameters to load into the agents if needed.",
            default=None,
        )

        parser.add_argument(
            "-mp",
            "--model_path",
            help="The model of the agent",
            type=str,
            default="./src/models/best_model_r1_0.zip",
        )

        parser.add_argument('--clip-config-path', type=str,
                            default='mineclip_official/config.yml')
        parser.add_argument('--clip-model-path', type=str,
                            default='mineclip_official/attn.pth')
        parser.add_argument(
            '--task_config_path',
            type=str,
            default='envs/hard_task_conf.yaml'
        )

        parser.add_argument('--skills-model-config-path',
                            type=str, default='skills/load_skills.yaml')

    # seed for test episodes, random seed for both np, torch and env
        parser.add_argument(
            "--seed",
            type=int,
            help="The seed value for the random number generation.",
            default=7
        )
        parser.add_argument(
            "--num_episodes",
            help="The number of episodes to visualise and save",
            type=int,
            default=30,
        )

    # save results
        parser.add_argument(
            "--save_path",
            help="Specify the outputs path to save the visualisations to.",
            type=str,
            default="./results/"
        )

        parser.add_argument(
            '--save_gif',
            help="choose 1 if save whole gifs",
            type=int,
            default=1
        )

        parser.add_argument(
            "-ms",
            "--max_steps",
            help="Specifies the maximum number of steps in each episode",
            type=int,
            default=int(1e5),
        )

        parser.add_argument(
            "--n_envs",
            help="Specifies the number of envs",
            type=int, default=1
        )

        parser.add_argument(
            "-v",
            "--verbose",
            help="Specify how much debug information should be shown (0-2)",
            action="count",
            default=0,
        )

        parser.add_argument(
            "--radius",
            help="Specify how far the radius is from the target to the compass",
            type=int,
            default=8,
        )

        parser.add_argument(
            "-t",
            "--transfer",
            help="load a existing model to train again",
            action="store_true",
            default=False,
        )

        parser.add_argument(
            "--distance",
            help="The distance from the diamond block",
            type=int,
            default=64,
        )

        parser.add_argument(
            "-rt",
            "--reward_type",
            help="The reward type of the env",
            type=int,
            default=1,
        )

        parser.add_argument(
            "--seeds",
            help="Specify the seeds for generated worlds",
            default=[0],
            nargs="*",
        )

        parser.add_argument(
            "--reset_threshold",
            help="reset threshold",
            type=float,
            default=2.,
        )

        # ----------------------------------
        # Compare this snippet from project/src/main.py:
        # set to 0 for zero-shot planning
        parser.add_argument('--progressive_search', type=int, default=1)
        # ablation for using 1/2 episode steps?
        parser.add_argument('--shorter_episode', type=int, default=0)
        # ablation without find-skill?
        parser.add_argument('--no-find-skill', type=int, default=0)

        # Parse the arguments
        args: argparse.Namespace = parser.parse_args()

        return args

    @staticmethod
    def split_agent(agents: str):
        agents = agents.strip().split(",")
        agents = [a for a in agents if a in AgentManager.agents]
        if len(agents) == 0:
            print("There is no valid agent")
            raise ValueError("There is no valid agent!")
        return agents
