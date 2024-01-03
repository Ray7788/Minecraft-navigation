import argparse
import random
from .utils import yaml_setting
from .mineclip_official.build import build_pretrain_model
from minedojo_official import InventoryItem
import gym
import minerl
import torch as tr
import pip
import numpy as np
import os
import torch

from agents.baseAgent import RandomAgent
from agents.compassBasedAgent import CompassBasedAgent
from Utils.agentPOV_visualiser import (
    POV_visualiser_args,
    visualise_agent,
    visualise_data,
    visualise_from_file,
)
from envs.mlg_wb_specs import MLGWB
from envs.envManager import EnvManager
from agents.agentManager import AgentManager

from utils.arg_parser import *
from Utils.plotting import draw_progress_for_all
from minedojo.sim import InventoryItem
# fix deprecation error in numpy
np.float = float
np.int = int
  
for agent_name in agents:
    # set the seed 
    if args.action == "testing":

            # obs = env.reset()

            # print(obs)
            # print(env.action_space)
            reward_histories = visualise_agent(
                agent,
                env,
                args.output_path,
                args.num_episodes,
                args.max_steps,
                verbose=args.verbose,
                radius=args.radius,
                colour=AgentManager.colour_map[agent_name],
                world_seeds=args.seeds,
                ep_per_seed=True,
            )

plot_path = os.path.join(
    args.output_path,
    f"[{'_'.join(agents)}]_{args.env_name}_{args.num_episodes}_progress.png",
)
draw_progress_for_all(agent_progress_map, plot_path)

# ---------------------------------------------------------------------------------------------
def main(args):
    # save path(parenmt dir)
    save_dir = args.save_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # create sub-directory for specific task
    save_dir = os.path.join(save_dir, args.task)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: ', device)
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # seed control
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # register env
    agents = ArgParser.split_agent(args.agent_class)
    EnvManager.register(dense=args.dense, extreme=args.extreme, radius=args.radius)

    AgentManager.set_up()
    # agent_progress_map = {}
    env = EnvManager.create_env_by_name(args.env_name)

    # load clip model
    clip_config = yaml_setting.get_yaml_data(args.clip_config_path)
    model_clip = build_pretrain_model(
        image_config=clip_config['image_config'],
        text_config=clip_config['text_config'],
        temporal_config=clip_config['temporal_config'],
        adapter_config=clip_config['adaptor_config'],
        state_dict=torch.load(args.clip_model_path)
    ).to(device)
    model_clip.eval()
    print('MineCLIP model is loaded from:', args.clip_model_path)

    # -----------------load task configs-----------------
    # load task configs
    task_conf = yaml_setting.get_yaml_data(args.task_config_path)[args.task]
    #print(task_conf)
    init_items = {}
    if 'initial_inventory' in task_conf:
        init_items = task_conf['initial_inventory']
        # TODO replace minedojo with minerl
        init_inv = [InventoryItem(slot=i, name=k, variant=None, quantity=task_conf['initial_inventory'][k]) 
        for i,k in enumerate(list(task_conf['initial_inventory'].keys()))]
        task_conf['initial_inventory'] = init_inv
    #print(init_inv)
        
    if args.shorter_episode:
        task_conf['max_steps'] = task_conf['max_steps']//2
       
    # Instantiate environment-----------------
    env = MinecraftHardHarvestEnv(
        image_size=(160,256),
        seed=seed,
        clip_model=model_clip,
        device=device,
        save_rgb=args.save_gif,
        **task_conf
        )
    
    # load skills
    skills_model = SkillsModel(device=device, path=args.skills_model_config_path)



if __name__ == "__main__":
    args = ArgParser.create_parser()
    print(args)
    main(args)