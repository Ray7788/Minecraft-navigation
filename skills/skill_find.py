import pickle
import copy
import torch
import numpy as np
from docopt import docopt
from recurrent_ppo_truncated_bptt.model import ActorCriticModel
from recurrent_ppo_truncated_bptt.environments.navigation_env import MinecraftNav


class SkillFind:
    # TODO if there are more than 1 GPU, how to do it?
    def __init__(self, device=torch.device('cuda:0')):
        self.device = device

    def execute(self, target, model_path, max_steps_high, max_steps_low, env, **kwargs):
        # 根据传入的目标名称（target）进行处理，将"log"映射为"wood"，将"cobblestone"映射为"stone"。
        if target=='log':
            target = 'wood'
        elif target=='cobblestone':
            target = 'stone'
            
        env_high = MinecraftNav(max_steps=max_steps_high, usage='deploy', env=env, low_level_policy_type='dqn',
            device=self.device)
        state_dict, config = pickle.load(open(model_path, "rb"))
        model = ActorCriticModel(config, env_high.observation_space, (env_high.action_space.n,))
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        # Init recurrent cell
        hxs, cxs = model.init_recurrent_cell_states(1, self.device)
        if config["recurrence"]["layer_type"] == "gru":
            recurrent_cell = hxs
        elif config["recurrence"]["layer_type"] == "lstm":
            recurrent_cell = (hxs, cxs)

        obs = env_high.reset()
        done = False
        while not done:
            with torch.no_grad():   # 禁用梯度计算，因为在执行时无需梯度
                policy, value, recurrent_cell = model(torch.tensor(np.expand_dims(obs, 0)).float(), recurrent_cell, self.device, 1)
            action = policy.sample().cpu().numpy()  # 从策略分布中采样得到动作
            obs, reward, done, info = env_high.step(int(action), target=target, max_steps_low=max_steps_low)
        
        # 如果任务完成或者信息中没有包含距离信息，则返回一个包含任务成功标志、任务成功奖励和任务完成标志的元组。
        if info['task_done'] or (not ('dis' in info)):
            return False, info['task_success'], info['task_done']

        # 方法通过循环执行环境的低层策略，不断尝试达到目标。循环中会根据目标的相对位置更新环境中的目标位置，判断是否成功找到目标
        success, r, done = env_high.reach(target, info)
        return success, r, done
