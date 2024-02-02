import pickle
import torch
import numpy as np
from recurrent_ppo_truncated_bptt.model import ActorCriticModel
from recurrent_ppo_truncated_bptt.environments.navigation_env import MinecraftNav


class SkillFind:
    def __init__(self, device=torch.device('cuda:0')):
        self.device = device

    def execute(self, target, model_path, max_steps_high, max_steps_low, env, **kwargs):
        """
        Complete the task of finding the target object in the environment.
        """
        # replace the target name with the corresponding block name
        if target == 'log':
            target = 'wood'
        elif target == 'cobblestone':
            target = 'stone'

        env_high = MinecraftNav(max_steps=max_steps_high, usage='deploy', env=env, low_level_policy_type='dqn',
                                device=self.device)
        # load pre-traied model parameters
        state_dict, config = pickle.load(open(model_path, "rb"))
        model = ActorCriticModel(
            config, env_high.observation_space, (env_high.action_space.n,))
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        # Init recurrent cell
        hxs, cxs = model.init_recurrent_cell_states(1, self.device)
        # If the recurrent layer is GRU, the recurrent cell is hxs; if the recurrent layer is LSTM, the recurrent cell is (hxs, cxs)
        if config["recurrence"]["layer_type"] == "gru":
            recurrent_cell = hxs
        elif config["recurrence"]["layer_type"] == "lstm":
            recurrent_cell = (hxs, cxs)

        obs = env_high.reset()
        done = False
        while not done:
            with torch.no_grad():
                policy, value, recurrent_cell = model(torch.tensor(
                    np.expand_dims(obs, 0)).float(), recurrent_cell, self.device, 1)
            # from policy distribution sample an action
            action = policy.sample().cpu().numpy()
            obs, reward, done, info = env_high.step(
                int(action), target=target, max_steps_low=max_steps_low)

        if info['task_done'] or (not ('dis' in info)):
            return False, info['task_success'], info['task_done']

        # 方法通过循环执行环境的低层策略，不断尝试达到目标。循环中会根据目标的相对位置更新环境中的目标位置，判断是否成功找到目标
        success, r, done = env_high.reach(target, info)
        return success, r, done
