
# TODO: review MinecraftNavTestEnv instead of MinecraftNavEnv
# reward is computed conditioned on the goal
from ..mineclip_official.normalize_image import torch_normalize
from .init_task import make
from collections import deque
import torch
import numpy as np
SUBGOAL_DISTANCE = 10  # subgoal distance
SUBGOAL_STEPS = 50  # steps to reach a subgoal


class MinecraftNavEnv:
    """
    Minecraft navigation environment for training find skill
    """

    def __init__(self, image_size=(160, 256), seed=0, biome='plains',
                 clip_model=None, device=None,  **kwargs):
        self.observation_size = (3, *image_size)
        self.action_size = 8
        self.biome = biome
        self.max_step = SUBGOAL_STEPS
        self.cur_step = 0
        self.seed = seed
        self.image_size = image_size
        self.kwargs = kwargs
        self.remake_env()
        # self._first_reset = True
        # self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]
        self.clip_model = clip_model  # use mineclip model to precompute embeddings
        self.device = device

    def __del__(self):
        if hasattr(self, 'base_env'):
            self.base_env.close()

    def set_goal(self, pos, g=None):
        """
        set goal position and direction

        pos: (x,y,z) current position
        g: (cos t, sin t) target yaw direction

        return: (cos t, sin t) target yaw direction
        """

        if g is None:
            # g = 2*np.pi*np.random.rand()
            g = 0.5*np.pi*np.random.randint(0, 4)  # simpler: 4 goals
            g = [np.cos(g), np.sin(g)]

        self.goal = np.array(g)
        self.init_pos = np.array([pos[0], pos[2]])
        # goal position (x',z')
        self.goal_pos = np.array(
            [pos[0]-SUBGOAL_DISTANCE*g[1], pos[2]+SUBGOAL_DISTANCE*g[0]])
        self.prev_distance = np.linalg.norm(self.init_pos-self.goal_pos)
        return g

    def add_goal_to_obs(self, obs):
        """
        add goal to observation

        obs: observation dict

        concatenate goal direction and goal position to obs['goal_emb']
        """
        yaw = np.deg2rad(obs["location_stats"]["yaw"])
        yaw = np.concatenate([np.cos(yaw), np.sin(yaw)])
        pos = obs['location_stats']['pos']
        pos = np.array([pos[0], pos[2]])  # [x,z]
        obs['goal_emb'] = np.concatenate([self.goal, yaw, self.goal_pos-pos])

    def remake_env(self):
        """
        call this to reset all the blocks and trees(logs)
        """

        if hasattr(self, 'base_env'):
            self.base_env.close()

        self.base_env = make(
            task_id="harvest",
            image_size=self.image_size,
            target_names='log',  # specify 'log'
            target_quantities=64,
            reward_weights=1,
            world_seed=self.seed,
            seed=self.seed,
            specified_biome=self.biome,
            use_voxel=True,
            voxel_size={'xmin': -1, 'ymin': -1, 'zmin': -
                        1, 'xmax': 1, 'ymax': 1, 'zmax': 1},
            **self.kwargs)
        # self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')

    def reset(self, reset_env=True, random_teleport=True):
        self.cur_step = 0
        # reset after finishing subgoal
        if not reset_env:
            return

        # big reset
        # if not self._first_reset:
            # for cmd in self._reset_cmds:
            #    self.base_env.unwrapped.execute_cmd(cmd)
        # self._first_reset = False
        self.prev_action = self.base_env.action_space.no_op()
        obs = self.base_env.reset()
        self.base_env.unwrapped.set_time(6000)
        self.base_env.unwrapped.set_weather("clear")
        # print(obs['location_stats']['pos'])

        # random teleport agent
        if random_teleport:
            self.base_env.random_teleport(200)
            self.base_env.step(self.base_env.action_space.no_op())
            obs, _, _, _ = self.base_env.step(
                self.base_env.action_space.no_op())
            # I find that position in obs is updated after 2 env.step
            # print(obs['location_stats']['pos'])

        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(
                    1, 1, *self.observation_size)
                img_emb = self.clip_model.image_encoder(
                    torch.as_tensor(img, dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy()  # (1,1,512)
                # print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        return obs

    def step(self, act):
        obs, _, done, info = self.base_env.step(act)
        # agent_dead = done
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            done = True

        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(
                    1, 1, *self.observation_size)
                img_emb = self.clip_model.image_encoder(
                    torch.as_tensor(img, dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy()  # (1,1,512)
                # print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        self.prev_action = act  # save the previous action for the agent's observation

        # compute navigation reward
        yaw = np.deg2rad(obs['location_stats']['yaw'][0])
        reward_yaw = np.cos(
            yaw) * self.goal[0] + np.sin(yaw) * self.goal[1]  # [-1,1]
        pitch = np.deg2rad(obs['location_stats']['pitch'][0])
        reward_pitch = np.cos(pitch)  # [0,1]
        pos = obs['location_stats']['pos']
        pos = np.array([pos[0], pos[2]])  # [x,z]
        dis = np.linalg.norm(pos-self.goal_pos)  # generally [-10,0]
        reward_dis = self.prev_distance - dis  # [-0.2, 0.2]
        reward = reward_yaw + reward_pitch + reward_dis*10  # [-3,4]
        self.prev_distance = dis
        # reward = reward_dis
        obs['reward_yaw'] = reward_yaw
        obs['reward_dis'] = reward_dis
        obs['reward_pitch'] = reward_pitch

        # info['agent_dead'] = agent_dead
        return obs, reward, done, info


class MinecraftNavTestEnv(MinecraftNavEnv):
    """
    Minecraft navigation environment for testing find skill

    Compared to MinecraftNavEnv, this class adds lidar to detect targets
    """

    def __init__(self, image_size=(160, 256), seed=0, biome='plains',
                 clip_model=None, device=None, **kwargs):
        super().__init__(image_size=image_size, seed=seed, biome=biome,
                         clip_model=clip_model, device=device,  **kwargs)

    def remake_env(self):
        """
        call this to reset all the blocks and trees
        """
        if hasattr(self, 'base_env'):
            self.base_env.close()

        self.base_env = make(
            task_id="harvest",
            image_size=self.image_size,
            target_names='log',
            target_quantities=64,
            reward_weights=1,
            world_seed=self.seed,
            seed=self.seed,
            specified_biome=self.biome,
            use_voxel=True,
            voxel_size={'xmin': -1, 'ymin': -1, 'zmin': -
                        1, 'xmax': 1, 'ymax': 1, 'zmax': 1},
            # Add lidar
            use_lidar=True,
            lidar_rays=[
                (np.pi * pitch / 180, np.pi * yaw / 180, 99)
                for pitch in np.arange(-30, 30, 5)
                for yaw in np.arange(-45, 45, 5)
            ],
            # spawn initial mobs
            initial_mobs=['cow']*3 + ['sheep']*3,
            initial_mob_spawn_range_low=(-30, 1, -30),
            initial_mob_spawn_range_high=(30, 1, 30),
            # teleport agent when reset
            fast_reset=True,
            fast_reset_random_teleport_range_low=0,
            fast_reset_random_teleport_range_high=200,
            **self.kwargs)
        # self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')

    def reset(self, reset_env=True):
        self.cur_step = 0
        # reset after finishing subgoal
        if not reset_env:
            return
        # big reset
        self.prev_action = self.base_env.action_space.no_op()
        # reset after random teleport, spawn mobs nearby
        self.base_env.reset(move_flag=True)
        self.base_env.unwrapped.set_time(6000)
        self.base_env.unwrapped.set_weather("clear")
        # make agent fall onto the ground after teleport
        for i in range(4):
            obs, _, _, _ = self.base_env.step(
                self.base_env.action_space.no_op())
        self.total_steps = 0

        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(
                    1, 1, *self.observation_size)
                img_emb = self.clip_model.image_encoder(
                    torch.as_tensor(img, dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy()  # (1,1,512)
                # print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action
        return obs

    def step(self, act):
        self.total_steps += 1
        return super().step(act)

    # detect target items
    def target_in_sight(self, obs, target, max_dis=20):
        if target in ['wood']:
            target_type = 'block'
        elif target in ['cow', 'sheep']:
            target_type = 'entity'
        else:
            raise NotImplementedError

        print("!!!!!", np.rad2deg(
            obs['rays']['ray_yaw']), obs['rays'][target_type+'_name'])
        names, distances = obs['rays'][target_type +
                                       '_name'], obs['rays'][target_type+'_distance']
        idxs = np.where(names == target)[0]
        if len(idxs) == 0:
            return False, None
        idx = idxs[np.argmin(distances[idxs])]
        dis = distances[idx]
        if dis > max_dis:
            return False, None
        # minedojo bug! yaw in lidar is opposite.
        yaw_relative = -np.rad2deg(obs['rays']['ray_yaw'][idx])
        yaw = obs["location_stats"]["yaw"][0] + yaw_relative
        pos = obs['location_stats']['pos']
        dr = [np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))]
        target_pos = np.array([pos[0]-dis*dr[1], pos[2]+dis*dr[0]])
        return True, {'dis': dis, 'yaw': yaw, 'yaw_relative': yaw_relative, 'target_pos': target_pos}


if __name__ == '__main__':
    # print(minedojo.ALL_TASKS_SPECS)
    env = MinecraftNavTestEnv()
    # reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]
    obs = env.reset()
    # print(obs.shape, obs.dtype)
    for t in range(1000):
        act = env.base_env.action_space.no_op()
        act[4] = 13
        next_obs, r, done, info = env.step(act)
