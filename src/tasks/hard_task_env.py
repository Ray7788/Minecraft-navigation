# environment for hard harvest tasks, support 3 types of skills
import numpy as np
import torch
from mineclip_official import torch_normalize
from init_task import make
SUBGOAL_DISTANCE = 10


class MinecraftHardHarvestEnv:
    def __init__(self, image_size=(160, 256), seed=0, biome='plains', clip_model=None, device=None, save_rgb=False,
                 target_name='log', target_quantity=1,  max_steps=3000, **kwargs):
        self.observation_size = (3, *image_size)
        self.action_size = 8
        self.max_step = max_steps
        self.cur_step = 0
        self.image_size = image_size
        self.kwargs = kwargs  # kwargs should contain: initial inventory, initial mobs
        self.target_name = target_name
        self.target_quantity = target_quantity
        self.remake_env()
        # self._first_reset = True
        # self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]
        self.seed = seed
        self.biome = biome
        self.clip_model = clip_model  # use mineclip model to precompute embeddings
        self.device = device
        self.save_rgb = save_rgb

    def __del__(self):
        if hasattr(self, 'base_env'):
            self.base_env.close()

    def remake_env(self):
        if hasattr(self, 'base_env'):
            self.base_env.close()

        self.target_item_name = self.target_name[:-7] if self.target_name.endswith(
            '_nearby') else self.target_name

        self.base_env = make(
            task_id="harvest",
            image_size=self.image_size,
            target_names=self.target_item_name,
            target_quantities=self.target_quantity,
            reward_weights=1,
            world_seed=self.seed,
            seed=self.seed,
            specified_biome=self.biome,
            use_voxel=True,
            voxel_size={'xmin': -1, 'ymin': -1, 'zmin': -
                        1, 'xmax': 1, 'ymax': 1, 'zmax': 1},
            use_lidar=True,
            lidar_rays=[
                (np.pi * pitch / 180, np.pi * yaw / 180, 99)
                for pitch in np.arange(-30, 30, 5)
                for yaw in np.arange(-45, 45, 5)
            ],
            # teleport agent to a new place when reset
            fast_reset=True,
            fast_reset_random_teleport_range_low=0,
            fast_reset_random_teleport_range_high=500,
            # spawn initial mobs
            # initial_mobs=['cow']*3 + ['sheep']*3,
            # initial_mob_spawn_range_low=(-30, 1, -30),
            # initial_mob_spawn_range_high=(30, 1, 30),
            **self.kwargs)
        # self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')

    def reset(self):
        """
        Reset the environment. This function is called when a new episode starts.
        """
        self.cur_step = 0
        self.prev_action = self.base_env.action_space.no_op()
        self.base_env.reset(move_flag=True)
        # Reset the environment: time, weather
        self.base_env.unwrapped.set_time(6000)
        self.base_env.unwrapped.set_weather("clear")
        # make agent fall onto the ground after teleport
        for i in range(4):
            obs, _, _, _ = self.base_env.step(
                self.base_env.action_space.no_op())

        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(
                    1, 1, *self.observation_size)
                img_emb = self.clip_model.image_encoder(
                    torch.as_tensor(img, dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy()  # (1,1,512)
                # print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        if self.save_rgb:
            self.rgb_list = [np.transpose(
                obs['rgb'], [1, 2, 0]).astype(np.uint8)]
            self.action_list = []

        self.obs = obs
        self.last_obs = obs
        return obs

    def step(self, act):
        obs, reward, done, info = self.base_env.step(act)
        if self.target_name.endswith('_nearby'):
            reward = self.reward_harvest(obs, self.target_name)
            done = True if reward > 0 else False

        if obs['life_stats']['life'] == 0:
            done = True
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            done = True
        if reward > 0:
            reward = 1
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
        self.last_obs = self.obs
        self.obs = obs
        if self.save_rgb:
            self.rgb_list.append(np.transpose(
                obs['rgb'], [1, 2, 0]).astype(np.uint8))
            self.action_list.append(np.asarray(act))
        return obs, reward, done, info

    # for Find skill: detect target items
    def target_in_sight(self, obs, target, max_dis=20):
        if target in ['wood', 'stone']:
            target_type = 'block'
        elif target in ['cow', 'sheep']:
            target_type = 'entity'
        else:
            raise NotImplementedError

        # print(np.rad2deg(obs['rays']['ray_yaw']), obs['rays'][target_type+'_name'])
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

    '''
    for goal-based find skill
    pos: (x,y,z) current position
    g: (cos t, sin t) target yaw direction
    '''

    def set_goal(self, pos, g=None):
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
        yaw = np.deg2rad(obs["location_stats"]["yaw"])
        yaw = np.concatenate([np.cos(yaw), np.sin(yaw)])
        pos = obs['location_stats']['pos']
        pos = np.array([pos[0], pos[2]])  # [x,z]
        obs['goal_emb'] = np.concatenate([self.goal, yaw, self.goal_pos-pos])

    # compute harvest reward under different cases

    def reward_harvest(self, obs, target_name, target_quantity=1, incremental=True):
        # target nearby
        if target_name.endswith('_nearby'):
            target_item_name = target_name[:-7]
            if target_item_name == 'furnace':
                return int(obs['nearby_tools']['furnace'])
            elif target_item_name == 'crafting_table':
                return int(obs['nearby_tools']['table'])
            else:
                if target_item_name == 'log':
                    target_item_name = 'wood'
                elif target_item_name == 'cobblestone':
                    target_item_name = 'stone'
                find, info = self.target_in_sight(obs, target_item_name)
                if find and info['dis'] <= 3:
                    return 1
                else:
                    return 0
        # target in inventory
        else:
            names, nums = obs['inventory']['name'], obs['inventory']['quantity']
            # print('inventory:',names)
            idxs = np.where(names == target_name.replace('_', ' '))[0]
            if len(idxs) == 0:
                return 0
            else:
                num_cur = np.sum(nums[idxs])
                num_last = 0
                if incremental:
                    names, nums = self.last_obs['inventory']['name'], self.last_obs['inventory']['quantity']
                    idxs = np.where(names == target_name.replace('_', ' '))[0]
                    if len(idxs) > 0:
                        num_last = np.sum(nums[idxs])
                if num_cur-num_last >= target_quantity:
                    return 1
                else:
                    return 0
