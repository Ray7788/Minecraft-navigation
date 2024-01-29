import utils
from .skill_manipulate import SkillManipulate
from .skill_craft import SkillCraft
from .skill_find import SkillFind

class SkillsModel:
    def __init__(self, device, path='skills/load_skills.yaml'):
        self.device = device
        self.skill_info = utils.get_yaml_data(path)
        self.skill_models = [
            # 预存的model
            SkillFind(device=device),   # 0
            SkillManipulate(device=device), # 1
            SkillCraft()
        ]
        #print(self.skill_info)

    def execute(self, skill_name, skill_info, env):
        """
        主要执行函数，分发并返回情况
        """
        # skill_info: skills.
        skill_type=skill_info['skill_type']
        equip=skill_info['equip']
        inventory = env.obs['inventory']['name']
        # equip tools from the player's inventory
        for e in equip:
            idx = inventory.tolist().index(e.replace('_',' '))
            act = env.base_env.action_space.no_op()
            # print("1nd", act)
            act[5] = 5
            act[7] = idx
            # print("2nd", act)
            # [ 0  0  0 12 12  5  0  0]
            obs, r, done, _ = env.step(act) # obs, reward, done, info
            if done:
                return False, bool(r), done # skill done, task success, task done
        '''
        # for manipulation skills with nothing to equip
        if len(equip)==0 and skill_type==1:
            idx = inventory.tolist().index('air')
            act = env.base_env.action_space.no_op()
            act[5] = 5
            act[7] = idx
            obs, r, done, _ = env.step(act)
            if done:
                return False, bool(r), done
        '''

        # execute skill
        if skill_type==0:   # SkillFind
            assert skill_name.endswith('_nearby')   # double check task name 以_nearby结尾
            return self.skill_models[0].execute(target=skill_name[:-7], env=env, **self.skill_info['find']) # 去除"_nearby", **将字典中的键值对解包为关键字参数传递给函数
        elif skill_type==1: # SkillManipulate
            if not (skill_name in self.skill_info):
                print('Warning: skill {} is not in load_skills.yaml'.format(skill_name))
                return False, False, False
            return self.skill_models[1].execute(target=skill_name, env=env, equip_list=equip, **self.skill_info[skill_name])
        elif skill_type==2: # SkillCraft
            return self.skill_models[2].execute(target=skill_name, env=env)
        else:
            raise Exception('Illegal skill_type.')