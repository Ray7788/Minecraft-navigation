from utils import get_yaml_data
from .skill_manipulate import SkillManipulate
from .skill_craft import SkillCraft
from .skill_find import SkillFind


class SkillsModel:
    def __init__(self, device, path='skills/load_skills.yaml'):
        self.device = device
        self.skill_info = get_yaml_data(path)
        self.skill_models = [
            SkillFind(device=device),
            SkillManipulate(device=device),
            SkillCraft()
        ]
        # print(self.skill_info)

    def execute(self, skill_name, skill_info, env):
        """
        General skill execution function.
        According to skill_name and skill_info, allocate the corresponding skill model to execute the skill.

        Args:
            skill_name: the name of the skill
            skill_info: the information of the skill from yaml file
            env: the environment
        """
        skill_type = skill_info['skill_type']
        equip = skill_info['equip']
        inventory = env.obs['inventory']['name']
        # equip tools from the player's inventory
        for e in equip:
            idx = inventory.tolist().index(e.replace('_', ' '))
            act = env.base_env.action_space.no_op()
            act[5] = 5
            act[7] = idx
            obs, r, done, _ = env.step(act)  # obs, reward, done, info
            if done:
                # skill done, task success, task done
                return False, bool(r), done

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

        # execute detailed skills based on skill_type
        if skill_type == 0:   # SkillFind
            # check if the skill name ends with "_nearby"
            assert skill_name.endswith('_nearby')
            # Remove "_nearby", and **Unpack the key-value pairs in the dictionary and pass them to the function
            return self.skill_models[0].execute(target=skill_name[:-7], env=env, **self.skill_info['find'])
        elif skill_type == 1:  # SkillManipulate
            if not (skill_name in self.skill_info):
                print('Warning: skill {} is not in load_skills.yaml'.format(skill_name))
                return False, False, False
            return self.skill_models[1].execute(target=skill_name, env=env, equip_list=equip, **self.skill_info[skill_name])
        elif skill_type == 2:  # SkillCraft(less frequently used)
            return self.skill_models[2].execute(target=skill_name, env=env)
        else:
            raise Exception('Illegal skill_type.')
