from utils import *
from .mineclip_official.build import build_pretrain_model
from tasks.sim.inventory import InventoryItem
from .skills.load_skills import SkillsModel
from .tasks.hard_task_env import MinecraftHardHarvestEnv
from .skills import skills, skill_search, SkillsModel, convert_state_to_init_items
import random
import imageio
import numpy as np
import os
import torch


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

    # load clip model
    clip_config = get_yaml_data(args.clip_config_path)
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
    task_conf = get_yaml_data(args.task_config_path)[args.task]
    # print(task_conf)
    init_items = {}
    if 'initial_inventory' in task_conf:
        init_items = task_conf['initial_inventory']
        # TODO replace minedojo with minerl
        init_inv = [InventoryItem(slot=i, name=k, variant=None, quantity=task_conf['initial_inventory'][k])
                    for i, k in enumerate(list(task_conf['initial_inventory'].keys()))]
        task_conf['initial_inventory'] = init_inv
    # print(init_inv)

    if args.shorter_episode:
        task_conf['max_steps'] = task_conf['max_steps']//2

    # Instantiate environment-----------------
    env = MinecraftHardHarvestEnv(
        image_size=(160, 256),
        seed=seed,
        clip_model=model_clip,
        device=device,
        save_rgb=args.save_gif,
        **task_conf
    )

    # load skills
    skills_model = SkillsModel(
        device=device, path=args.skills_model_config_path)

    # Decompose task into skills-----------------
    target_name = task_conf['target_name']
    # TODO: maybe rewrite search algorithm
    skill_sequence, init_items_miss = skill_search(target_name, init_items)
    if len(init_items_miss) > 0:
        raise Exception(
            'Cannot finish task because of missing initial items: {}'.format(init_items_miss))
    print('Task {} is decomposed into skill sequence: {}'.format(
        args.task, skill_sequence))

    #  Show the skill sequence situation-----------------
    skill_success_cnt = np.zeros(len(skill_sequence))
    print('Initial skill sequence: {}, length: {}'.format(
        skill_sequence, len(skill_sequence)))
    skill_sequence_unique = list(set(skill_sequence))
    skill_sequence_unique.sort(key=skill_sequence.index)    # sequential 排序
    skill_success_cnt_unique = np.zeros(len(skill_sequence_unique))
    print('Unique skill list: {}, length: {}'.format(
        skill_sequence_unique, len(skill_sequence_unique)))
    test_success_rate = 0

    for ep in range(args.num_episodes):
        env.reset()
        episode_snapshots = [('begin', np.transpose(
            env.obs['rgb'], [1, 2, 0]).astype(np.uint8))]  # beginning status

        if not args.progressive_search:
            assert args.no_find_skill == 0    # TODO 搞清楚意思
            assert args.shorter_episode == 0
            episode_skill_success_unique = np.zeros(len(skill_sequence_unique))
            for i_sk, sk in enumerate(skill_sequence):
                print('executing skill:', sk)
                # 测试的skill type是1
                skill_done, task_success, task_done = skills_model.execute(
                    skill_name=sk, skill_info=skills[sk], env=env)  # choose task type and task name from skills.yaml
                if skill_done or task_success:
                    skill_success_cnt[i_sk] += 1
                    episode_skill_success_unique[skill_sequence_unique.index(
                        sk)] = 1
                    episode_snapshots.append(
                        (sk, np.transpose(env.obs['rgb'], [1, 2, 0]).astype(np.uint8)))
                if (not skill_done) or task_done:
                    break
            # print(skill_success_cnt)
            print('skill done {}, task success {}, task done {}'.format(
                skill_done, task_success, task_done))
            skill_success_cnt_unique += episode_skill_success_unique
        # update the future skill sequence after each skill. 每次都更新
        else:
            # create a new skill sequence
            episode_skill_success = np.zeros(len(skill_sequence))
            episode_skill_success_unique = np.zeros(len(skill_sequence_unique))
            episode_skill_idx = 0

            skill_next = skill_sequence[0]  # 下一个任务
            # ablation: skip find skills    跳过find（一般不进入该循环）
            if args.no_find_skill and skills[skill_next]['skill_type'] == 0:
                skill_next = skill_sequence[1]  # 跳过当前
                assert skills[skill_next]['skill_type'] != 0  # 再次检查
            init_items_next = init_items
            while True:
                print('executing skill:', skill_next)
                skill_done, task_success, task_done = skills_model.execute(
                    skill_name=skill_next, skill_info=skills[skill_next], env=env)

                if skill_done or task_success:
                    if skill_next in skill_sequence[episode_skill_idx:]:
                        episode_skill_idx += skill_sequence[episode_skill_idx:].index(
                            skill_next)
                        episode_skill_success[episode_skill_idx] = 1
                        episode_skill_idx += 1
                    if skill_next in skill_sequence_unique:
                        episode_skill_success_unique[skill_sequence_unique.index(
                            skill_next)] = 1
                    episode_snapshots.append((skill_next, np.transpose(
                        env.obs['rgb'], [1, 2, 0]).astype(np.uint8)))
                if task_done:
                    break

                init_items_next = convert_state_to_init_items(init_items_next, skill_next, skills[skill_next]['skill_type'],
                                                              skill_done, env.obs['inventory']['name'], env.obs['inventory']['quantity'])
                skill_sequence_next, items_miss = skill_search(
                    target_name, init_items_next)
                skill_next = skill_sequence_next[0]
                print('recomputed skill sequence:', skill_sequence_next)
                # ablation: skip find skills
                if args.no_find_skill and skills[skill_next]['skill_type'] == 0:
                    skill_next = skill_sequence_next[1]
                    assert skills[skill_next]['skill_type'] != 0
                if len(items_miss) > 0:
                    print('cannot execute some skills:', items_miss)
                    break
            print('task done {}'.format(task_done))
            skill_success_cnt += episode_skill_success
            skill_success_cnt_unique += episode_skill_success_unique
            print('episode skill success', episode_skill_success)

        if task_success:
            test_success_rate += 1
        # save gif
        if args.save_gif:
            imageio.mimsave(os.path.join(save_dir, 'episode{}_success{}.gif'.format(
                ep, int(task_success))), env.rgb_list, duration=0.1)
        # save snapshots
        save_dir_snapshots = os.path.join(
            save_dir, 'episode{}_success{}'.format(ep, int(task_success)))
        if not os.path.exists(save_dir_snapshots):
            os.mkdir(save_dir_snapshots)
        for i, (sk, im) in enumerate(episode_snapshots):
            imageio.imsave(os.path.join(save_dir_snapshots,
                           '{}_{}.png'.format(i, sk)), im)
        print()

    draw_skill_success_figure(
        skill_sequence=skill_sequence, skill_sequence_unique=skill_sequence_unique,
        skill_success_cnt=skill_success_cnt, skill_success_cnt_unique=skill_success_cnt_unique,
        args=args, save_dir=save_dir
    )

    print('success_skills', skill_success_cnt/args.test_episode,
          'success_skills_unique', skill_success_cnt_unique/args.test_episode)
    test_success_rate /= args.test_episode
    print('success rate:', test_success_rate)


if __name__ == "__main__":
    args = ArgParser.create_parser()
    print(args)
    # main(args)
