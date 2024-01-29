import matplotlib.pyplot as plt
import os


def draw_skill_success_figure(skill_sequence, skill_sequence_unique, skill_success_cnt, skill_success_cnt_unique, args, save_dir):
    """
    Draw the skill success figure
    """
    # draw skill success figure
    plt.bar([i for i in range(len(skill_sequence))], skill_success_cnt/args.test_episode, align="center", color="b",
        tick_label=skill_sequence)
    plt.ylabel('success rate')
    plt.title('Success Rate for Each Skill')
    plt.savefig(os.path.join(save_dir,'success_skills.png'))
    plt.cla()

    plt.bar([i for i in range(len(skill_sequence_unique))], skill_success_cnt_unique/args.test_episode, align="center", color="b",
        tick_label=skill_sequence_unique)
    plt.ylabel('success rate')
    plt.title('Success Rate for Each Skill (Unique)')
    plt.savefig(os.path.join(save_dir,'success_skills_unique.png'))
    plt.cla()