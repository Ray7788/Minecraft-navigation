import numpy as np
import torch
from mineagent.features.voxel.flattened_voxel_block import VOXEL_BLOCK_NAME_MAP
from mineagent.batch import Batch


def preprocess_obs(obs, device):
    """
    Here you preprocess the raw env obs to pass to the agent.
    Preprocessing includes, for example, use MineCLIP to extract image feature and prompt feature,
    flatten and embed voxel names, mask unused obs, etc.
    """
    B = 1

    def cvt_voxels(vox):
        """
        Convert the 3*3*3 voxel block to a 3*3*3*3 embedding.
        """
        ret = np.zeros(3*3*3, dtype=np.long)
        for i, v in enumerate(vox.reshape(3*3*3)):
            if v in VOXEL_BLOCK_NAME_MAP:
                ret[i] = VOXEL_BLOCK_NAME_MAP[v]
        return ret

    # I consider the move and functional action only, because the camera space is too large?
    # construct a 3*3*4*3 action embedding
    def cvt_action(act):
        """
        Convert the action to a 3*3*4*3 embedding.
        act[5]: functional actions, 0: no_op, 1: use, 2: drop, 3: attack 4: craft 5: equip 6: place 7: destroy
        """
        if act[5] <= 1:
            return act[0] + 3*act[1] + 9*act[2] + 36*act[5]
        elif act[5] == 3:
            return act[0] + 3*act[1] + 9*act[2] + 72
        else:
            # raise Exception('Action[5] should be 0,1,3')
            return 0

    yaw_ = np.deg2rad(obs["location_stats"]["yaw"])
    pitch_ = np.deg2rad(obs["location_stats"]["pitch"])
    compass = torch.as_tensor(
        [np.concatenate(
            [np.cos(yaw_), np.sin(yaw_), np.cos(pitch_), np.sin(pitch_)]
        )], device=device)

    obs_ = {
        "compass": compass,
        "gps": torch.as_tensor([obs["location_stats"]["pos"]], device=device),
        "voxels": torch.as_tensor(
            [cvt_voxels(obs["voxels"]["block_name"])], dtype=torch.long, device=device
        ),
        "biome_id": torch.tensor(
            [int(obs["location_stats"]["biome_id"])], dtype=torch.long, device=device
        ),
        "prev_action": torch.tensor(
            [cvt_action(obs["prev_action"])], dtype=torch.long, device=device
        ),
        "prompt": torch.as_tensor(obs["rgb_emb"], device=device).view(B, 512),
        # this is actually the image embedding, not prompt embedding (for single task)
        # "goal": torch.as_tensor(obs["goal_emb"], dtype=torch.float, device=device).view(B, 6),
    }
    # print(obs_["prev_action"])
    # print(obs_["compass"], yaw_, pitch_)
    # print(obs_["goal"])

    # print(Batch(obs=obs_))
    return Batch(obs=obs_)



# Map mine-agent action to env action.
# [12, 3] action space, 1 choice among walk, jump and camera
# preserve 4 camera actions
def transform_action(act, allow_use=True):
    """
    Transform the action from the agent to the environment.
    """
    assert act.ndim == 2  # (1, 2)
    act = act[0]
    act = act.cpu().numpy()
    act1, act2 = act[0], act[1]

    action = [0, 0, 0, 12, 12, 0, 0, 0]  # self.base_env.action_space.no_op()
    assert act1 < 12
    if act1 == 0:  # no op
        action = action
    elif act1 < 3:  # forward backward
        action[0] = act1
    elif act1 < 5:  # left right
        action[1] = act1 - 2
    elif act1 < 8:  # jump sneak sprint
        action[2] = act1 - 4
    elif act1 == 8:  # camera pitch 10
        action[3] = 10
    elif act1 == 9:  # camera pitch 14
        action[3] = 14
    elif act1 == 10:  # camera yaw 10
        action[4] = 10
    elif act1 == 11:  # camera yaw 14
        action[4] = 14

    assert act2 < 3
    if act2 == 1 and allow_use:  # use
        action[5] = 1
    elif act2 == 2:  # attack
        action[5] = 3
    return action  # (8)
