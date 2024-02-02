import numpy as np
import torch
from ...mineagent.batch import Batch
from ...mineagent.features.voxel.flattened_voxel_block import VOXEL_BLOCK_NAME_MAP


def preprocess_obs(obs, device):
    """
    Here you preprocess the raw env obs to pass to the agent.
    Preprocessing includes, for example, use MineCLIP to extract image feature and prompt feature,
    flatten and embed voxel names, mask unused obs, etc.
    """
    B = 1   # prompt related

    def cvt_voxels(vox):
        """
        this function is used to map voxel data to an integer.
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
        this function is used to map action data to an integer. 
        The mapping method involves different elements in the action and ultimately generates an integer representing the action.
        """
        if act[5]<=1:
            return act[0] + 3*act[1] + 9*act[2] + 36*act[5]
        elif act[5]==3:
            return act[0] + 3*act[1] + 9*act[2] + 72
        else:
            #raise Exception('Action[5] should be 0,1,3')
            return 0

    yaw_ = np.deg2rad(obs["location_stats"]["yaw"])
    pitch_ = np.deg2rad(obs["location_stats"]["pitch"])
    obs_ = {
        "compass": torch.as_tensor([np.concatenate([np.cos(yaw_), np.sin(yaw_), np.cos(pitch_), np.sin(pitch_)])], device=device),
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
        "goal": torch.as_tensor(obs["goal_emb"], dtype=torch.float, device=device).view(B, 6), 
    }
    # print("goal")
    # print(obs_["prev_action"]) # Integer
    # print(obs_["compass"], yaw_, pitch_) # tensor([[ 0.8660, -0.5000,  0.5000,  0.8660]]) [5.7595863] [1.0471976]
    # print(obs_["goal"]) # tensor([[ 0.0000,  1.0000,  1.0000,  0.0000, -4.7456, -0.4660]])

    #print(Batch(obs=obs_))
    return Batch(obs=obs_)



# Map agent action to env action.
# [12, 3] action space, 1 choice among walk, jump and camera
# preserve 4 camera actions
def transform_action(act):
    assert act.ndim == 2 # (1, 2)
    act = act[0]
    act = act.cpu().numpy()
    act1, act2 = act[0], act[1]
    
    action = [0,0,0,12,12,0,0,0] #self.base_env.action_space.no_op()
    assert act1 < 12
    if act1 == 0: # no op
        action = action
    elif act1 < 3: # forward backward
        action[0] = act1
    elif act1 < 5: # left right
        action[1] = act1 - 2
    elif act1 < 8: # jump sneak sprint
        action[2] = act1 - 4
    elif act1 == 8: # camera pitch 10
        action[3] = 10
    elif act1 == 9: # camera pitch 14
        action[3] = 14
    elif act1 == 10: # camera yaw 10
        action[4] = 10
    elif act1 == 11: # camera yaw 14
        action[4] = 14

    assert act2 < 3
    '''
    if act2 == 1: # use
        action[5] = 1
    elif act2 == 2: #attack
        action[5] = 3
    '''
    # for find skill, ban(cancel) the use action
    if act2 == 2: #attack
        action[5] = 3
    return action #(8)
