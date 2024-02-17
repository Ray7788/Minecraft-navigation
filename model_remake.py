
import torch

mineclip_weight = torch.load('mineclip_official/attn.pth')
clip4mc_weight = torch.load('mineclip_official/CLIP4MC.pt')
   
print('len=', len(mineclip_weight.keys()))          
print('keys():', mineclip_weight.keys())   

print('len=', len(clip4mc_weight.keys()))          
print('keys():', clip4mc_weight.keys())   

# 创建映射字典
key_mapping = {
    "gpt": "text_encoder",
    "vit": "image_encoder",
    "video_adapter.adapter": "reward_adapter.video_adapter",
    "video_adapter.residual_weight": "reward_adapter.video_residual_weight",
    "temporal_encoder2": "temporal_encoder"
}

# 遍历clip4mc_weight字典的键
for key in list(clip4mc_weight.keys()):
    new_key = key
    # 替换关键字
    for old, new in key_mapping.items():
        new_key = new_key.replace(old, new)
    # 检查新键是否存在于mineclip_weight字典中
    if new_key in mineclip_weight:
        # 替换mineclip_weight字典中的值
        mineclip_weight[new_key] = clip4mc_weight[key]

# 保存更新后的mineclip_weight字典
torch.save(mineclip_weight, 'mineclip_official/attn_updated.pth')