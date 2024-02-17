
import torch

mineclip_weight = torch.load('mineclip_official/attn_updated.pth')
clip4mc_weight = torch.load('mineclip_official/CLIP4MC.pt')
   
print('len=', len(mineclip_weight.keys()))          
print('keys():', mineclip_weight.keys())   

print('len=', len(clip4mc_weight.keys()))          
print('keys():', clip4mc_weight.keys())   

# for item in clip4mc_weight['model'].keys():
#     print(item)
#     if item in mineclip_weight.keys():
#         mineclip_weight[item] = clip4mc_weight['model'][item]


# 保存权重文件
# torch.save(mineclip_weight,'new_weights.pth')


# 未使用
# import csv

# # 创建一个新的CSV文件
# with open('weights.csv', 'w', newline='') as csvfile:
#     # 创建一个CSV写入器
#     writer = csv.writer(csvfile)

#     # 写入标题行
#     writer.writerow(['Key', 'Mineclip Weight', 'Clip4MC Weight'])

#     # 遍历mineclip_weight的键
#     for key in mineclip_weight.keys():
#         # 如果key也在clip4mc_weight中，那么将两个权重写入同一行
#         if key in clip4mc_weight.keys():
#             writer.writerow([key, mineclip_weight[key].cpu().numpy(), clip4mc_weight[key].cpu().numpy()])
#         # 否则，只写入mineclip_weight
#         else:
#             writer.writerow([key, mineclip_weight[key].cpu().numpy(), 'N/A'])

#     # 遍历clip4mc_weight的键，找出只在clip4mc_weight中的键
#     for key in clip4mc_weight.keys():
#         if key not in mineclip_weight.keys():
#             writer.writerow([key, 'N/A', clip4mc_weight[key].cpu().numpy()])

# import csv

# # 创建一个新的CSV文件，两个相同则保存在一起，否则分开保存
# with open('weights.csv', 'w', newline='') as csvfile:
#     # 创建一个CSV写入器
#     writer = csv.writer(csvfile)

#     # 写入标题行
#     writer.writerow(['Key', 'Mineclip Weight Length', 'Clip4MC Weight Length'])

#     # 遍历mineclip_weight的键
#     for key in mineclip_weight.keys():
#         # 如果key也在clip4mc_weight中，那么将两个权重的长度写入同一行
#         if key in clip4mc_weight.keys():
#             writer.writerow([key, mineclip_weight[key].numel(), clip4mc_weight[key].numel()])
#         # 否则，只写入mineclip_weight的长度
#         else:
#             writer.writerow([key, mineclip_weight[key].numel(), 'N/A'])

#     # 遍历clip4mc_weight的键，找出只在clip4mc_weight中的键
#     for key in clip4mc_weight.keys():
#         if key not in mineclip_weight.keys():
#             writer.writerow([key, 'N/A', clip4mc_weight[key].numel()])

# 创建一个新的文件保存重名
with open('common_keys2.txt', 'w') as file:
    # 遍历mineclip_weight的键
    for key in mineclip_weight.keys():
        # 如果key也在clip4mc_weight中，那么将它写入文件
        if key in clip4mc_weight.keys():
            file.write(f'{key}, {mineclip_weight[key].numel()}, {clip4mc_weight[key].numel()}\n')


import csv

# 创建一个新的CSV文件用于保存mineclip_weight的键和长度
with open('mineclip_weights2.csv', 'w', newline='') as csvfile:
    # 创建一个CSV写入器
    writer = csv.writer(csvfile)

    # 写入标题行
    writer.writerow(['Key', 'Mineclip Weight Length'])

    # 遍历mineclip_weight的键
    for key in mineclip_weight.keys():
        # 将键和长度写入CSV文件
        writer.writerow([key, mineclip_weight[key].numel()])

# 创建一个新的CSV文件用于保存clip4mc_weight的键和长度
with open('clip4mc_weights2.csv', 'w', newline='') as csvfile:
    # 创建一个CSV写入器
    writer = csv.writer(csvfile)

    # 写入标题行
    writer.writerow(['Key', 'Clip4MC Weight Length'])

    # 遍历clip4mc_weight的键
    for key in clip4mc_weight.keys():
        # 将键和长度写入CSV文件
        writer.writerow([key, clip4mc_weight[key].numel()])


print('Done!')