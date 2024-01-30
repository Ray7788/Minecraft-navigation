import torch
from torch import nn
import numpy as np
from .videoCLIP import videoCLIP
from .modules import VisionTransformer, GPT, TemporalTransformer, AdapterHead


def build_pretrain_model(image_config, text_config, temporal_config, adapter_config, state_dict=None) -> videoCLIP:
    """
    Build MineCLIP model
    Args:
        image_config: config for image encoder
        text_config: config for text encoder
        temporal_config: config for temporal encoder
        adapter_config: config for adapter head
        state_dict: pretrained weights
    Returns:
        model: assembled videoCLIP model
    """
    image_encoder = VisionTransformer(resolution=image_config['resolution'], patch_size=image_config['patch_size'],
                                      width=image_config['width'], layers=image_config['layers'],
                                      heads=image_config['heads'], output_dim=image_config['output_dim'])

    text_encoder = GPT(embed_dim=text_config['embed_dim'], context_length=text_config['context_length'], vocab_size=text_config['vocab_size'],
                       layers=text_config['layers'], width=text_config['width'], heads=text_config['heads'])

    temporal_encoder = TemporalTransformer(input_dim=temporal_config['input_dim'], depth=temporal_config['depth'], num_heads=temporal_config['num_heads'],
                                           max_seq_len=temporal_config['video_seq_len'], ff_glu=True, ff_swish=True)

    reward_adapter = AdapterHead(
        adapter_config['video_layers'], adapter_config['text_layers'], adapter_config['feature_dim'])

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    model = videoCLIP(image_encoder, text_encoder,
                      temporal_encoder, reward_adapter, logit_scale)

    if not state_dict is None:
        state_dict_back = model.state_dict()
        state_dict_back.update(state_dict)
        model.load_state_dict(state_dict_back)

    return model
