import numpy as np
import torch
from torch import nn


class AdapterHead(nn.Module):
    def __init__(self, video_adapter_layers, text_adapter_layers, feature_dim) -> None:
        super().__init__()
        self.video_adapter_layers = video_adapter_layers
        self.text_adapter_layers = text_adapter_layers
        self.feature_dim = feature_dim

        self.video_residual_weight = None
        self.text_residual_weight = None

        if video_adapter_layers == 0:
            self.video_adapter = nn.Identity()
        else:
            self.video_adapter = nn.Sequential(*([nn.Linear(feature_dim, feature_dim), nn.ReLU()]*(
                video_adapter_layers-1)), nn.Linear(feature_dim, feature_dim))
            self.video_residual_weight = nn.Parameter(torch.tensor(4.0))

        if text_adapter_layers == 0:
            self.text_adapter = nn.Identity()
        else:
            self.text_adapter = nn.Sequential(*([nn.Linear(feature_dim, feature_dim), nn.ReLU()]*(
                text_adapter_layers-1)), nn.Linear(feature_dim, feature_dim))
            self.text_residual_weight = nn.Parameter(torch.tensor(4.0))

    def forward(self, video_features, text_features):
        if self.video_residual_weight is None:
            adapted_video = self.video_adapter(video_features)
        else:
            res = torch.sigmoid(self.video_residual_weight)
            adapted_video = res*video_features + \
                (1.0-res)*self.video_adapter(video_features)

        if self.text_residual_weight is None:
            adapted_text = self.text_adapter(text_features)
        else:
            res = torch.sigmoid(self.text_residual_weight)
            adapted_text = res*text_features + \
                (1.0-res)*self.text_adapter(text_features)
        return adapted_video, adapted_text
