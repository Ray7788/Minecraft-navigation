import torch
from torch import nn
from .ResidualAttentionBlock import ResidualAttentionBlock


class VisionTransformer(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self._resolution = resolution
        self._patch_size = patch_size
        self._layers = layers
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(
            scale * torch.randn(161, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

    # def resize_pos_embed(self, new_resolution):
    #     """
    #     NOTE: call this method AFTER you load pretrained weights!
    #     """
    #     if isinstance(new_resolution, int):
    #         new_resolution = (new_resolution, new_resolution)
    #     else:
    #         assert len(new_resolution) == 2
    #     for r in new_resolution:
    #         assert (
    #             r % self._patch_size == 0
    #         ), f"{new_resolution} is not divisible by {self._patch_size}"

    #     with torch.no_grad():
    #         old_embed = self.pos_embed.data.detach()
    #         cls_embed, old_embed = old_embed[:1], old_embed[1:]
    #         new_embed = interpolate_resize_pos_embed(
    #             old_embed,
    #             self._resolution // self._patch_size,
    #             [r // self._patch_size for r in new_resolution],
    #         )
    #         self.pos_embed = nn.Parameter(torch.cat([cls_embed, new_embed], dim=0))
        
    def forward(self, x: torch.Tensor):
        bs,ts,c,h,w = x.shape
        x = x.reshape(bs*ts,c,h,w)
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.repeat((B, 1, 1)), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.projection is not None:
            x = x @ self.projection

        x = x.reshape(bs,ts,-1)
        return x
