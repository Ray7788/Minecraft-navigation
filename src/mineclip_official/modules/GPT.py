
import torch
from torch import nn
from .ResidualAttentionBlock import ResidualAttentionBlock


class GPT(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        layers: int,
        width: int,
        heads: int,
        is_discrete_text: bool = True,
    ):
        """
        Args:
            is_discrete_text: False to use regular discrete tokens
              True for video sequence of image tokens, and `vocab_size` will be
              interpreted as the dim of each image feature.
        """
        super().__init__()
        self.context_length = context_length
        self._width = width
        self._layers = layers
        self.vocab_size = vocab_size

        self._is_discrete_text = is_discrete_text
        if is_discrete_text:
            self.token_embedding = nn.Embedding(vocab_size, width)
        else:
            self.token_embedding = nn.Linear(vocab_size, width, bias=False)
        self.pos_embed = nn.Parameter(torch.empty(self.context_length, width))
        self.blocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width, heads, attn_mask=self.build_attention_mask()
                )
                for _ in range(layers)
            ]
        )

        self.ln_final = nn.LayerNorm(width)
        self.projection = nn.Parameter(torch.empty(width, embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        if self._is_discrete_text:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

        proj_std = (self._width**-0.5) * ((2 * self._layers) ** -0.5)
        attn_std = self._width**-0.5
        fc_std = (2 * self._width) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.projection is not None:
            nn.init.normal_(self.projection, std=self._width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.pos_embed  # x = x + self.pos_embed[: x.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.projection
        return x
