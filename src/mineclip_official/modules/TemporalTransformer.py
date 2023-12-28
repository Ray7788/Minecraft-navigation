from x_transformers.x_transformers import Encoder, ContinuousTransformerWrapper
import torch
from torch import nn


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int = None,
        depth: int,
        num_heads: int,
        max_seq_len: int,
        # ----- extra tricks, see x_transformers repo ----
        ff_glu=False,
        ff_swish=False,
        attn_one_kv_head=False,
        rel_pos_bias=False,
    ):
        """
        Reference arch:
            bert_base:
                embed_dim = 768
                depth = 12
                num_heads = 12
            bert_large:
                embed_dim = 1024
                depth = 24
                num_heads = 16
        Args:
            input_dim: continuous input feature dimension
            max_seq_len: max sequence length
            embed_dim: embedding dimension, if None, then it is the same as input_dim
                BUT will not add a projection layer from input -> first embedding
                if embed_dim is specified, a projection layer will be added even if
                input_dim == embed_dim
        """
        super().__init__()
        assert isinstance(max_seq_len, int)
        assert isinstance(input_dim, int)
        assert isinstance(depth, int)
        assert isinstance(num_heads, int)

        self.model = ContinuousTransformerWrapper(
            max_seq_len=max_seq_len,
            attn_layers=Encoder(
                dim=input_dim if embed_dim is None else embed_dim,
                depth=depth,
                heads=num_heads,
                ff_glu=ff_glu,
                ff_swish=ff_swish,
                attn_one_kv_head=attn_one_kv_head,
                rel_pos_bias=rel_pos_bias,
            ),
            # if embed_dim is None, do NOT add an input feature projection layer
            dim_in=None if embed_dim is None else input_dim,
            dim_out=None,
        )
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.num_heads = num_heads

    @property
    def output_dim(self):
        return self.input_dim if self.embed_dim is None else self.embed_dim

    def forward(self, x):
        B, L, F = x.size()
        x = self.model(x)
        x = x.mean(dim=1)
        assert x.shape == (B, self.output_dim)
        return x
