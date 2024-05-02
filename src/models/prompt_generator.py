from typing import Any, Optional, Tuple, Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat, reduce

from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 52,
        num_cross_attention_heads: int = 1,
        attention_dropout: float = 0,
        ffn_dropout: float = 0,
    ) -> None:
        super().__init__()

        self.pre_norm1 = nn.LayerNorm(embed_dim)
        self.pre_norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_cross_attention_heads,
            batch_first=True,
            dropout=attention_dropout,
        )

        self.pre_norm3 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout=ffn_dropout)

    def forward(self, x, context):
        x = self.pre_norm1(x)
        context = self.pre_norm2(context)
        x = self.cross_attention(query=x, key=context, value=context)[0] + x

        x = self.pre_norm3(x)
        x = self.ffn(x) + x
        return x


class DomainPromptGenerator(nn.Module):
    def __init__(
        self,
        embed_dim: int = 52,
        num_prompts: int = 10,
        num_cross_attention_heads: int = 1,
        depth: int = 1,
        attention_dropout: float = 0,
        ffn_dropout: float = 0,
        use_prompt_loss: bool = True,
        use_correlation_loss: bool = False,
        contrastive_loss: Type[nn.Module] = None,
    ) -> None:
        """
        Generate a domain-specific prompt through cross-attention mechanisms, leveraging the few-shot unlabeled data
        from the support set.
        Specifically, the process involves conditioning the knowledge bank with image features
        extracted from the support set to generate prompts tailored to the specific domain.
        The knowledge bank is designed to encapsulate shareable knowledge applicable across different domains.

        Arguments:
          embed_dim (int): The domain prompt embedding's dimension (D)
          num_prompts (int): The number of embedding in the knowledge bank. (L)
          num_cross_attention_heads (int): The number of heads in cross-attention layer.
          per_head_dim (int): The dimension of each head in in cross-attention layer.
          attention_dropout (float): The dropout rate for the cross-attetnion layer.
          ffn_dropout (float): The dropout rate for the Feed-forward network.
          use_prompt_loss (bool): Whether to use contrastive loss.
          use_correlation_loss (bool): Whether to use corellation loss.
          contrastive_loss (nn.Module): The specific contrastive loss used.
        """
        super().__init__()

        self.use_prompt_loss = use_prompt_loss
        self.use_correlation_loss = use_correlation_loss
        self.contrastive_loss = contrastive_loss

        self.knowledge_bank = nn.Parameter(torch.randn(num_prompts, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                CrossAttentionBlock(
                    embed_dim=embed_dim,
                    num_cross_attention_heads=num_cross_attention_heads,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                )
            )

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # correlation loss
        self.corr_loss = nn.MSELoss()
        self.idt = torch.eye(n=num_prompts)
        self.gt = torch.zeros_like(self.idt)

        print("Using prompt_l: ", self.use_prompt_loss)
        print("Using corr_l: ", self.use_correlation_loss)

    def forward(
        self, image_embeds: torch.Tensor, domain_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
            image_embeds (torch.Tensor): The image embedding of the support images, \
                including the postive images from the same domain and the negative images \
                from other distinct domains. BxCxHxW.
            domain_ids (torch.Tensor)): The domain ids of the examples in the mini-batch. \
                The first S examples share the same domain, considered as the positive domain. \
                The rest of examples sampled from other distinct domains, considered as the negative domains.

        Returns:
            prompt_embed (torch.Tensor)： The embedding for the domain prompt that encodes
                the domain specific knowledge extracted from the support set. 
                The tensor shape is (L, D), where L is the number of prompts in the knowledge bank 
                and D represents embedding dimension.
            prompt_loss (torch.Tensor): The calculated (contrastive) loss for the generated domain prompt.
        """
        b, c, h, w = image_embeds.shape

        image_pe = self.pe_layer((h, w))
        image_pe = repeat(image_pe, "c h w -> b c h w", b=b)
        image_embeds = image_embeds + image_pe
        image_embeds = rearrange(image_embeds, "b c h w -> b (h w) c")

        x = repeat(self.knowledge_bank, "l c -> b l c", b=b)

        for block in self.blocks:
            x = block(x, image_embeds)

        positive_indices = [
            index for (index, item) in enumerate(domain_ids) if item == domain_ids[0]
        ]

        # TODO: Explore other techniques to process the support set's prompt embeds [b, l, c]
        # prompt_embed = reduce(x[positive_indices, :, :], "b l c -> l c", "mean")
        prompt_embed = reduce(x[positive_indices, :, :], "b l c -> l c", "max")

        self.prompt_embed = prompt_embed
        self.has_prompt_emebd = True

        if self.use_prompt_loss:
            # Hardcode for soft_nearest_neighbour_loss
            x = rearrange(x, "b l c -> b (l c)")
            loss = self.contrastive_loss(x, domain_ids)
            if self.use_correlation_loss:
                corr = torch.matmul(
                    self.knowledge_bank, torch.transpose(self.knowledge_bank, 0, 1)
                )

                self.idt = self.idt.to(device=corr.device)
                self.gt = self.gt.to(device=corr.device)
                # print(corr.device, self.idt.device)
                diff = corr - corr * self.idt
                corr_loss = self.corr_loss(diff, self.gt)
                return prompt_embed, loss, corr_loss

            return prompt_embed, loss, 0.0

        # TODO: Add correlation loss

        return prompt_embed, 0.0, 0.0
