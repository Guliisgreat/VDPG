import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type
from einops import rearrange, repeat, reduce

from segment_anything.modeling import TwoWayTransformer


class DomainGuidance(nn.Module):
    """
    Inspired by the decoder in Segment Anything.
    But in our design, we work on classification task.

    """

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_class: int,
    ) -> None:
        """
        Do sth

        Arguments:
            transformer_dim (int): the channel dimension of the transformer
            transformer (nn.Module): the transformer used to predict masks
            activation (nn.Module): the type of activation to use when upscaling masks

        """
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.cls_token = nn.Embedding(1, transformer_dim)

        # TODO: Need to consider more complex MLP and its initialization
        self.cls_head = nn.Linear(transformer_dim * 2, num_class)

    def forward(
        self, 
        image_embed: torch.Tensor, 
        image_pe: torch.Tensor,
        prompt_embed: torch.Tensor,
        cls_tokens_out=None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ToDO

        Arguments:
            image_embed (torch.Tensor): The embeddings from the image encoder. BxCxHxW
            image_pe (torch.Tensor): The positional encoding in the shape of image_embeddings. CxHxW
            prompt_embed (torch.Tensor): The embeddings of the generated domain prompt. LxC
        
        Return:
            logits (torch.Tensor): the predicted logits of the image
        """
        batch_size, c, h, w = image_embed.shape
        image_pe =  repeat(image_pe, 'c h w -> b c h w', b=batch_size)

        prompt_embed_batch = repeat(prompt_embed, 'l c -> b l c', b=batch_size)
        cls_tokens_dim = repeat(cls_tokens_out, 'b c -> b l c', l=1)
        tokens = torch.cat([cls_tokens_dim, prompt_embed_batch], dim=1)

        output_tokens, keys = self.transformer(image_embed, image_pe, tokens)


        # cls_token = reduce(output_tokens, "b l c -> b c", "mean")
        cls_token = reduce(output_tokens, "b l c -> b c", "max")

        keys_reduce = reduce(keys, "b l c -> b c", "max")
        cls_token = torch.cat([cls_token, cls_tokens_out], dim=1)

        logits = self.cls_head(cls_token)

        return logits, cls_token
