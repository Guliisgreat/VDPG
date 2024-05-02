import math
from typing import Any, Optional, Tuple, Type, List
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sanity_check_domain_ids(domain_ids):
    """
    During training, the first several images must come from the same domain,
    while the rest of images from the other random domains
    """
    positive_domian_id = domain_ids[0]

    # Find the point where the first element changes
    first_change_index = next(
        (i for i, x in enumerate(domain_ids) if x != positive_domian_id),
        len(domain_ids),
    )
    if first_change_index < 1:
        raise ValueError(
            f"In a support set {domain_ids}, there should be at least one data \
                         examples as positives from a same domain"
        )

    # Check the rest of the list for distinct elements
    rest_of_list = domain_ids[first_change_index:]
    if len(set(rest_of_list)) != len(rest_of_list):
        raise ValueError(
            "In a support set, the rest of data examples as the negatives \
                         must be sampled from other distinct domains"
        )


class VisualDomainPromptGenerator(nn.Module):
    def __init__(
        self,
        image_encoder,
        prompt_generator,
        guidance_module,
    ) -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.prompt_generator = prompt_generator
        self.guidance_module = guidance_module

        self._prompt_embed = None
        self.has_prompt_embed = False

        self.norm_feature = False
        if self.norm_feature:
            print("::::: normalize Clip features")

    def generate_prompt(self, support_images: torch.Tensor, domain_ids: torch.Tensor):
        """
        Arguments:
            image_embeds (torch.Tensor): The image embedding of the support images, including the postive images from the same domain \
                and the negative images from other distinct domains. BxCxHxW.
            domain_ids (torch.Tensor)): The domain ids of the examples in the mini-batch. \
                The first S examples share the same domain, considered as the positive domain. \
                The rest of examples sampled from other distinct domains, considered as the negative domains.

        Returns:
            prompt_embed (torch.Tensor)： The embedding for the domain prompt that encodes
                the domain specific knowledge extracted from the support set. 
                The tensor shape is (L, D), where L is the number of elements in the knowledge bank 
                and D represents embedding dimension.
            prompt_loss (torch.Tensor): The calculated (contrastive) loss for the generated domain prompt.

        """
        cls_token, image_embeds = self.image_encoder(support_images)
        b, l, c = image_embeds.shape

        _sanity_check_domain_ids(domain_ids)
        h = w = round(math.sqrt(l))
        image_embeds = rearrange(image_embeds, "b (h w) c -> b c h w", h=h, w=w)
        if self.norm_feature:
            image_embeds = F.normalize(image_embeds, dim=-1)
        prompt_embed, prompt_loss, corr_loss = self.prompt_generator(
            image_embeds, domain_ids
        )

        self._prompt_embed = prompt_embed
        self.has_prompt_embed = True

        return prompt_embed, prompt_loss, corr_loss

    def forward(self, query_images, prompt_embed=None):
        """
        Arguments:
            query_images (torch.Tensor): The images from the query set. BxCxHxW.
            prompt_embed (torch.Tensor)): The embedding of the domain prompt. LxD
        """

        if not isinstance(prompt_embed, torch.Tensor):
            if not self.has_prompt_embed:
                raise ValueError(
                    "Need to use a prompt encoder to produce prompt tokens at first"
                )
            prompt_embed = self._prompt_embed

        cls_token, image_embeds = self.image_encoder(query_images)
        b, l, c = image_embeds.shape
        h = w = round(math.sqrt(l))

        image_embeds = rearrange(image_embeds, "b (h w) c -> b c h w", h=h, w=w)
        image_pe = self.prompt_generator.pe_layer((h, w))  # CxHxW

        if self.norm_feature:
            image_embeds = F.normalize(image_embeds, dim=-1)
            cls_token = F.normalize(cls_token, dim=-1)

        logits, embedding = self.guidance_module(
            image_embeds, image_pe, prompt_embed, cls_token
        )

        return logits, embedding

    def reset_prompt_embed(self):
        self._prompt_embed = None
        self.has_prompt_embed = False

    @property
    def prompt_embed(self):
        if self.has_prompt_embed:
            return self._prompt_embed
        raise ValueError(
            "The prompt encoder have not already generated a prompt embedding."
        )
