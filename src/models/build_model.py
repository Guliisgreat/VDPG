import os
from typing import List

import torch.nn as nn
import open_clip
from segment_anything.modeling.transformer import TwoWayTransformer

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vdpg import VisualDomainPromptGenerator
from src.models.prompt_generator import DomainPromptGenerator
from src.models.guidance import DomainGuidance

from src.lightning.utils import PROJECT_ROOT
from src.models.contrastive_loss import _CONTRASTIVE_LOSS


_M2FEATDIM = {
    "sup_vitb16_224": 768,
    "sup_vitb16": 768,
    "sup_vitl16_224": 1024,
    "sup_vitl16": 1024,
    "sup_vitb8_imagenet21k": 768,
    "sup_vitb16_imagenet21k": 768,
    "sup_vitb32_imagenet21k": 768,
    "sup_vitl16_imagenet21k": 1024,
    "sup_vitl32_imagenet21k": 1024,
    "sup_vith14_imagenet21k": 1280,
    "clip_vit_L14": 1024,
    "clip_vit_b16": 768,
}


def _freeze_components(model: nn.Module, components: List[str] = "image_encoder"):
    print("frozn the components: ", components)
    for component_name in components:
        if component_name == "image_encoder":
            for _, p in model.image_encoder.named_parameters():
                p.requires_grad = False
        if component_name == "prompt_generator":
            for _, p in model.prompt_generator.named_parameters():
                p.requires_grad = False
    return model


def _build_clip_image_encoder(
    model_name: str,
    pretrained_source: str,
    pretrained_weights_dir: str,
    freeze: bool = True,
):
    pretrained_weights_dir = os.path.join(PROJECT_ROOT, pretrained_weights_dir)
    image_encoder = open_clip.create_model(
        model_name=model_name,
        pretrained=pretrained_source,
        cache_dir=pretrained_weights_dir,
    ).visual
    image_encoder.proj = None
    image_encoder.output_tokens = True

    return image_encoder


def build_vdpg(
    CLIP_model_name: str,
    CLIP_checkpoints_source: str,
    CLIP_checkpoints_dir: str,
    ViT: str = "clip_vit_L14",
    num_prompts: int = 50,
    num_cross_attn_heads: int = 1,
    per_cross_attn_head_dim: int = 64,
    attention_dropout: float = 1.0,
    ffn_dropout: float = 1.0,
    contrastive_loss_type: str = "soft",
    contrastive_loss_temperature: float = 1.0,
    encoder_depth: int = 2,
    decoder_depth: int = 1,
    decoder_mlp_dim: int = 2048,
    decoder_num_heads: int = 8,
    decoder_attention_downsample_rate: int = 1,
    decoder_skip_first_layer_pe: bool = False,
    num_class: int = 182,
    freeze_components: List[str] = ["image_encoder"],
    correlation_loss: bool = True,
    use_prompt_loss: bool = True,
):
    print(
        ":::check decoder ", encoder_depth, decoder_depth, decoder_num_heads, num_class
    )
    embed_dim = _M2FEATDIM[ViT]
    model = VisualDomainPromptGenerator(
        image_encoder=_build_clip_image_encoder(
            model_name=CLIP_model_name,
            pretrained_source=CLIP_checkpoints_source,
            pretrained_weights_dir=CLIP_checkpoints_dir,
            freeze=True,
        ),
        prompt_generator=DomainPromptGenerator(
            embed_dim=embed_dim,
            num_prompts=num_prompts,
            num_cross_attention_heads=num_cross_attn_heads,
            depth=encoder_depth,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            use_prompt_loss=use_prompt_loss,
            contrastive_loss=_CONTRASTIVE_LOSS[contrastive_loss_type](
                temperature=contrastive_loss_temperature
            ),
            use_correlation_loss=correlation_loss,
        ),
        guidance_module=DomainGuidance(
            transformer_dim=embed_dim,
            transformer=TwoWayTransformer(
                depth=decoder_depth,
                embedding_dim=embed_dim,
                mlp_dim=decoder_mlp_dim,
                num_heads=decoder_num_heads,
                attention_downsample_rate=decoder_attention_downsample_rate,
            ),
            num_class=num_class,
        ),
    )

    model = _freeze_components(model, components=freeze_components)

    return model
