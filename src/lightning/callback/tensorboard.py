from typing import Optional, Tuple

import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer


class VisualizeParameterDistributionTensorboard(Callback):
    """
    Use Tensorboard to visualize the change of the distribution of parameters (weights OR gradients) during training

    Example::
        from src.lightning.callbacks import TensorboardModelDistribution
        trainer = Trainer(callbacks=[TensorboardModelDistributio()])
    """

    def __init__(self, distribution_type="weight") -> None:
        """
            Args:
                distribution_type (str): Determine whose distribution to be visualized. 
                    Need to make a choice between "weight" OR "gradient" 
        """
        super().__init__()

        if distribution_type not in ["weight", "gradient"]:
            raise NotImplementedError("Need to make a choice between 'weight' OR 'gradient'")
        self.distribution_type = distribution_type

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # if not isinstance(trainer.logger.experiment, torch.utils.tensorboard.writer.SummaryWriter):
        #     raise ValueError("The logger must be TensorBoard")
        
        if "weight" in self.distribution_type:
            for name, p in pl_module.model.named_parameters():
                if p.nelement() == 0:
                    continue
                if p.requires_grad:
                    trainer.logger.experiment.add_histogram(
                        "weight_" + name, p.data, trainer.global_step
                    )
        else:
            raise NotImplementedError("So far, it only supports to visualize the distribution of weights")
