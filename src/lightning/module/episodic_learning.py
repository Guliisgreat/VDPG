import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm

import src.utils.logging as logging
import torch.optim as optim

logger = logging.get_logger("smart_canada_goose")


def prepare_train_inputs(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    domain_ids: torch.Tensor,
    metadata: torch.Tensor,
    support_ratio: float = 0.5,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Spilit the batch into the support set for adaptation and the query set for inference.
    During training
    - Each batch is divided into three segments: support_positive, support_negative, and query.
    - The support-set includes a subset of examples from a positive domain, termed support_positive, 
      and additional examples from various negative domains, termed support_negative.

    Arguments:
        inputs (torch.Tensor): The images in a mini-batch. BxCxHxW
        labels (torch.Tensor): The labels in a mini-batch.
        domain_ids (torch.Tensor): The domain ids of images in a mini-batch.
        support_ratio (float): The ratio of the number of images as the support_positive.
    Returns:
    """

    positive_domian_id = domain_ids[0]
    first_change_index = next(
        (i for i, x in enumerate(domain_ids) if x != positive_domian_id),
        len(domain_ids),
    )

    positive_inputs = inputs[:first_change_index]
    n_positive = len(positive_inputs)
    n_support_positive = round(n_positive * support_ratio)

    support_inputs = torch.cat(
        [inputs[:n_support_positive], inputs[first_change_index:]]
    )
    support_targets = torch.cat(
        [labels[:n_support_positive], labels[first_change_index:]]
    )
    support_domain_ids = torch.cat(
        [domain_ids[:n_support_positive], domain_ids[first_change_index:]]
    )

    query_inputs = inputs[n_support_positive:first_change_index]
    query_targets = labels[n_support_positive:first_change_index]
    query_metadata = metadata[n_support_positive:first_change_index]
    query_domain_ids = domain_ids[n_support_positive:first_change_index]

    return (
        support_inputs,
        support_targets,
        support_domain_ids,
        query_inputs,
        query_targets,
        query_metadata,
        query_domain_ids,
    )


def prepare_test_inputs(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    metadata: torch.Tensor,
    domain_ids: torch.Tensor,
    support_size: int = 1,
    is_adapt: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_inputs = len(inputs)
    if num_inputs <= support_size:
        return inputs, labels, domain_ids, inputs, labels, metadata

    if not is_adapt:
        return inputs, labels, domain_ids, inputs, labels, metadata
    else:
        support_inputs = inputs[:support_size]
        support_targets = labels[:support_size]
        support_domain_ids = domain_ids[:support_size]
        return (
            support_inputs,
            support_targets,
            support_domain_ids,
            inputs,
            labels,
            metadata,
        )


def get_domain_ids(batch, datamodule):
    metadata = batch[-1]
    domain_ids = datamodule.grouper.metadata_to_group(metadata.cpu())
    if not isinstance(domain_ids, torch.Tensor):
        raise ValueError(f"The current batch has not domain_ids: {batch, metadata}")
    return domain_ids


class DomainSpecificEpisodicLearningLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_func: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        train_support_ratio: float = 0.5,
        train_coef_prompt_loss: float = 1.0,
        test_support_size: float = 1,
        train_coef_corr_loss: float = 0.1,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = model
        self.loss_func = loss_func

        self.train_support_ratio = train_support_ratio
        self.test_support_size = test_support_size

        self.coef = train_coef_prompt_loss
        self.corr_coef = train_coef_corr_loss
        self.validation_step_outputs = []

    def adapt_and_inference(self, query_imgs, support_imgs, support_domain_ids):
        prompt_embed, prompt_loss, corr_loss = self.model.generate_prompt(
            support_imgs, support_domain_ids
        )
        query_outputs, embedding = self.model(query_imgs, prompt_embed)
        return query_outputs, prompt_loss, corr_loss, embedding

    def inference(self, query_imgs):
        query_outputs, embedding = self.model(query_imgs)
        return query_outputs, embedding

    def training_step(self, batch, batch_idx):
        inputs, labels, metadata = batch

        domain_ids = get_domain_ids(batch, self.trainer.datamodule)
        (
            support_inputs,
            _,
            support_domain_ids,
            query_inputs,
            query_targets,
            query_metadata,
            _,
        ) = prepare_train_inputs(
            inputs, labels, domain_ids, metadata, self.train_support_ratio
        )

        query_outputs, prompt_loss, corr_loss, _ = self.adapt_and_inference(
            query_inputs, support_inputs, support_domain_ids
        )
        preds = query_outputs.argmax(dim=1, keepdim=True).view_as(query_targets)

        loss = self.loss_func(query_outputs, query_targets)
        total_loss = loss + self.coef * prompt_loss + self.corr_coef * corr_loss

        metrics, _ = self.trainer.datamodule.train_dataset.eval(
            preds.cpu(), query_targets.cpu(), metadata=query_metadata.cpu()
        )
        self.log("train_top1_acc", metrics["acc_avg"])

        self.log("train_ce_loss", loss, on_step=True, on_epoch=False)
        self.log("train_prompt_loss", prompt_loss, on_step=True, on_epoch=False)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=False)
        self.log("train_corr_los", corr_loss, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Each dataloader has multiple mini-batches of images from one test domain.

        For instance, 'dataloader_ids' begins at 0, meaning that all data samples from the 0th domain
        are loaded in mini-batches. Consequently, `batch_idx` will range from 0 to N until
        the dataloader completes processing of the current domain and increments the `dataloader_ids`
        to move to the next domain.

        In few-shot test-time domain adaptation (FSTT-DA), the test-time adaptation process
        for each domain proceeds as follows:
        1. The model initially adapts using the first mini-batch (support-set), which consists of few-shot data.
        2. Following adaptation, the model performs inference on all mini-batches (query-set).
        """
        imgs, targets, metadata = batch

        domain_ids = get_domain_ids(batch, self.trainer.datamodule)
        if len(torch.unique(domain_ids)) != 1:
            raise ValueError(
                "The images in a mini-batch must come from a same domain. But the domain ids in the current batch is {}".format(
                    domain_ids
                )
            )
        domain_id = domain_ids.tolist()[0]

        if (
            batch_idx == 0
        ):  # Step 1: Adapt and Inference on the the first mini-batch (support-set)
            (
                support_imgs,
                _,
                support_domain_ids,
                query_imgs,
                query_targets,
                query_metadata,
            ) = prepare_test_inputs(
                imgs,
                targets,
                metadata,
                domain_ids,
                self.test_support_size,
                is_adapt=True,
            )
            query_outputs, prompt_loss, _, _ = self.adapt_and_inference(
                query_imgs, support_imgs, support_domain_ids
            )
            self.log("val_prompt_loss", prompt_loss, on_step=False, on_epoch=True)
        else:  # Step 2: Perform inference on the rest of mini-batches.
            (
                support_imgs,
                _,
                support_domain_ids,
                query_imgs,
                query_targets,
                query_metadata,
            ) = prepare_test_inputs(
                imgs,
                targets,
                metadata,
                domain_ids,
                self.test_support_size,
                is_adapt=False,
            )
            query_outputs, _ = self.inference(query_imgs)

        preds = query_outputs.argmax(dim=1, keepdim=True).view_as(query_targets)

        results = dict()
        results["logits"] = query_outputs
        results["predictions"] = preds
        results["targets"] = query_targets
        results["metadata"] = query_metadata
        results["domain_id"] = domain_id

        self.validation_step_outputs.append(results)

    def on_validation_epoch_start(self) -> None:
        logger.info("Start validation ...")
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        # 1. Resume the experiment and skip validation
        if len(self.validation_step_outputs) < 15:
            logger.info(
                f"There is no predictions in Epoch: {self.trainer.current_epoch}. Skip the validation."
            )

            self.log("r_all", 0)
            self.log("test_ood_acc_avg", 0)
            self.log("val_ood_f1_score", 0)
            return

        # 2. Calculate metrics in ood_test
        preds = []
        gts = []
        metadata = []
        for r in self.validation_step_outputs:
            if r["domain_id"] in self.trainer.datamodule.ood_test_domain_ids:
                preds.append(r["predictions"])
                gts.append((r["targets"]))
                metadata.append(r["metadata"])

        logger.info("OOD data number: {}".format(len(metadata)))

        ood_metrics, _ = self.trainer.datamodule.ood_test_dataset.eval(
            torch.cat(preds).cpu(),
            torch.cat(gts).cpu(),
            metadata=torch.cat(metadata).cpu(),
        )

        if self.trainer.datamodule.dataset.name == "fmow":
            metric_names = ["acc_avg", "acc_worst_region"]
        elif self.trainer.datamodule.dataset.name == "iwildcam":
            metric_names = ["acc_avg", "F1-macro_all"]
        else:
            metric_names = ["acc_avg"]

        def log_eval_metrics(
            metrics,
            split="id",
            metric_names=["acc_avg", "F1-macro_all", "acc_worst_region"],
        ):
            for metric_name in metric_names:
                self.log(f"test_{split}_{metric_name}", metrics[metric_name])
                logger.info(
                    f"Epoch {self.trainer.current_epoch}: test_{split}_{metric_name} = {metrics[metric_name]}"
                )

        log_eval_metrics(ood_metrics, split="ood", metric_names=metric_names)

        # 3. Calculate metrics in id_test (if required)
        if self.trainer.datamodule.dataset.id_test_split:
            preds = []
            gts = []
            metadata = []
            for r in self.validation_step_outputs:
                if r["domain_id"] in self.trainer.datamodule.id_test_domain_ids:
                    preds.append(r["predictions"])
                    gts.append((r["targets"]))
                    metadata.append(r["metadata"])

            logger.info("ID data number: {}".format(len(metadata)))
            id_metrics, _ = self.trainer.datamodule.id_test_dataset.eval(
                torch.cat(preds).cpu(),
                torch.cat(gts).cpu(),
                metadata=torch.cat(metadata).cpu(),
            )
            log_eval_metrics(id_metrics, split="id", metric_names=metric_names)

        self.validation_step_outputs.clear()

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.model.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
