from typing import Optional, List
from dataclasses import dataclass
import copy
import numpy as np

from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from pytorch_lightning import seed_everything

import wilds
from wilds.common.data_loaders import get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.datasets.dataloader import get_contrastive_train_loader
from src.datasets.datasets_cfgs import DatasetConfigurator, _DATASETS
import src.utils.logging as logging

logger = logging.get_logger("smart_canada_goose")


def get_subset_with_domain_id(dataset, grouper, domain=None):
    if type(dataset) != wilds.datasets.wilds_dataset.WILDSSubset:
        raise NotImplementedError
    subset = copy.deepcopy(dataset)

    if domain is not None:
        domain_name = grouper.groupby_fields[0]

        domain_idx = dataset.metadata_fields.index(domain_name)

        idx = np.argwhere(
            np.isin(
                subset.dataset.metadata_array[:, domain_idx][subset.indices], domain
            )
        ).ravel()
        subset.indices = subset.indices[idx]
    else:
        raise NotImplementedError

    return subset


def get_test_loaders(dataset, grouper, batch_size=16, num_workers=0):
    all_domain_ids = list(
        set(
            grouper.metadata_to_group(
                dataset.dataset.metadata_array[dataset.indices]
            ).tolist()
        )
    )
    test_domain_loaders = []

    for domain in all_domain_ids:
        domain_data = get_subset_with_domain_id(dataset, grouper, domain=domain)
        domain_loader = get_eval_loader(
            "standard", domain_data, batch_size=batch_size, num_workers=num_workers
        )
        test_domain_loaders.append(domain_loader)

    return test_domain_loaders


class LabeledDomainContrastiveDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str = "/data01/tao/wilds/data",
        dataset_name: str = "iwildcam",
        input_resolution: int = 448,
        domain_type: str = "contrastive",
        batch_size: int = 16,
        num_workers: int = 0,
        n_negative_groups_per_batch: int = 4,
        n_points_per_negative_group: int = 2,
    ):
        super().__init__()

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.input_resolution = input_resolution
        self.domain_type = domain_type
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.n_negative_groups_per_batch = n_negative_groups_per_batch
        self.n_points_per_negative_group = n_points_per_negative_group
        
        self.dataset = _DATASETS[dataset_name]

    def setup(self, stage: Optional[str] = None):

        configurator = DatasetConfigurator(
            self.dataset_name, self.data_path, self.input_resolution
        )
        dataset_config = configurator.get_dataset_configuration()
        datasets = dataset_config["dataset"]
        train_transform = dataset_config["train_transform"]
        val_transform = dataset_config["val_transform"]
        self.uniform_over_group = dataset_config["uniform_over_group"]
        self.uniform_sampler = dataset_config["uniform_sampler"]

        logger.info(
            ":::: Train transform for DATASET: {}".format(self.dataset_name),
            train_transform,
        )
        logger.info(
            ":::: Valid transform for DATASET: {}".format(self.dataset_name),
            val_transform,
        )

        self.grouper = CombinatorialGrouper(datasets, [self.dataset.domain_name])
        if self.dataset.name == "fmow":
            self.test_grouper = CombinatorialGrouper(
                datasets, [self.dataset.test_domain_name]
            )

        # Train-set
        self.train_dataset = datasets.get_subset(
            self.dataset.train_split, transform=train_transform
        )

        self.train_domain_ids = list(
            set(
                self.grouper.metadata_to_group(
                    self.train_dataset.dataset.metadata_array[
                        self.train_dataset.indices
                    ]
                ).tolist()
            )
        )
        logger.info(
            f"{self.dataset.name}'s train-set has {len(self.train_dataset)} data examples, {self.train_dataset.n_classes} classes and {len(self.train_domain_ids)} domains"
        )

        # Test-set (id_test_set may not exist in some wild datasets)
        if self.dataset.id_test_split:
            self.id_test_dataset = datasets.get_subset(
                self.dataset.id_test_split, transform=val_transform
            )
            self.id_test_domain_ids = list(
                set(
                    self.grouper.metadata_to_group(
                        self.id_test_dataset.dataset.metadata_array[
                            self.id_test_dataset.indices
                        ]
                    ).tolist()
                )
            )
            logger.info(
                f"{self.dataset.name}'s id-test-set has {len(self.id_test_dataset)} data examples, {self.id_test_dataset.n_classes} classes and {len(self.id_test_domain_ids)} domains"
            )
            logger.info("check ID domains: ", self.id_test_domain_ids)

        self.ood_test_dataset = datasets.get_subset(
            self.dataset.ood_test_split, transform=val_transform
        )

        self.ood_test_domain_ids = list(
            set(
                self.grouper.metadata_to_group(
                    self.ood_test_dataset.dataset.metadata_array[
                        self.ood_test_dataset.indices
                    ]
                ).tolist()
            )
        )

        logger.info(
            f"{self.dataset.name}'s ood-test-set has {len(self.ood_test_dataset)} data examples, {self.ood_test_dataset.n_classes} classes and {len(self.ood_test_domain_ids)} domains"
        )

    def train_dataloader(self) -> DataLoader:
        if self.domain_type == "contrastive":
            loader = get_contrastive_train_loader(
                self.train_dataset,
                grouper=self.grouper,
                n_negative_groups_per_batch=self.n_negative_groups_per_batch,
                n_points_per_negative_group=self.n_points_per_negative_group,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                uniform_over_groups=self.uniform_over_group,
                uniform_sampler=self.uniform_sampler,
            )
        else:
            raise NotImplementedError
        return loader

    def val_dataloader(self) -> List[DataLoader]:
        """
        The doc for combining dataloaders:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.utilities.combined_loader.html


        """
        loaders = []
        ood_loaders = get_test_loaders(
            self.ood_test_dataset,
            self.grouper,
            batch_size=self.batch_size,
            num_workers=0,
        )
        loaders += [ood_loaders]

        if self.dataset.id_test_split:
            id_loaders = get_test_loaders(
                self.id_test_dataset,
                self.grouper,
                batch_size=self.batch_size,
                num_workers=0,
            )
            loaders += id_loaders

        return CombinedLoader(loaders, "sequential")


if __name__ == "__main__":
    seed_everything(42)
    from torchvision.utils import save_image

    bs = 64
    datamodule = LabeledDomainContrastiveDataModule(
        # data_path="/data01/tao/wilds/data",
        data_path="/data01/dataset",
        dataset_name="domain_net_clipart",
        domain_type="contrastive",
        batch_size=bs,
        num_workers=8,
        input_resolution=336,
        # n_negative_groups_per_batch=1,
        # n_points_per_negative_group= 16,
    )
    datamodule.setup()

    print(datamodule.ood_test_domain_ids)
    domain_loaders = datamodule.train_dataloader()
    print(len(domain_loaders))
    domain_loaders = datamodule.val_dataloader()
    print(len(iter(datamodule.val_dataloader())))

