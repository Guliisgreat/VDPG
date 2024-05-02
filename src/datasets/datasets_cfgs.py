from dataclasses import dataclass

import wilds

from src.datasets.datasets_transform import (
    openclip_transform,
    wilds_default_train_transform,
    fmow_train_transform,
    fmow_test_transform,
    domainnet_train_transform,
    camelyon_train_transform,
)
from src.datasets.multi_source_domain_net import MultiSourceDomainNetDataset


@dataclass
class IwildCam:
    name: str = "iwildcam"
    domain_name: str = "location"
    n_classes: int = 182
    train_split: str = "train"
    val_split: str = "val"
    id_test_split: str = "id_test"
    ood_test_split: str = "test"
    unlabeled_train_split: str = "extra_unlabeled"
    unlabeled_test_split: str = "test"


@dataclass
class Camelyon17:
    name: str = "camelyon17"  # 2 classes
    domain_name: str = "hospital"  # 3 domains in train-set, 1 domain in ood-test
    n_classes: int = 2
    train_split: str = "train"
    val_split: str = "val"
    id_test_split = None
    ood_test_split: str = "test"
    unlabeled_train_split: str = "train_unlabeled"
    unlabeled_test_split: str = "test"


@dataclass
class Rxrx1:
    name: str = (
        "rxrx1"  # 1139 classes  ['cell_type', 'experiment', 'plate', 'well', 'site', 'y', 'from_source_domain']
    )
    domain_name: str = (
        "experiment"  # 33 domians in tranin & id-test, 14 domians in ood-test
    )
    n_classes: int = 1139
    train_split: str = "train"
    val_split: str = "val"
    id_test_split: str = None
    # id_test_split: str = "id_test"
    ood_test_split: str = "test"
    unlabeled_train_split = None
    unlabeled_test_split: str = "test"


@dataclass
class FMoW:
    name: str = "fmow"  # 62 classes
    domain_name: str = "year"  # 12 domains in train & id-test, 2 domains in ood-test
    # domain_name: str = "region" # 12 domains in train & id-test, 2 domains in ood-test
    test_domain_name: str = "year"
    n_classes: int = 62
    train_split: str = "train"
    val_split: str = "val"
    id_test_split: str = "id_test"
    id_test_split: str = None
    ood_test_split: str = "test"
    unlabeled_train_split: str = "train_unlabeled"
    unlabeled_test_split: str = "id_test"


@dataclass
class PovertyMap:
    name: str = "poverty"
    domain_name: str = "country"
    train_split: str = "train"
    val_split: str = "val"
    # id_test_split: str = "id_test"
    id_test_split: str = None
    ood_test_split: str = "test"
    unlabeled_train_split: str = "train_unlabeled"
    n_classes: int = 1


@dataclass
class DomainNet_quickdraw:
    name: str = "domainnet"
    target_domain: str = "quickdraw"
    domain_name: str = "domain"
    n_classes: int = 345
    train_split: str = "train"
    val_split: str = "val"
    id_test_split = None
    ood_test_split: str = "test"
    unlabeled_train_split: str = None
    unlabeled_test_split: str = None


@dataclass
class DomainNet_clipart:
    name: str = "domainnet"
    target_domain: str = "clipart"
    domain_name: str = "domain"
    n_classes: int = 345
    train_split: str = "train"
    val_split: str = "val"
    id_test_split = None
    ood_test_split: str = "test"
    unlabeled_train_split: str = None
    unlabeled_test_split: str = None


@dataclass
class DomainNet_infograph:
    name: str = "domainnet"
    target_domain: str = "infograph"
    domain_name: str = "domain"
    n_classes: int = 345
    train_split: str = "train"
    val_split: str = "val"
    id_test_split = None
    ood_test_split: str = "test"
    unlabeled_train_split: str = None
    unlabeled_test_split: str = None


@dataclass
class DomainNet_painting:
    name: str = "domainnet"
    target_domain: str = "painting"
    domain_name: str = "domain"
    n_classes: int = 345
    train_split: str = "train"
    val_split: str = "val"
    id_test_split = None
    ood_test_split: str = "test"
    unlabeled_train_split: str = None
    unlabeled_test_split: str = None


@dataclass
class DomainNet_real:
    name: str = "domainnet"
    target_domain: str = "real"
    domain_name: str = "domain"
    n_classes: int = 345
    train_split: str = "train"
    val_split: str = "val"
    id_test_split = None
    ood_test_split: str = "test"
    unlabeled_train_split: str = None
    unlabeled_test_split: str = None


@dataclass
class DomainNet_sketch:
    name: str = "domainnet"
    target_domain: str = "sketch"
    domain_name: str = "domain"
    n_classes: int = 345
    train_split: str = "train"
    val_split: str = "val"
    id_test_split = None
    ood_test_split: str = "test"
    unlabeled_train_split: str = None
    unlabeled_test_split: str = None


_DATASETS = {
    "iwildcam": IwildCam,
    "camelyon17": Camelyon17,
    "rxrx1": Rxrx1,
    "fmow": FMoW,
    "poverty": PovertyMap,
    "domain_net_clipart": DomainNet_clipart,
    "domain_net_infograph": DomainNet_infograph,
    "domain_net_painting": DomainNet_painting,
    "domain_net_quickdraw": DomainNet_quickdraw,
    "domain_net_real": DomainNet_real,
    "domain_net_sketch": DomainNet_sketch,
}


class DatasetConfigurator:
    def __init__(self, dataset_name, data_path, input_resolution):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.input_resolution = input_resolution

        if dataset_name not in _DATASETS:
            raise ValueError(f"Please choose a dataset from {list(_DATASETS.keys())}")

        dataset = _DATASETS[dataset_name]
        self.metadata = dataset

        # Configuration for datasets and transformations
        self.configs = {
            "fmow": {
                "dataset": lambda: wilds.get_dataset(
                    dataset=dataset.name, download=False, root_dir=self.data_path
                ),
                "train_transform": lambda: fmow_train_transform(
                    image_size=self.input_resolution
                ),
                "val_transform": lambda: fmow_test_transform(image_size=self.input_resolution),
                "uniform_over_group": False,
                "uniform_sampler": False,
            },
            "camelyon17": {
                "dataset": lambda: wilds.get_dataset(
                    dataset=self.dataset_name, download=False, root_dir=self.data_path
                ),
                "train_transform": lambda: camelyon_train_transform(
                    image_size=self.input_resolution
                ),
                "val_transform": lambda: openclip_transform(
                    image_size=self.input_resolution, is_train=False
                ),
                "uniform_over_group": False,
                "uniform_sampler": True,
            },
            "default": {
                "dataset": lambda: wilds.get_dataset(
                    dataset=self.dataset_name, download=False, root_dir=self.data_path
                ),
                "train_transform": lambda: wilds_default_train_transform(
                    image_size=self.input_resolution
                ),
                "val_transform": lambda: openclip_transform(
                    image_size=self.input_resolution, is_train=False
                ),
                "uniform_over_group": True,
                "uniform_sampler": True,
            },
        }
        
        # Specific logic for "domainnet"
        if "domain_net" in self.dataset_name:
            self.configs[self.dataset_name] = {
                "dataset": lambda: MultiSourceDomainNetDataset(
                    download=False,
                    target_domain=dataset.target_domain,
                    root_dir=self.data_path,
                ),
                "train_transform": lambda: domainnet_train_transform(image_size=self.input_resolution),
                "val_transform": lambda: openclip_transform(image_size=self.input_resolution, is_train=False),
                "uniform_over_group": False,
                "uniform_sampler": False,
            }

    def get_dataset_configuration(self):
        # Fetch the specific configuration for the dataset or fall back to default
        config = self.configs.get(self.dataset_name, self.configs["default"])
        dataset_cfg = {
            "dataset": config["dataset"](),
            "dataset_metadata": self.metadata,
            "train_transform": config["train_transform"](),
            "val_transform": config["val_transform"](),
            "uniform_over_group": config["uniform_over_group"],
            "uniform_sampler": config["uniform_sampler"],
        }
        return dataset_cfg
