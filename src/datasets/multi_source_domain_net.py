import csv
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import pandas as pd
from PIL import Image
import copy 

from wilds.common.utils import map_to_id_array
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.datasets.domainnet_dataset import SENTRY_DOMAINS, DOMAIN_NET_DOMAINS, DOMAIN_NET_CATEGORIES


class MultiSourceDomainNetDataset(WILDSDataset):
    _dataset_name: str = "domainnet"
    _versions_dict: Dict[str, Dict[str, Union[str, int]]] = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x0b8ca76eef384b98b879d0c8c4681a32/contents/blob/",
            "compressed_size": 19_255_770_459,
        },
    }

    def __init__(
        self,
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official",
        target_domain: str = "real",
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (224, 224)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = self.initialize_data_dir(root_dir, download)

        # The original dataset contains 345 categories. The SENTRY version contains 40 categories.
        # if use_sentry:
        #     assert source_domain in SENTRY_DOMAINS
        #     assert target_domain in SENTRY_DOMAINS
        #     print("Using the SENTRY version of DomainNet...")
        #     metadata_filename = "sentry_metadata.csv"
        #     self._n_classes = 40
        # else:
        #     metadata_filename = "metadata.csv"
        #     self._n_classes = 345
        metadata_filename = "metadata.csv"
        self._n_classes = 345

        metadata_df: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_dir, metadata_filename),
            dtype={
                "image_path": str,
                "domain": str,
                "split": str,
                "category": str,
                "y": int,
            },
            keep_default_na=False,
            na_values=[],
            quoting=csv.QUOTE_NONNUMERIC,
        )
        multi_source_domains = copy.copy(DOMAIN_NET_DOMAINS) 
        multi_source_domains.remove(target_domain)
        # multi_source_domains.remove('quickdraw')
        # multi_source_domains.remove('painting')
        # multi_source_domains.remove('sketch')
        # multi_source_domains.remove('real')

        source_metadata_df = metadata_df.loc[metadata_df["domain"].isin(multi_source_domains)]
        source_metadata_df["split"] = "train"
        target_metadata_df = metadata_df.loc[metadata_df["domain"] == target_domain]
        # target_metadata_df["split"] = "test"
        metadata_df = pd.concat([source_metadata_df, target_metadata_df])

        self._input_image_paths = metadata_df["image_path"].values
        self._y_array = torch.from_numpy(metadata_df["y"].values).type(torch.LongTensor)
        self.initialize_split_dicts()
        self.initialize_split_array(metadata_df, multi_source_domains, target_domain)

        # Populate metadata fields
        self._metadata_fields = ["domain", "category", "y"]
        metadata_df = metadata_df[self._metadata_fields]
        possible_metadata_values = {
            "domain": DOMAIN_NET_DOMAINS,
            "category": DOMAIN_NET_CATEGORIES,
            "y": range(self._n_classes),
        }
        self._metadata_map, metadata = map_to_id_array(
            metadata_df, possible_metadata_values
        )
        self._metadata_array = torch.from_numpy(metadata.astype("long"))

        # Eval
        self.initialize_eval_grouper()
        super().__init__(root_dir, download, self._split_scheme)

    def get_input(self, idx) -> str:
        img_path = os.path.join(self.data_dir, self._input_image_paths[idx])
        img = Image.open(img_path).convert("RGB")
        return img

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric, self._eval_grouper, y_pred, y_true, metadata
        )

    def initialize_split_dicts(self):
        if self.split_scheme == "official":
            self._split_dict: Dict[str, int] = {
                "train": 0,
                "val": 1,
                "test": 2,
                "id_test": 3,
            }
            self._split_names: Dict[str, str] = {
                "train": "Train",
                "val": "Validation (OOD)",
                "test": "Test (OOD)",
                "id_test": "Test (ID)",
            }
            self._source_domain_splits = [0, 3]
        else:
            raise ValueError(f"Split scheme {self.split_scheme} is not recognized.")

    def initialize_split_array(self, metadata_df, multi_source_domains, target_domain):
        def get_split(row):
            if row["domain"] in multi_source_domains:
                if row["split"] == "train":
                    return 0
                elif row["split"] == "test":
                    return 3
            elif row["domain"] == target_domain:
                if row["split"] == "train":
                    return 1
                elif row["split"] == "test":
                    return 2
            else:
                raise ValueError(
                    f"Domain should be one of {multi_source_domains}, {target_domain}"
                )

        self._split_array = metadata_df.apply(
            lambda row: get_split(row), axis=1
        ).to_numpy()

    def initialize_eval_grouper(self):
        if self.split_scheme == "official":
            self._eval_grouper = CombinatorialGrouper(
                dataset=self, groupby_fields=["category"]
            )
        else:
            raise ValueError(f"Split scheme {self.split_scheme} not recognized.")
