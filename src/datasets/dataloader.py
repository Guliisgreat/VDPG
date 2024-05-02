import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.utils import get_counts, split_into_groups
import random

def get_contrastive_train_loader(
    dataset,
    batch_size,
    uniform_over_groups=None,
    grouper=None,
    distinct_groups=True,
    n_negative_groups_per_batch=None,
    n_points_per_negative_group=None,
    uniform_sampler=True,
    **loader_kwargs,
):
    """
    Constructs and returns the data loader for contrastive training for TTA.
    Args:
        - dataset (WILDSDataset or WILDSSubset): Data
        - batch_size (int): Batch size
        - uniform_over_groups (None or bool): Whether to sample the groups uniformly or according
                                              to the natural data distribution.
                                              Setting to None applies the defaults for each type of loaders.
                                              For standard loaders, the default is False. For group loaders,
                                              the default is True.
        - grouper (Grouper): Grouper used for group loaders or for uniform_over_groups=True
        - distinct_groups (bool): Whether to sample distinct_groups within each minibatch for group loaders.
        - n_groups_per_batch (int): Number of groups to sample in each minibatch for group loaders.
        - loader_kwargs: kwargs passed into torch DataLoader initialization.
    Output:
        - data loader (DataLoader): Data loader.
    """

    if uniform_over_groups is None:
        uniform_over_groups = True


    assert grouper is not None
    assert n_negative_groups_per_batch is not None
    if n_negative_groups_per_batch + 1 > grouper.n_groups:
        raise ValueError(
            f"n_groups_per_batch was set to {n_negative_groups_per_batch + 1} \
                         but there are only {grouper.n_groups} groups specified."
        )

    group_ids = grouper.metadata_to_group(dataset.metadata_array)
    if uniform_sampler:
        batch_sampler = ConstrastiveSamplerUniform(
            group_ids=group_ids,
            batch_size=batch_size,
            n_negative_groups_per_batch=n_negative_groups_per_batch,
            n_points_per_negative_group=n_points_per_negative_group,
            uniform_over_groups=uniform_over_groups,
            distinct_groups=distinct_groups,
        )
    else:
        batch_sampler = ConstrastiveSampler(
            group_ids=group_ids,
            batch_size=batch_size,
            n_negative_groups_per_batch=n_negative_groups_per_batch,
            n_points_per_negative_group=n_points_per_negative_group,
            uniform_over_groups=uniform_over_groups,
            distinct_groups=distinct_groups,
        )

    return DataLoader(
        dataset,
        shuffle=None,
        sampler=None,
        collate_fn=dataset.collate,
        batch_sampler=batch_sampler,
        drop_last=False,
        **loader_kwargs,
    )


class ConstrastiveSamplerUniform:
    """
    Constructs batches for contrastive learning. Each batch has one positive group and a few of negative group.
    Sample groups first and then sample data points for each group.
    """

    def __init__(
        self,
        group_ids,
        batch_size,
        n_negative_groups_per_batch,
        n_points_per_negative_group,
        uniform_over_groups,
        distinct_groups,
    ):
        if len(group_ids) < batch_size:
            raise ValueError(
                f"The dataset has only {len(group_ids)} examples but the batch size is {batch_size}. There must be enough examples to form at least one complete batch."
            )

        self.group_ids = group_ids
        self.unique_groups, self.group_indices, unique_counts = split_into_groups(
            group_ids
        )

        self.distinct_groups = distinct_groups
        self.dataset_size = len(group_ids)
        self.num_batches = self.dataset_size // batch_size

        if uniform_over_groups:  # Sample uniformly over groups
            self.group_prob = None
        else:  # Sample a group proportionately to its size
            self.group_prob = unique_counts.numpy() / unique_counts.numpy().sum()

        self.n_points_per_negative_group = n_points_per_negative_group
        self.n_negative_groups_per_batch = n_negative_groups_per_batch
        self.n_groups_per_batch = self.n_negative_groups_per_batch + 1
        self.n_points_per_positive_group = batch_size - (
            self.n_negative_groups_per_batch * self.n_points_per_negative_group
        )
        self.n_points_per_positive_group = batch_size

        if self.n_points_per_positive_group <= 4:
            raise ValueError(
                f"The number of data examples sampled from the positive group (support + query) must more than 4."
            )
        # for circle sampling
        self.idx = list(range(len(self.unique_groups)))

    def __iter__(self):
        for batch_id in range(self.num_batches):
            # Note that we are selecting group indices rather than groups

            # random sampling
            groups_for_batch = np.random.choice(
                len(self.unique_groups),
                size=self.n_groups_per_batch,
                replace=(not self.distinct_groups),
                p=self.group_prob,
            )

            sampled_ids = []
            for idx, group in enumerate(groups_for_batch):
                if idx == 0:
                    sampled_positive_ids = np.random.choice(
                        self.group_indices[group],
                        size=self.n_points_per_positive_group,
                        replace=len(self.group_indices[group])
                        <= self.n_points_per_positive_group,  # False if the group is larger than the sample size
                        p=None,
                    )
                    sampled_ids += sampled_positive_ids.tolist()
                else:
                    sampled_negative_ids = np.random.choice(
                        self.group_indices[group],
                        size=self.n_points_per_negative_group,
                        replace=len(self.group_indices[group])
                        <= self.n_points_per_negative_group,  # False if the group is larger than the sample size
                        p=None,
                    )
                    sampled_ids += sampled_negative_ids.tolist()

            sampled_ids = np.array(sampled_ids)
            yield sampled_ids

    def __len__(self):
        return self.num_batches

class ConstrastiveSampler:
    """
    Constructs batches for contrastive learning. Each batch has one positive group and a few of negative group.
    Sample groups first and then sample data points for each group.
    """

    def __init__(
        self,
        group_ids,
        batch_size,
        n_negative_groups_per_batch,
        n_points_per_negative_group,
        uniform_over_groups,
        distinct_groups,
    ):
        if len(group_ids) < batch_size:
            raise ValueError(
                f"The dataset has only {len(group_ids)} examples but the batch size is {batch_size}. There must be enough examples to form at least one complete batch."
            )

        self.group_ids = group_ids
        self.unique_groups, self.group_indices, unique_counts = split_into_groups(
            group_ids
        )

        self.distinct_groups = distinct_groups
        self.dataset_size = len(group_ids)
        self.num_batches = self.dataset_size // batch_size

        if uniform_over_groups:  # Sample uniformly over groups
            self.group_prob = None
        else:  # Sample a group proportionately to its size
            self.group_prob = unique_counts.numpy() / unique_counts.numpy().sum()

        self.n_points_per_negative_group = n_points_per_negative_group
        self.n_negative_groups_per_batch = n_negative_groups_per_batch
        self.n_groups_per_batch = self.n_negative_groups_per_batch + 1

        # self.n_points_per_positive_group = batch_size - (
        #     self.n_negative_groups_per_batch * self.n_points_per_negative_group
        # )

        self.n_points_per_positive_group = batch_size

        if self.n_points_per_positive_group <= 4:
            raise ValueError(
                f"The number of data examples sampled from the positive group (support + query) must more than 4."
            )

        print('check group index :::::', len(self.group_indices), self.unique_groups)
        print('Replacement: ', self.distinct_groups)
        total_idx = []

        for idx in range(len(unique_counts)):
            print(self.group_prob[idx], unique_counts[idx])


        for idx, domain_ID in enumerate(self.unique_groups):
            print('Group ID: {} has {} examples, {}, prob_computed: {}'.format(idx, len(self.group_indices[idx]), unique_counts[idx], self.group_prob[idx]))
            domain_idx = self.group_indices[idx]
            total_idx += domain_idx

        self.check_idx = [0] * len(self.unique_groups)
        self.domain_idx_acc = [[]] * len(self.unique_groups)

    def __iter__(self):
        for batch_id in range(self.num_batches):
            # circle sampling shuffle

            if batch_id == 0:
                self.domain_idx_acc = [[]] * len(self.unique_groups)
                for idx, item in enumerate(self.group_indices):
                    indexes = torch.randperm(item.shape[0])
                    self.group_indices[idx] = item[indexes]


            # uniform sampling:

            groups_for_batch = np.random.choice(
                len(self.unique_groups),
                size=self.n_groups_per_batch,
                replace=(not self.distinct_groups),
                # replace=True,
                p=self.group_prob,
            )

            sampled_ids = []
            for idx, group in enumerate(groups_for_batch):
                if idx == 0:

                    # circle sampling
                    current_group_idx = self.group_indices[group]
                    sampled_positive_ids = current_group_idx[:self.n_points_per_positive_group]

                    current_group_idx = torch.cat((current_group_idx[self.n_points_per_positive_group:],
                                                  current_group_idx[:self.n_points_per_positive_group]))
                    self.group_indices[group] = current_group_idx

                    sampled_ids += sampled_positive_ids.tolist()

                    self.domain_idx_acc[group] = self.domain_idx_acc[group] + sampled_positive_ids.tolist()

                    self.check_idx[group] += 1


                else:
                    sampled_negative_ids = np.random.choice(
                        self.group_indices[group],
                        size=self.n_points_per_negative_group,
                        replace=len(self.group_indices[group])
                        <= self.n_points_per_negative_group,  # False if the group is larger than the sample size
                        p=None,
                    )
                    sampled_ids += sampled_negative_ids.tolist()

            sampled_ids = np.array(sampled_ids)
            yield sampled_ids

    def __len__(self):
        return self.num_batches
