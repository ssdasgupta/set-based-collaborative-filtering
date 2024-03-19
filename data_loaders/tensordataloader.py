# Copyright 2023 The pytorch-utils Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The primary contribution of this module is TensorDataLoader, a DataLoader optimized for performance when your Dataset
object is able to return multiple elements at once by passing a LongTensor to the __getitem__ method.

For example, TensorDataset objects support access such as a[[1,24,6,76,3,2]], however traditional DataLoader objects
will actually create a batch by performing collate_fn([a[1], a[24], a[6], a[76], a[3], a[2]]) which can be much slower.

To address this, we create TensorBatchSampler which grabs a batch of indices at a time, returning [a[[1,24,6,76,3,2]]]
and a dummy collate_fn which simply unwraps the outer list. This is all handled appropriately by using a
TensorDataLoader in place of a normal DataLoader.
"""

import math

import torch
from torch.utils.data import Sampler, DataLoader

__all__ = [
    "TensorBatchSampler",
    "TensorDataLoader",
]


class TensorBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=False, drop_last=False):
        # We don't need to check for this, other datasets might support this type of
        # list-based indexing.
        # if not isinstance(data_source, TensorDataset):
        #     raise ValueError(
        #         f"data_source should be an instance of torch.utils.data.TensorDataset, but got data_source={data_source}"
        #     )
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(shuffle, bool):
            raise ValueError(
                f"shuffle should be a boolean value, but got shuffle={shuffle}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle:
            self.idxs = torch.randperm(len(self.data_source))
        else:
            self.idxs = torch.arange(len(self.data_source))
        self.current_idx = 0
        self.next_idx = self.batch_size
        return self

    def __next__(self):
        out = self.idxs[self.current_idx : self.next_idx]
        out_of_data = self.current_idx >= len(self.data_source)
        not_full_batch = self.next_idx > len(self.data_source)
        if out_of_data or (not_full_batch and self.drop_last):
            del self.idxs, self.current_idx, self.next_idx
            raise StopIteration
        else:
            self.current_idx = self.next_idx
            self.next_idx += self.batch_size
            return out

    def __len__(self):
        return (math.floor if self.drop_last else math.ceil)(
            len(self.data_source) / self.batch_size
        )


def _unwrap_collate_fn(batch):
    return batch[0]


class TensorDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        *,
        drop_last=False,
        collate_fn=None,
        **kwargs,
    ):
        if sampler is not None or batch_sampler is not None or collate_fn is not None:
            raise ValueError(
                "TensorDataLoader does not support alternate samplers, batch samplers, or collate functions."
            )
        sampler = TensorBatchSampler(dataset, batch_size, shuffle, drop_last)
        super().__init__(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            collate_fn=_unwrap_collate_fn,
            **kwargs,
        )
