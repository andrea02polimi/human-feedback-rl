"""
Batch iterator utility.
"""

from typing import List, Iterator
import numpy as np


def batch_iter(data: list, batch_size: int, shuffle: bool = True) -> Iterator[list]:
    """
    Yield lists of items from data in batches of batch_size.

    Args:
        data:       list of items to iterate over
        batch_size: number of items per batch
        shuffle:    if True, shuffle indices before batching
    """
    idxs = list(range(len(data)))

    if shuffle:
        np.random.shuffle(idxs)

    start = 0
    while start < len(data):
        end = min(start + batch_size, len(data))
        batch_idxs = idxs[start:end]
        yield [data[i] for i in batch_idxs]
        start += batch_size
