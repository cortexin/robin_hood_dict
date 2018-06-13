from collections.abc import MutableMapping
from itertools import chain, zip_longest, repeat

from typing import Any, Hashable, Union, NamedTuple, List, Tuple, Optional, Iterable, Iterator, TypeVar


K = 2 / 3


class RobinValue(NamedTuple):
    hash: int
    key: Hashable
    value: Any


Bucket = Optional[RobinValue]
EMPTY = None


class BucketList(List[Bucket]):
    def __init__(self, itable: Iterable, n_full: int = 0) -> None:
        self.n_full = n_full
        self.deleted = set()
        super().__init__(itable)

    def __setitem__(self, idx: int, value: Any) -> None:
        old_value = super().__getitem__(idx)
        super().__setitem__(idx, value)

        delta = (-1 if old_value else 1) + (1 if value else -1)
        self.n_full += delta
        self.deleted.discard(idx)

    def __delitem__(self, idx: int) -> None:
        old_value = super().__getitem__(idx)

        if old_value:
            self.n_full -= 1

        super().__setitem__(idx, None)
        self.deleted.add(idx)


class RobinHoodDict(MutableMapping):
    buckets: BucketList
    mean_dist: int
    max_dist: int

    def __init__(self, base: Optional[dict] = None,  **kwargs: dict) -> None:
        n_buckets = max(int(len(kwargs) // K), 10)
        self.buckets = BucketList(repeat(EMPTY, n_buckets))
        self.mean_dist = 1  # mean distance from the original bucket
        self.max_dist = 1

        for k, v in (base or kwargs).items():
            self.__setitem__(k, v)

    def __setitem__(self, key: Hashable, val: Any) -> None:
        h = self._compute_hash(key)
        idx, bucket = self._find_bucket(h)
        print(f'{key}. {val} => {idx}...{bucket}')

        self.buckets[idx] = RobinValue(h, key, val)

        distance = idx - h
        self._add_to_mean(distance)
        self.max_dist = max(self.max_dist, distance)

        if isinstance(bucket, RobinValue):
            self.__setitem__(bucket.key, bucket.value)

    def __getitem__(self, key: Hashable) -> Any:
        h = self._compute_hash(key)

        print(self._get_smart_search_indexes(h))
        for i in self._get_smart_search_indexes(h):
            bucket = self.buckets[i]
            print(bucket)

            if bucket is EMPTY:
                raise KeyError

            elif bucket.hash == h:
                if self._compare_keys(bucket.key, key):
                    return bucket.value

        raise KeyError

    def __delitem__(self, key: Hashable) -> None:
        h = self._compute_hash(key)

        for i in self._get_smart_search_indexes(h):
            bucket = self.buckets[i]

            if bucket is EMPTY:
                raise KeyError

            if bucket.hash == h and self._compare_keys(bucket.key, key):
                del self.buckets[i]
                distance = i - h
                self._remove_from_mean(distance)
                # TODO: decrease max_dist
                break
        else:
            raise KeyError

    def __len__(self) -> int:
        return self.buckets.n_full

    def __iter__(self) -> Iterator[Hashable]:
        for bucket in self.buckets:
            if isinstance(bucket, RobinValue):
                yield bucket.key

    def _find_bucket(self, h: int) -> Tuple[int, Bucket]:
        for i, bucket in enumerate(self.buckets[h:]):
            if not bucket or bucket.hash < h:
                return h+i, bucket

    def _compute_hash(self, key: Hashable) -> int:
        return hash(key) % len(self.buckets)

    def _compare_keys(self, key1, key2) -> bool:
        return (key1 is key2) or (key1 == key2)

    def _remove_from_mean(self, distance: int) -> None:
        self.mean_dist = int((self.mean_dist * self.buckets.n_full -
                              distance) // (self.buckets.n_full - 1))

    def _add_to_mean(self, distance: int) -> None:
        self.mean_dist = int(
            (self.mean_dist + distance / self.buckets.n_full) / 2)

    def _get_smart_search_indexes(self, h: int) -> Iterable:
        # could be an iterator
        indexes = zip_longest(range(self.mean_dist + h, self.max_dist + h + 1),
                              range(self.mean_dist + h - 1, h-1, -1),
                              fillvalue=0)

        return (i for i in chain(*indexes) if i >= h and i not in self.buckets.deleted)
