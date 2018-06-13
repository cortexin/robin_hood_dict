from collections.abc import MutableMapping
from itertools import chain, zip_longest, repeat
from typing import Any, Hashable, Union, NamedTuple, List, Tuple, Optional, Iterable, Iterator, TypeVar


MAX_LOAD = 2 / 3
MIN_LOAD = 1 / 2
AVG_LOAD = (MAX_LOAD + MIN_LOAD) / 2


class RobinValue(NamedTuple):
    hash: int
    key: Hashable
    value: Any


Bucket = Optional[RobinValue]


class BucketList(List[Bucket]):
    def __init__(self, itable: Iterable, n_full: int = 0) -> None:
        self.n_full = n_full
        self.tombstones: set = set()
        super().__init__(itable)

    def __setitem__(self, idx, value):
        old_value = super().__getitem__(idx)
        super().__setitem__(idx, value)

        delta = (-1 if old_value else 1) + (1 if value else -1)
        self.n_full += delta
        self.tombstones.discard(idx)

    def __delitem__(self, idx):
        old_value = super().__getitem__(idx)

        if old_value:
            self.n_full -= 1

        super().__setitem__(idx, None)
        self.tombstones.add(idx)


class RobinHoodDict(MutableMapping):
    buckets: BucketList
    mean_dist: int
    max_dist: int

    def __init__(self, base: Optional[dict] = None,  **_kwargs: dict) -> None:
        kwargs = base or _kwargs
        n_buckets = max(int(len(kwargs) / AVG_LOAD), 10)
        print(n_buckets, len(kwargs))
        self.buckets = BucketList(repeat(None, n_buckets))
        self.mean_dist = 1  # mean distance from the original bucket
        self.max_dist = 1

        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def __setitem__(self, key: Hashable, val: Any) -> None:
        h = self._compute_hash(key)
        idx, bucket = self._find_bucket(h)
        print(f'{key} => {val} == {bucket} @ {idx}')

        self.buckets[idx] = RobinValue(h, key, val)

        self._update_statistics_add(idx, h)
        if bucket:
            self.__setitem__(bucket.key, bucket.value)

        elif self.load_factor >= MAX_LOAD:
            self._rehash()

    def __getitem__(self, key: Hashable) -> Any:
        h = self._compute_hash(key)

        print(self._get_smart_search_indexes(h))
        for i in self._get_smart_search_indexes(h):
            bucket = self.buckets[i]

            if bucket is None:
                raise KeyError

            elif bucket.hash == h:
                if self._compare_keys(bucket.key, key):
                    return bucket.value

        raise KeyError

    def __delitem__(self, key: Hashable) -> None:
        h = self._compute_hash(key)

        for idx in self._get_smart_search_indexes(h):
            bucket = self.buckets[idx]

            if bucket is None:
                raise KeyError

            if bucket.hash == h and self._compare_keys(bucket.key, key):
                del self.buckets[idx]
                self._update_statistics_remove(idx, h)

                if self.load_factor <= MIN_LOAD / 2:
                    self._rehash
                break
        else:
            raise KeyError

    def __len__(self) -> int:
        return self.buckets.n_full

    def __iter__(self) -> Iterator[Hashable]:
        for bucket in filter(None, self.buckets):
            yield bucket.key

    def _find_bucket(self, h: int) -> Tuple[int, Bucket]:
        for i, bucket in enumerate(self.buckets[h:]):
            if not bucket or bucket.hash < h:
                return h+i, bucket

    def _compute_hash(self, key: Hashable) -> int:
        return hash(key) % len(self.buckets)

    def _compare_keys(self, key1, key2) -> bool:
        return (key1 is key2) or (key1 == key2)

    def _update_statistics_add(self, idx: int, orig_idx: int) -> None:
        distance = idx - orig_idx

        self.mean_dist = int(
            (self.mean_dist + distance / self.buckets.n_full) / 2)
        self.max_dist = max(self.max_dist, distance)

    def _update_statistics_remove(self, idx: int, orig_idx: int) -> None:
        distance = idx - orig_idx

        self.mean_dist = int((self.mean_dist * self.buckets.n_full -
                              distance) // (self.buckets.n_full - 1))

        if self.max_dist == distance:
            self.max_dist = self._get_max_dist()

    def _get_smart_search_indexes(self, h: int) -> Iterator[int]:
        indexes = zip_longest(range(self.mean_dist + h, self.max_dist + h + 1),
                              range(self.mean_dist + h - 1, h-1, -1),
                              fillvalue=0)

        for idx in indexes:
            if idx >= h and idx not in self.buckets.tombstones:
                yield idx

    def _get_max_dist(self):
        dist = 0
        for idx, bucket in enumerate(self.buckets):
            if bucket:
                dist = max(dist, idx - bucket.hash)

        return dist

    def _rehash(self):
        print('Called rehash.')
        old_buckets = self.buckets

        n_buckets = int(self.buckets.n_full / MIN_LOAD)
        self.buckets = BucketList(repeat(None, n_buckets))

        for _, key, value in filter(None, old_buckets):
            self.__setitem__(key, value)

    @property
    def load_factor(self) -> int:
        return self.buckets.n_full / len(self.buckets)
