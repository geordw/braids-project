"""
Functions for dealing with buckets of braids in an efficient way.
"""

import abc
import dataclasses
import heapq
import itertools
import random
from typing import (
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import polymat
from .braid import GNF, NFBase
from .jonesrep import JonesCellRep

T = TypeVar('T')

class BraidStatBase(Generic[T]):
    name: str

    @staticmethod
    @abc.abstractmethod
    def calculate(rep: JonesCellRep, braids: Sequence[NFBase], images: npt.NDArray) -> Sequence[T]:
        """Calculate this statistic for all (braid, image) pairs."""


class Stats:
    class NFKind(BraidStatBase[str]):
        """The string 'GNF' or 'DGNF' depending on which normal form the braid is."""
        name = 'nf'

        @staticmethod
        def calculate(rep: JonesCellRep, braids: Sequence[NFBase], images: npt.NDArray) -> Sequence[str]:
            return [braid.__class__.__name__ for braid in braids]

    class CanonicalLength(BraidStatBase[int]):
        """The canonical length of the braid (Garside length for a GNF, dual Garside length for a DGNF)."""
        name = 'length'

        @staticmethod
        def calculate(rep: JonesCellRep, braids: Sequence[NFBase], images: npt.NDArray) -> Sequence[int]:
            return [braid.canonical_length() for braid in braids]

    class ProjectiveLength(BraidStatBase[int]):
        """The projlen of a matrix: how many nonzero polynomial terms there are."""
        name = 'projlen'

        @staticmethod
        def calculate(rep: JonesCellRep, braids: Sequence[NFBase], images: npt.NDArray) -> Sequence[int]:
            return polymat.projlen(images)

    class NonzeroTerms(BraidStatBase[int]):
        """The number of nonzero terms in the projectivised matrix."""
        name = 'nz_terms'

        @staticmethod
        def calculate(rep: JonesCellRep, braids: Sequence[NFBase], images: npt.NDArray) -> Sequence[int]:
            return polymat.nz_terms(images)


def batched(iterable, chunk_length: int):
    """Split an iterable into chunks of length at most chunk_length."""
    assert chunk_length >= 1

    chunk = []
    for x in iterable:
        chunk.append(x)
        if len(chunk) == chunk_length:
            yield chunk
            chunk = []

    if len(chunk) > 0:
        yield chunk


@dataclasses.dataclass
class Bucket:
    """
    A bucket is a collection of (braid, image, weight) triples of a fixed capacity.
    """

    tag: Tuple                          # Tuple identifying the bucket.
    capacity: int                       # Maximum capacity of the bucket.

    braids: List[NFBase]                # List of braids in the bucket (determines size).
    images: npt.NDArray                 # Images in the same order.
    weights: npt.NDArray[np.float64]    # Weights in the same order.

    reservoir_count: int                # Total number of items this bucket has seen.
    reservoir_weight: float             # Total weight this bucket has seen.

    # Only used for EvictMin strategy.
    heap: List[Tuple]                   # Min-heap of (weight, index) pairs.

    @classmethod
    def create(cls, tag: Tuple, capacity: int, image_shape: Tuple[int, ...], image_dtype: npt.DTypeLike):
        return cls(
            tag=tag,
            capacity=capacity,
            braids=[],
            images=np.zeros(shape=(capacity, *image_shape), dtype=image_dtype),
            weights=np.zeros(shape=(capacity,), dtype=np.float64),
            reservoir_count=0,
            reservoir_weight=0.0,
            heap=[],
        )

    def size(self) -> int:
        """The number of braids currently in the bucket."""
        return len(self.braids)

    def is_full(self) -> bool:
        return len(self.braids) == self.capacity

    def braids_images(self) -> Tuple[List[NFBase], npt.NDArray]:
        """Return two parallel lists of braids and their images."""
        return self.braids, self.images[:self.size()]

    def show_reservoir(self, weight: float):
        """Indicate to the reservoir that it has seen another element of the given weight."""
        self.reservoir_count += 1
        self.reservoir_weight += weight

    def append_braid(self, braid: NFBase, image: npt.NDArray, weight: float) -> int:
        """Append the triple to the bucket, returning the new index. Errors if the bucket is full."""
        if self.size() == self.capacity:
            raise ValueError(f"Bucket {self.tag} is full.")

        index = self.size()
        self.braids.append(braid)
        self.images[index] = image
        self.weights[index] = weight

        return index

    def replace_braid(self, index: int, braid: NFBase, image: npt.NDArray, weight: float):
        """Replace a braid at the given index."""
        if not 0 <= index < self.size():
            raise ValueError(f"Index {index} is out of bounds for bucket {self.tag} of size {self.size()}")

        self.braids[index] = braid
        self.images[index] = image
        self.weights[index] = weight



class Tracker:
    """
    A Tracker manages a search space for braids.

    Braids are lumped into groups called buckets based on the statistics passed into bucket_braidstats.
    Each bucket has a bounded size throughout the search, and reservoir sampling is used once a bucket
    fills up, so that at the end of the process each bucket should be holding a uniform sample of the
    (potentially very many) braids it was offered.

    Each bucket has to be at least as fine as the canonical length of a braid and the projlen of a braid.
    The fineness on the length is because evaluating braids of the same length in a representation is
    quick and easy. The fineness on projlen is so that buckets do not waste space: everything in each
    bucket has uniform length in the last (the degree) dimension.

    The reservoir_weight stat can affects how reservoir sampling is done, biasing the sampling so that
    heavier items are included more items. The reservoir_weight stat should take values in [0, ∞).
    """
    def __init__(
        self,
        rep: JonesCellRep,
        bucket_capacity: int,
        bucket_braidstats: Sequence[Type[BraidStatBase]],
        other_braidstats: Sequence[Type[BraidStatBase]],
        reservoir_method: Literal['GreedyFirst', 'Uniform', 'AChao', 'EvictMin'] = 'Uniform',
        reservoir_weight: Optional[Type[BraidStatBase[float]]] = None,
        rand: random.Random = None,
    ):
        self.rep = rep
        self.bucket_capacity = bucket_capacity

        # Bucket keys should at least be as fine as 'length' and 'projlen'
        assert Stats.CanonicalLength in bucket_braidstats
        assert Stats.ProjectiveLength in bucket_braidstats
        self.bucket_braidstats = list(bucket_braidstats)
        self.other_braidstats = list(other_braidstats)

        # Bucket store
        self.buckets: Dict[Tuple, Bucket] = {}

        # BraidStat used for calculating weights for the reservoir.
        self.reservoir_weight_braidstat = reservoir_weight

        # Method used for reservoir sampling.
        self.reservoir_method = reservoir_method

        # If rand is not None, then reservoir sampling will be used for the buckets.
        self.rand = rand if rand is not None else random.Random()

        # For advancing forward, we should save the braids and images which are suffixes of each normal form entry.
        self.suffix_cache: Dict[Tuple, Tuple[List[NFBase], npt.NDArray]] = {}


    def discard_bucket(self, tag: Tuple):
        """Forget all the data in a bucket."""
        if tag in self.buckets:
            del self.buckets[tag]


    def braid_image_stats(self, braids: List[NFBase], images: npt.NDArray) -> pd.DataFrame:
        assert len(images.shape) == 4 and images.shape[0] == len(braids)
        return pd.DataFrame.from_dict(dict(
            braid=braids,
            **{
                stat.name: stat.calculate(self.rep, braids, images)
                for stat in self.bucket_braidstats + self.other_braidstats
            },
        ))


    def bucket_stats(self, tag: Tuple):
        if tag not in self.buckets:
            raise ValueError(f"Bucket {tag} not found.")

        return self.braid_image_stats(*self.bucket_braids_images(tag))


    def bucket_braids_images(self, tag: Tuple) -> Tuple[List[NFBase], npt.NDArray]:
        return self.buckets[tag].braids_images()


    def stats(self) -> pd.DataFrame:
        """Retrieve a Dataframe of statistics on the buckets."""
        df = pd.DataFrame(
            columns=[
                'bucket',
                'count',
                *[stat.name for stat in self.bucket_braidstats],
                'reservoir_count',
                'reservoir_weight_avg',
                'reservoir_weight_avg_seen',
            ],
            data=[
                (
                    bucket.tag,
                    bucket.size(),
                    *tag,
                    bucket.reservoir_count,
                    bucket.weights.sum() / bucket.size(),
                    bucket.reservoir_weight / bucket.reservoir_count,
                )
                for tag, bucket in self.buckets.items()
            ],
        )
        return df


    def valiate_buckets(self):
        """Ensure that the braids in each bucket evaluate to the matrices stored there."""
        for tag, bucket in self.buckets.items():
            braids, images = bucket.braids_images()
            actual = self.rep.polymat_evaluate_braids_of_same_length(braids)
            if not np.all(images == actual):
                print(f"A braid image is wrong in bucket {tag}")
                for i in range(len(braids)):
                    if not np.all(images[i] == actual[i]):
                        print(f"Braid index {i}, {braids[i]}")
                        print("Actual representation value:")
                        print(actual[i])
                        print("What we have:")
                        print(images[i])
                        return braids[i], actual[i], images[i]

        return "All bucket entries are sound."


    def add_braids_images(self, braids: List[NFBase], images: npt.NDArray):
        """
        Add a collection of (braid, image) pairs to the tracker.
        """
        assert len(images.shape) == 4 and len(braids) == images.shape[0]
        stats = [stat.calculate(self.rep, braids, images) for stat in self.bucket_braidstats]
        tags = [tuple(stat[i] for stat in stats) for i in range(len(braids))]
        weights: Sequence[float] = self.reservoir_weight_braidstat.calculate(self.rep, braids, images) if self.reservoir_weight_braidstat is not None else [1.0]*len(braids)

        # Calling polymat.projectivise on each braid as we find it will take a while. We can do something
        # a bit more manual: save the bottom and top degrees here, and then chop them up as they're placed
        # into the buckets.
        starts, ends = polymat.proj_starts_ends(images)

        for i in range(len(braids)):
            # Get a (braid, projectivised image) pair.
            braid = braids[i]
            image: npt.NDArray = images[i][..., starts[i]:ends[i]]
            tag = tags[i]
            weight = float(weights[i])

            # Initialise a new bucket if we have not seen it before.
            if tag not in self.buckets:
                self.buckets[tag] = Bucket.create(tag, self.bucket_capacity, image.shape, image.dtype)

            bucket = self.buckets[tag]

            # Update reservoir stats
            bucket.show_reservoir(weight)

            # Use a sampling method to decide how to add the braid.
            match self.reservoir_method:
                case 'GreedyFirst':
                    if not bucket.is_full():
                        bucket.append_braid(braid, image, weight)

                case 'Uniform':
                    if not bucket.is_full():
                        bucket.append_braid(braid, image, weight)
                    elif self.rand.random() <= bucket.capacity / bucket.reservoir_count:
                        replace = self.rand.randrange(0, bucket.capacity)
                        bucket.replace_braid(replace, braid, image, weight)

                case 'AChao':
                    if not bucket.is_full():
                        bucket.append_braid(braid, image, weight)
                    else:
                        if self.rand.random() <= weight / bucket.reservoir_weight:
                            replace = self.rand.randrange(0, bucket.capacity)
                            bucket.replace_braid(replace, braid, image, weight)

                case 'EvictMin':
                    if not bucket.is_full():
                        index = bucket.append_braid(braid, image, weight)
                        heapq.heappush(bucket.heap, (weight, index))
                    else:
                        min_weight, min_idx = bucket.heap[0]
                        if weight > min_weight:
                            heapq.heappop(bucket.heap)
                            heapq.heappush(bucket.heap, (weight, min_idx))
                            bucket.replace_braid(min_idx, braid, image, weight)


    def nf_descendants(self, tag: Tuple, length: int = 1):
        """Try exhaustively looking at the normal form descendants of all elements in a bucket up to the given length."""
        # We evaluate tons of stuff here. We also don't need to keep the partial products, we should just jump forward
        # by the given length? (No, we should allow partial products since it gives some interaction between a group
        # which is going well and an independent group which is going poorly).
        #
        # It would be best to first sort the braids by their last entry. This fully determines the set of suffixes which
        # can be used for each braid. Then we can evaluate the suffix once, and multiply it onto all braids in one step,
        # rather than re-evaluating huge amounts of suffixes every time.
        #
        # If we put some logic in here about the automaton, this could be quite fast.

        # Let's sort this bucket first, so that elements ending in the same normal form factor appear next to each other.
        bucket_braids, bucket_images = self.buckets[tag].braids_images()
        last_factors = np.array([braid.factors[-1] for braid in bucket_braids])
        sorted_indices = np.argsort(last_factors)

        sorted_braids = [bucket_braids[i] for i in sorted_indices]
        sorted_images = bucket_images[sorted_indices]

        # Now we can iterate through each group ending in the same factor.
        for last, indices in itertools.groupby(range(len(sorted_braids)), key=lambda i: sorted_braids[i].factors[-1]):
            index_list = list(indices)
            chunk = slice(min(index_list), max(index_list)+1)
            braids = sorted_braids[chunk]
            left = sorted_images[chunk]

            # Cache suffixes and their images depending on the last factor.
            last_factor = braids[0].canonical_factor(-1)
            if (length, last_factor) not in self.suffix_cache:
                suffixes = list(braids[0].nf_suffixes(length))
                rights = self.rep.polymat_evaluate_braids_of_same_length(suffixes)
                self.suffix_cache[length, last_factor] = (suffixes, rights)
            else:
                suffixes, rights = self.suffix_cache[length, last_factor]

            images = self.rep.polymat_mul(left[None, ...], rights[:, None, ...])
            images = images.reshape((-1, *images.shape[2:]))
            self.add_braids_images([braid * suffix for suffix in suffixes for braid in braids], images)



    def bootstrap_exhaustive(self, cls: Type[NFBase], upto_length: int):
        """
        Iterate over all braids up to the given Garside length to bootstrap the buckets.
        """
        for length in range(1, upto_length + 1):
            for braids in batched(cls.all_of_length(self.rep.n, length), 10_000):
                images = self.rep.polymat_evaluate_braids_of_same_length(braids)
                self.add_braids_images(braids, images)


    def hash_bucket(self, bucket: Tuple) -> Dict[int, List[GNF]]:
        """Return a list of the GNFs in the bucket, by the hash of their representation matrices."""
        assert bucket in self.buckets

        braids, images = self.bucket_braids_images(bucket)
        hashes = {}
        for i in range(len(braids)):
            h = hash(images[i].data.tobytes())
            if h not in hashes:
                hashes[h] = []

            hashes[h] += [braids[i]]

        return hashes
