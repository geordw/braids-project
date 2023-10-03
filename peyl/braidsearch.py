"""
Convenience functions for searching for braids with low rhogap/length.

Second version: we do everything projectively, and allow modulo p, with p = 0 meaning no mod.
When working mod p we also use 32-bit integers, so the prime chosen should be at most 2^13 or so.
"""

import collections
import dataclasses
import functools
import math
import random
import typing
from typing import (
    Dict,
    List,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import polymat, tl
from .braid import GNF, BraidGroup
from .lpoly import LPoly
from .matrix import Matrix
from .permutations import SymmetricGroup


def tl_gens_invs(n):
    """Generators for the braid group B_n inside TL(n, n)."""
    v = LPoly(0, (0, 1))
    es = [tl.TLMor.e(i, n) for i in range(n - 1)]
    gens = [-v*(e - v**-1) for e in es]
    invs = [-v**-1 * (e - v) for e in es]
    assert all(x*y == tl.TLMor.id(n) for x, y in zip(gens, invs))
    return gens, invs


def cell_matrix(tlmor, r):
    """
    Given a TLMor of type (n, n), return the matrix of its
    action on the cell module with n - 2r strands.
    """
    assert tlmor.top == tlmor.bot

    basis = tl.cell_basis(tlmor.bot, r)
    columns = [
        [prod.quotient(tlmor.top - 2*r).coefficient(b) for b in basis]
        for prod in [tlmor * b for b in basis]
    ]
    return Matrix.from_rows(tuple(zip(*columns)))

def cell_matrices(n, r):
    """Return matrices for the Artin generators and their inverses in (n-r, r)."""
    tlgens, tlinvs = tl_gens_invs(n)
    return [cell_matrix(g, r) for g in tlgens], [cell_matrix(g, r) for g in tlinvs]


@dataclasses.dataclass(frozen=True)
class JonesSummand:
    """
    Specifies a representation of a braid group, along with whether that representation should be mod p,
    or projective with respect to powers of v.
    """
    n: int             # Number of strands of the braid group.
    r: int             # Partition labelling the representation is (n - r, r), for n - 2r ≥ 0.
    p: int = 0         # Zero (coefficients are int64) or a prime (coefficients are int32 mod p).

    def __post_init__(self):
        assert self.n >= 1
        assert self.n - 2 * self.r >= 0
        assert self.p >= 0

    def __repr__(self):
        ring = 'ℤ' if self.p == 0 else f'F{self.p}'
        return f'Two-rowed representation ({self.n - self.r}, {self.r}) of B{self.n}, over {ring}'

    def dtype(self) -> npt.DTypeLike:
        """The dtype which should be used for elements of this representation."""
        return np.int32 if self.p > 0 else np.int64

    def dimension(self) -> int:
        """The dimension of this representation."""
        return math.comb(self.n, self.r) - math.comb(self.n, self.r - 1)

    @functools.cache
    def id(self):
        """The identity matrix for this representation."""
        return np.identity(self.dimension(), dtype=self.dtype())[..., None]

    @functools.cache
    def artin_gens_invs(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Returns two NDArrays of shape (n-1, dim, dim, ?) where dim is the dimension of the representation.
        The first NDArray are the images of the Artin generators in the representation, while the second NDArray
        are the images of their inverses.
        """
        gens, invs = cell_matrices(self.n, self.r)
        matgens = polymat.projectivise(polymat.pack([polymat.from_matrix(x, dtype=self.dtype()) for x in gens]))
        matinvs = polymat.projectivise(polymat.pack([polymat.from_matrix(x, dtype=self.dtype(), proj=True) for x in invs]))
        return matgens, matinvs

    def mul(self, left: npt.NDArray, right: npt.NDArray):
        """Multiplication in the representation (performs mod p if necessary, and projectivisation)."""
        product = polymat.mul(left, right)
        if self.p != 0:
            product = product % self.p

        return polymat.projectivise(product)


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


@functools.cache
def symmetric_table(rep: JonesSummand):
    """A table mapping permutations x to the projective image ρ(x) in the given representation."""
    gens, invs = rep.artin_gens_invs()
    eye = rep.id()
    table = {
        perm: polymat.projectivise(functools.reduce(rep.mul, [gens[s] for s in perm.shortlex()], eye))
        for perm in SymmetricGroup(rep.n).elements()
    }

    # We rely on an assumption later that ρ(Δ)^2 = id (at least projectively).
    w0 = SymmetricGroup(rep.n).longest_element()
    assert np.equal(rep.mul(table[w0], table[w0]), eye).all()

    return table


def evaluate_braid_factors(rep: JonesSummand, braid: GNF):
    """Evaluate the canonical factors of a braid in the given representation."""
    inf, factors = braid.canonical_decomposition()
    start = rep.id()

    # We know that Δ^2 = id in our representations.
    table = symmetric_table(rep)
    w0 = SymmetricGroup(rep.n).longest_element()
    start = rep.id() if inf % 2 == 0 else table[w0]

    return polymat.projectivise(functools.reduce(rep.mul, [table[factor] for factor in factors], start))


def evaluate_braids_of_same_length(rep: JonesSummand, braids: Sequence[GNF]) -> npt.NDArray:
    """
    Given a list of braids all of the same Garside length, evaluate them in the given representation.
    A single NDArray is returned.
    """
    length = braids[0].garside_length()
    assert all(braid.garside_length() == length for braid in braids)

    factors = [braid.canonical_factors() for braid in braids]
    table = symmetric_table(rep)
    w0 = SymmetricGroup(rep.n).longest_element()
    eye = rep.id()
    anti = table[w0]
    images = polymat.pack([eye if braid.inf() % 2 == 0 else anti for braid in braids])

    for l in range(length):
        perm_images = polymat.pack([table[factors[i][l]] for i in range(len(braids))])
        images = rep.mul(images, perm_images)

    return images


def evaluate_prefixes_of_same_length(rep: JonesSummand, braids: Sequence[GNF]) -> List[npt.NDArray]:
    """
    Given a list of braids all of the same Garside length, evaluate them in the Burau representation.
    Returns a list of NDArrays: if f(1), ..., f(l) are the canonical factors in a braid, the list returned
    has length l+1, and the first element is the identity, the next f(1), the next f(1) f(2), etc.
    """
    length = braids[0].garside_length()
    assert all(braid.garside_length() == length for braid in braids)

    factors = [braid.canonical_factors() for braid in braids]
    table = symmetric_table(rep)
    w0 = SymmetricGroup(rep.n).longest_element()
    eye = rep.id()
    anti = table[w0]
    images = [polymat.pack([eye if braid.inf() % 2 == 0 else anti for braid in braids])]

    for l in range(length):
        perm_images = polymat.pack([table[factors[i][l]] for i in range(len(braids))])
        images += [rep.mul(images[-1], perm_images)]

    return images


def evaluate_braids(rep: JonesSummand, braids: Sequence[GNF]) -> Sequence[npt.NDArray]:
    """
    Given a list of braids of varying lengths, evaluate them in the Burau representation.
    A sequence of NDArrays is returned, since this function internally separates the braids into
    classes of the same length, evaluates those, and then returns the result.
    """
    indices_by_length: typing.DefaultDict[int, List[int]] = collections.defaultdict(list)
    index_location = []
    for i, braid in enumerate(braids):
        length = braid.garside_length()
        index_location.append((length, len(indices_by_length[length])))
        indices_by_length[length] += [i]

    images_by_length = {
        length: evaluate_braids_of_same_length(rep, [braids[i] for i in indices])
        for length, indices in indices_by_length.items()
    }

    return [images_by_length[length][i] for length, i in index_location]


def sample_braids_images(rep: JonesSummand, length: int, count: int, rand=None):
    """
    Sample 'count' many braids of a certain Garside length in the braid group, all with Δ^0,
    and return two parallel lists of [braids] and [images]. The image list is really a tensor.
    """
    braids = [BraidGroup(rep.n).sample_braid_perm(length, rand=rand) for _ in range(count)]
    return braids, evaluate_braids_of_same_length(rep, braids)


def sample_braids_images_upto_length(rep: JonesSummand, from_length: int, upto_length: int, count: int, rand=None):
    """Same as above, but this time sample all lengths [from_length, ..., upto_length]."""
    pairs = [
        sample_braids_images(rep, length=length, count=count, rand=rand)
        for length in range(from_length, upto_length + 1)
    ]
    braids = [braid for braids, _ in pairs for braid in braids]
    images = polymat.concatenate([images for _, images in pairs])
    return braids, images


def braids_images_to_dataframe(braids, images):
    """
    Put many (braid, image) pairs into a dataframe, annotated by statistics.
    The rows in the dataframe are in the same order as the input arrays.
    """
    if isinstance(images, np.ndarray):
        assert len(images.shape) == 4, f"Images should have a shape of length 4, instead got {images.shape}"
        assert images.shape[0] == len(braids), f"There are {len(braids)} braids but {images.shape[0]} images"
        image_list = [images[i] for i in range(images.shape[0])]
    elif isinstance(images, list):
        image_list = list(images)
    else:
        raise ValueError("Cannot determine input type")

    df = pd.DataFrame.from_dict({
        'braid': braids,
        'image': image_list,
    })
    df['length'] = df['braid'].apply(lambda x: x.garside_length())
    df['nz_terms'] = df['image'].apply(polymat.nz_terms)
    df['degmax'] = df['image'].apply(polymat.degmax)
    df['valmin'] = df['image'].apply(polymat.valmin)
    df['rhogap'] = df['image'].apply(polymat.rhogap)
    df['rhogap/length'] = df['rhogap'] / df['length']
    return df


def all_of_length_below_1(rep: JonesSummand, length: int):
    """
    Check all braids of the given Garside length, and return those (and their images) for which
    rhogap/length < 1.
    """

    good_braids = []
    good_images = []
    B = BraidGroup(rep.n)

    for braids in batched(B.all_of_garside_length(length), 1000):
        images = evaluate_braids_of_same_length(rep, braids)
        rhogap = polymat.rhogap(images)
        indices, = np.nonzero(rhogap/length < 1.0)

        if len(indices) > 0:
            good_braids += [braids[i] for i in indices]
            good_images += [images[indices]]

    return good_braids, polymat.concatenate(good_images)




class Tracker:
    def __init__(self, rep: JonesSummand, bucket_size: int, bucket_keys: Tuple[str, ...], criterion, rand: random.Random = None, minimise=None):
        self.rep = rep
        self.bucket_size = bucket_size

        # Bucket keys should at least be as fine as 'length' and 'projlen'
        assert 'length' in bucket_keys
        assert 'projlen' in bucket_keys
        self.bucket_keys = list(bucket_keys)

        # The criterion should read the braid image stats, and return the set of braids that
        # it wants to keep.
        self.criterion = criterion

        # Buckets map tuples (pairs of statistics, for instance (length, projlen)) to collections of braids,
        # and their images in the projectivised Burau representation. The images within a single bucket are
        # stored in one tensor of shape (bucket_size, n-1, n-1, D) where the D is determined by the first image
        # to go into the bucket (so images within a bucket must have consistent D).
        self.buckets: Set[Tuple] = set()
        self.bucket_braids: Dict[Tuple, List[GNF]] = {}
        self.bucket_images: Dict[Tuple, npt.NDArray] = {}

        # Store the hashes of matrix elements.
        self.image_hashes: Dict[int, List[GNF]] = {}

        # The bucket braid set is used if and only if reservoir sampling is not used.
        self.bucket_braid_set: Dict[Tuple, Set[GNF]] = {}
        self.bucket_reservoir_counts: Dict[Tuple, int] = {}

        # If rand is not None, then reservoir sampling will be used for the buckets.
        self.rand = rand

        # If minimise is not None, then within each bucket we will attempt to minimise the given statistic.
        # Minimise should be a function (α, dim, dim, ?) ↦ (α).
        # The heap is a max-heap, made of pairs (-minimise statistic, index).
        self.bucket_heap: Dict[Tuple, List[Tuple]] = {}
        self.minimise = minimise if minimise is not None else lambda x: np.full(shape=x.shape[:-3], fill_value=0, dtype=np.int64)


    def discard_bucket(self, bucket: Tuple):
        if bucket not in self.buckets:
            return

        self.buckets.remove(bucket)
        del self.bucket_braids[bucket]
        del self.bucket_images[bucket]

        if bucket in self.bucket_braid_set:
            del self.bucket_braid_set[bucket]

        # We're not removing the reservoir counts since that's kinda interesting.


    def braid_image_stats(self, braids: List[GNF], images: npt.NDArray) -> pd.DataFrame:
        assert len(images.shape) == 4 and images.shape[0] == len(braids)
        braid_lengths = np.array([braid.garside_length() for braid in braids])
        return pd.DataFrame.from_dict({
            #'braid': braids,
            #'image': [images[i] for i in range(len(braids))],
            'length': braid_lengths,
            'nz_terms': polymat.nz_terms(images),
            'degmax': polymat.degmax(images),
            'valmin': polymat.valmin(images),
            'rhogap': polymat.rhogap(images),
            'projlen': polymat.projlen(images),
            # Make sure we don't divide by zero here:
            'rhogap/length': polymat.rhogap(images) / braid_lengths.clip(min=1),
            'projrank2': polymat.projrank(images, terms=2),
        })


    def bucket_stats(self, bucket: Tuple):
        if bucket not in self.buckets:
            raise ValueError(f"Bucket {bucket} not found.")

        return self.braid_image_stats(*self.bucket_braids_images(bucket))


    def bucket_braids_images(self, bucket: Tuple) -> Tuple[List[GNF], npt.NDArray]:
        assert bucket in self.buckets
        braids = self.bucket_braids[bucket]
        images = self.bucket_images[bucket][:len(braids), ...]
        return braids, images


    def stats(self):
        df = pd.DataFrame(
            columns=['bucket', 'count', *self.bucket_keys],
            data=[
                (bucket, len(self.bucket_braids[bucket]), *bucket)
                for bucket in self.buckets
            ],
        )
        df['rhogap/length'] = (df['projlen'] - 1) / 2 / df['length']
        if self.rand is not None:
            df['reservoir_count'] = df['bucket'].apply(self.bucket_reservoir_counts.get)
        return df

    def stats_reservoir(self):
        """Sometimes we only keep around the reservoir counts."""
        return pd.DataFrame(
            columns=['bucket', *self.bucket_keys, 'reservoir_count'],
            data=[
                (bucket, *bucket, count)
                for bucket, count in self.bucket_reservoir_counts.items()
            ]
        )


    def valiate_buckets(self):
        """Ensure that the braids in each bucket evaluate to the matrices stored there."""
        for bucket in self.buckets:
            braids, images = self.bucket_braids_images(bucket)
            actual = evaluate_braids_of_same_length(BraidGroup(braids[0].n), braids, p=self.p)
            if not np.all(images == actual):
                print(f"A braid image is wrong in bucket {bucket}")
                for i in range(len(braids)):
                    if not np.all(images[i] == actual[i]):
                        print(f"Braid index {i}, {braids[i]}")
                        print("Actual representation value:")
                        print(actual[i])
                        print("What we have:")
                        print(images[i])
                        return braids[i], actual[i], images[i]

        return "All bucket entries are sound."


    def add_braids_images(self, braids: List[GNF], images: npt.NDArray):
        """
        Add a bundle of braids and their images to the collection, returning a summary of those
        new entries which were introduced by the operation.
        """
        stats = self.braid_image_stats(braids, images)
        keep = self.criterion(stats)
        indices = [i for i in range(len(keep)) if keep[i]]

        lengths = [braid.garside_length() for braid in braids]
        projlens = polymat.projlen(images)
        self.minimise(images)

        kept = []
        for i in indices:
            braid, image = braids[i], images[i]
            bucket = (lengths[i], projlens[i])

            if bucket not in self.buckets:
                self.buckets |= {bucket}
                self.bucket_braids[bucket] = [braid]
                self.bucket_braid_set[bucket] = {braid}
                # self.bucket_heap[bucket] = [(-minstat[i], 0)]
                self.bucket_reservoir_counts[bucket] = 1
                image = polymat.projectivise(image)
                self.bucket_images[bucket] = np.zeros(shape=(self.bucket_size, *image.shape), dtype=image.dtype)
                self.bucket_images[bucket][0, :, :, :] = image
                kept += [i]
                continue

            # if -self.bucket_heap[bucket][0][0] <= minstat[i]:


            self.bucket_reservoir_counts[bucket] += 1
            if len(self.bucket_braids[bucket]) == self.bucket_size:
                if self.rand is not None:
                    j = self.rand.randint(1, self.bucket_reservoir_counts[bucket])
                    if j <= self.bucket_size:
                        # Replace an element
                        self.bucket_braids[bucket][j-1] = braid
                        image = polymat.projectivise(image)
                        self.bucket_images[bucket][j-1] = image
                continue

            if braid in self.bucket_braid_set[bucket]:
                continue

            idx = len(self.bucket_braids[bucket])
            self.bucket_braids[bucket] += [braid]
            self.bucket_braid_set[bucket] |= {braid}
            image = polymat.projectivise(image)
            self.bucket_images[bucket][idx, :, :, :] = image
            kept += [i]

        #if kept:
        #    print(f"{len(stats[keep])=}, {min(kept)=}, {max(kept)=}")
        #return keep.iloc[kept]


    def bucket_product(self, bucket1: Tuple, bucket2: Tuple):
        """
        Take the product of everything in bucket1 with everything in bucket2, and place the
        resulting braids and their images back into the buckets.
        """
        braids1, images1 = self.bucket_braids_images(bucket1)
        braids2, images2 = self.bucket_braids_images(bucket2)

        braids = [a*b for a in braids1 for b in braids2]
        images = self.rep.mul(images1[:, None, ...], images2[None, :, ...])
        self.add_braids_images(braids, images.reshape((-1, *images.shape[-3:]), order='C'))


    def nf_descendants(self, bucket: Tuple, length: int = 1):
        """Try exhaustively looking at the normal form descendants of all elements in a bucket up to the given length."""
        all_pairs = [(i, braid, suffix) for i, braid in enumerate(self.bucket_braids[bucket]) for suffix in braid.nf_suffixes(length)]
        for pairs in batched(all_pairs, 1000):
            left = polymat.pack([self.bucket_images[bucket][i] for i, _, _ in pairs])
            rights = evaluate_prefixes_of_same_length(self.rep, [suffix for _, _, suffix in pairs])
            for i in range(1, length+1):
                images = self.rep.mul(left, rights[i])
                self.add_braids_images([braid * suffix.substring(0, i) for _, braid, suffix in pairs], images)

    def substrings(self, bucket: Tuple):
        """Try all substrings of a bucket of length 2 or more."""
        assert bucket in self.buckets
        braids = self.bucket_braids[bucket]
        length = braids[0].garside_length()
        for i in range(1, length - 1):
            images = evaluate_prefixes_of_same_length(self.rep, [braid.substring(i, length) for braid in braids])
            for j in range(i+2, length - 1):
                self.add_braids_images([braid.substring(i, j) for braid in braids], images[j-i])

    def bootstrap_exhaustive(self, upto_length: int):
        """Iterate over all braids up to the given Garside length to bootstrap the buckets."""
        B = BraidGroup(self.rep.n)
        for length in range(1, upto_length + 1):
            for braids in batched(B.all_of_garside_length(length), 10_000):
                images = evaluate_braids_of_same_length(self.rep, braids)
                self.add_braids_images(braids, images)
