# Braids project

Code for the paper *4-strand Burau is unfaithful modulo 5* by Joel Gibson, Geordie Williamson, and Oded Yacobi.

[*4-strand Burau is Unfaithful Modulo 5*](https://arxiv.org/abs/2310.02403)


## Installation

Make sure you have both Python 3.10, and the [Poetry](https://python-poetry.org/) package manager.
Clone the project and install the dependencies, then run `pytest` to make sure everything is working:

```shell
$ git clone git@github.com:geordw/braids-project.git
$ cd braids-project
$ poetry install

# Confirm that everything is working.
$ poetry run pytest
=== 68 passed in 0.37s ===
```


## Verify results (Python)

The summary of results quoted in the paper are in the IPython Notebook [`p=5 kernel elements.ipynb`](p=5%20kernel%20elements.ipynb).
You can view this on GitHub, or re-run it yourself by starting a Jupyter notebook server:

```shell
$ poetry run jupyter notebook
```

## Searching for braids (p=2 example)

The code is separated into the `peyl` library (a pun on "Weyl" and "Python"), some very basic `tests` for that library, and a script called `search.py` which performs the search algorithm described in the paper.

To start with an easy case, make sure that you can find some braids in the modulo 2 kernel for the 4-strand Burau, which should be nearly instant.

```shell
$ poetry run python search.py 4 1 2 --bootstrap-length 3 --step-size 1 --seed 1 --bucket-size 10 --use-best 100 --stop-at-projlen-1
...
Found kernel elements:
(n=4, r=1, p=2) near-kernel element: Garside length 8, Garside form (0, [(1, 3, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (2, 0, 3, 1)])
```

The `--stop-at-projlen-1` argument tells the program to stop when it sees a projlen 1 braid (which is necessarily in the kernel times a power of the Garside element), and to print a representation of the braid.
Otherwise, the program will continue and not report the actual braid (it is intended to be used in a different mode where it gradually collects "small" elements into a database. But we will not need that here).

The braid itself is given as a pair (infimum, word in permutations), where the infimum is the power of the Garside element appearing in the normal form, and the permutations are the rest of the Garside form (or rather, their positive lifts are).

We can go and verify that this element is indeed in the kernel, first using IPython:

```python
In [1]: import peyl

In [2]: B = peyl.BraidGroup(n=4)

In [3]: braid = B.from_gnf_pair((0, [(1, 3, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (2, 0, 3, 1)]))

In [4]: braid.canonical_length()
Out[4]: 8

In [5]: rep = peyl.JonesCellRep(n=4, r=1, p=2)

In [6]: rep.evaluate(braid)
Out[6]:
Matrix([
    [LPoly('v^16'), LPoly('0'), LPoly('0')],
    [LPoly('0'), LPoly('v^16'), LPoly('0')],
    [LPoly('0'), LPoly('0'), LPoly('v^16')],
])

In [7]: braid.magma_artin_word()
Out[7]: [1, 3, 2, 2, 1, 3, 1, 3, 2, 2, 1, 3, 1, 3, 2, 2, 1, 3, 1, 3, 2, 2, 1, 3]
```

The last line prints a long list of the Artin generators, 1-indexed (so that the 4-strand braid group is generated by 1, 2, 3), with -1, -2, -3 representing inverses. This is a convenient format that Magma understands for specifying elements of the braid group, and we use Magma to give another double-check of our results.

## Searching for braids (nontrivial example)

Some playing with parameters very similar to the p=2 case above gives p=3 kernel elements.
The search power needs to be turned up considerably to generate elements in the p=5 kernel.
This search below should be reproducible (provided that you are using a version of Numpy with the same random number algorithm, which you should be if you installed the packages with Poetry):

```shell
$ time poetry run python search.py 4 1 5 --bootstrap-length 5 --step-size 1 --seed 3 --bucket-size 15000 --use-best 30000 --stop-at-projlen-1
...
Found kernel elements:
(n=4, r=1, p=5) near-kernel element: Garside length 65, Garside form (0, [(3, 1, 2, 0), (1, 0, 3, 2), (1, 3, 0, 2), (2, 0, 3, 1), (0, 3, 1, 2), (2, 0, 3, 1), (0, 3, 1, 2), (2, 0, 3, 1), (3, 1, 2, 0), (3, 1, 0, 2), (2, 0, 3, 1), (3, 1, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (0, 2, 1, 3), (2, 0, 3, 1), (0, 3, 1, 2), (2, 0, 3, 1), (0, 3, 1, 2), (2, 0, 3, 1), (3, 1, 2, 0), (3, 1, 0, 2), (2, 0, 3, 1), (3, 1, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (0, 2, 1, 3), (2, 0, 3, 1), (0, 3, 1, 2), (2, 0, 3, 1), (0, 3, 1, 2), (2, 0, 3, 1), (3, 1, 2, 0), (3, 1, 0, 2), (2, 0, 3, 1), (3, 1, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (0, 2, 1, 3), (2, 0, 3, 1), (0, 3, 1, 2), (2, 0, 3, 1), (0, 3, 1, 2), (2, 0, 3, 1), (3, 1, 2, 0), (3, 1, 0, 2), (2, 0, 3, 1), (3, 1, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (0, 2, 1, 3), (2, 0, 3, 1), (0, 3, 1, 2), (2, 0, 3, 1), (0, 3, 1, 2), (2, 0, 3, 1), (3, 1, 2, 0), (3, 1, 0, 2), (2, 0, 3, 1), (3, 1, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2), (2, 3, 0, 1), (2, 3, 0, 1), (0, 2, 1, 3)])
(n=4, r=1, p=5) near-kernel element: Garside length 65, Garside form (0, [(1, 0, 3, 2), (1, 0, 3, 2), (1, 3, 0, 2), (2, 0, 3, 1), (1, 2, 0, 3), (2, 0, 3, 1), (1, 2, 0, 3), (2, 0, 3, 1), (3, 1, 2, 0), (1, 3, 2, 0), (2, 0, 3, 1), (1, 3, 2, 0), (2, 0, 3, 1), (1, 3, 0, 2), (0, 2, 1, 3), (2, 0, 3, 1), (1, 2, 0, 3), (2, 0, 3, 1), (1, 2, 0, 3), (2, 0, 3, 1), (3, 1, 2, 0), (1, 3, 2, 0), (2, 0, 3, 1), (1, 3, 2, 0), (2, 0, 3, 1), (1, 3, 0, 2), (0, 2, 1, 3), (2, 0, 3, 1), (1, 2, 0, 3), (2, 0, 3, 1), (1, 2, 0, 3), (2, 0, 3, 1), (3, 1, 2, 0), (1, 3, 2, 0), (2, 0, 3, 1), (1, 3, 2, 0), (2, 0, 3, 1), (1, 3, 0, 2), (0, 2, 1, 3), (2, 0, 3, 1), (1, 2, 0, 3), (2, 0, 3, 1), (1, 2, 0, 3), (2, 0, 3, 1), (3, 1, 2, 0), (1, 3, 2, 0), (2, 0, 3, 1), (1, 3, 2, 0), (2, 0, 3, 1), (1, 3, 0, 2), (0, 2, 1, 3), (2, 0, 3, 1), (1, 2, 0, 3), (2, 0, 3, 1), (1, 2, 0, 3), (2, 0, 3, 1), (3, 1, 2, 0), (1, 3, 2, 0), (2, 0, 3, 1), (1, 3, 2, 0), (2, 0, 3, 1), (1, 3, 0, 2), (2, 3, 0, 1), (2, 3, 0, 1), (2, 3, 0, 1)])
Selected 29958 braids of length 65:
      bucket  count  length  projlen  rhogap/length  reservoir_count
0    (65, 1)      2      65        1       0.000000                2
1    (65, 3)      8      65        3       0.015385                8
2    (65, 4)      8      65        4       0.023077                8
3    (65, 5)     10      65        5       0.030769               10
4    (65, 6)     10      65        6       0.038462               10
5    (65, 7)     38      65        7       0.046154               38
6    (65, 8)     34      65        8       0.053846               34
7    (65, 9)     78      65        9       0.061538               78
8   (65, 10)     38      65       10       0.069231               38
9   (65, 11)    216      65       11       0.076923              216
10  (65, 12)    136      65       12       0.084615              136
11  (65, 13)    452      65       13       0.092308              452
12  (65, 14)    200      65       14       0.100000              200
13  (65, 15)   1096      65       15       0.107692             1096
14  (65, 16)    628      65       16       0.115385              628
15  (65, 17)   1512      65       17       0.123077             1512
16  (65, 18)    672      65       18       0.130769              672
17  (65, 19)   6028      65       19       0.138462             6028
18  (65, 20)   3792      65       20       0.146154             3792
19  (65, 21)  15000      65       21       0.153846            17660

Moving braids forward by 1 GNF letters...
   Done in 6.49 seconds.

________________________________________________________
Executed in  321.83 secs    fish           external
   usr time  317.26 secs    0.09 millis  317.26 secs
   sys time    4.96 secs    1.63 millis    4.96 secs
```

As indicated above, the search takes about 6 minutes (and only uses around 200 MB of RAM) on a moderately powered desktop computer.


## Verify results (Magma)

As alluded above, we have verified our results using the Magma computer algebra system, using the program [`MagmaCheck.m`](MagmaCheck.m).
For readers without a copy of Magma, the results of the command `magma MagmaCheck.m` have been saved to [`MagmaCheck.out`](MagmaCheck.out)
