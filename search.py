"""
This script conducts a search of small projlen braids in a representation for as long as it can,
saving the results to a file (an SQLite database) along the way. The argument order is:
"""

import argparse
import dataclasses
import random
import sqlite3
import time

import peyl

DB_PRAGMAS = '''
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
'''

DB_SCHEMA = '''
CREATE TABLE IF NOT EXISTS good_braids (
    n INT,
    r INT,
    p INT,
    length INT,
    projlen INT,
    gnf TEXT
)
'''

parser = argparse.ArgumentParser('Perform a long search')
parser.add_argument('n', type=int, help='Number of strands in braid group')
parser.add_argument('r', type=int, help='Specifies the two-rowed representation (n - r, r)')
parser.add_argument('p', type=int, help='Prime to reduce modulo')
parser.add_argument('--bootstrap-length', type=int, default=6, help='Garside length to exhaustively bootstrap with')
parser.add_argument('--bucket-size', type=int, default=3000, help='Size of buckets for search')
parser.add_argument('--use-best', type=int, default=50_000, help='Max num braids to use at each step')
parser.add_argument('--save-best', type=int, default=500, help='Max num braids to save to the database at each step')
parser.add_argument('--step-size', type=int, default=1, help='Number of GNF letters to move forward at each step')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--database', type=str, help='Name of database file to record to')
parser.add_argument('--stop-at-projlen-1', action='store_true', help='Exit once near-kernel elements have been found')

args = parser.parse_args()
assert 3 <= args.n <= 7
assert 0 <= 2*args.r <= args.n
assert 0 < args.p


# Utility to time blocks of code.
class elapsed:
    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.time


def main(conn: sqlite3.Connection = None):
    # Initialise buckets with an exhaustive init procedure
    rep = peyl.JonesSummand(n=args.n, r=args.r, p=args.p)
    B = peyl.BraidGroup(rep.n)
    track = peyl.Tracker(
        rep=rep,
        bucket_size=args.bucket_size,
        bucket_keys=('length', 'projlen'),
        criterion=lambda x: (
            x['length'] >= 1
        ),
        rand=random.Random(args.seed),
    )

    print(f"Representation: {rep}")
    print(f"Tracker initialised with bucket size {args.bucket_size}, random seed {args.seed}.")
    print(f"Bootstrapping up to Garside length {args.bootstrap_length} ({B.count_all_of_garside_length(args.bootstrap_length):,} braids)...")

    with elapsed() as t:
        track.bootstrap_exhaustive(upto_length=args.bootstrap_length)
    print(f"Bootstrapping took {t.time:.2f} seconds")
    print("Initial buckets:")
    print(track.stats().sort_values(['length', 'projlen']))
    print()

    print("Proceeding on to search in 1 seconds...")
    time.sleep(1)

    should_halt = False

    for process_length in range(args.bootstrap_length - args.step_size + 1, 500):
        print(f"\n------- Length {process_length}")

        stats = track.stats()

        # If we've found any near-kernel elements, print them.
        nearker_buckets = stats[stats['projlen'] == 1]
        if len(nearker_buckets) >= 1:
            print("Found kernel elements:")
            for row in nearker_buckets.itertuples(index=False):
                for braid in track.bucket_braids[row.length, row.projlen]:
                    inf, perms = braid.canonical_decomposition()
                    print(f"(n={args.n}, r={args.r}, p={args.p}) near-kernel element: Garside length {len(perms)}, Garside form ({inf}, {[perm.word for perm in perms]})")
            
            if args.stop_at_projlen_1:
                should_halt = True

        # Select only those of this particular length, in increasing order of projlen.
        selection_length = stats[(stats['length'] == process_length)].sort_values('projlen', ignore_index=True)

        # Filter so that we are moving at most X forward, starting from the most promising.
        selection = selection_length[selection_length['count'].cumsum() <= args.use_best]
        selected_buckets = list(selection['bucket'])
        print(f"Selected {selection['count'].sum()} braids of length {process_length}:")
        print(selection)
        print()

        # Save this selection to the database
        if conn is not None:
            save_selection = selection_length[selection_length['count'].cumsum() <= args.save_best]
            print("Save selection:")
            print(save_selection)
            print(f"Saving {save_selection['count'].sum()} braids of length {process_length} to the database...")
            with elapsed() as t:
                conn.execute('BEGIN TRANSACTION')
                for length, projlen in save_selection['bucket']:
                    # Sometimes these end up being numpy int64s...
                    length, projlen = int(length), int(projlen)
                    conn.executemany(
                        'INSERT INTO good_braids VALUES (?, ?, ?, ?, ?, ?)',
                        [
                            (rep.n, rep.r, rep.p, length, projlen, str(dataclasses.astuple(braid)))
                            for braid in track.bucket_braids[length, projlen]
                        ]
                    )
                conn.commit()

            print(f"    Saved in {t.time:.2f} seconds.")

        # Move everything forward.
        print(f"Moving braids forward by {args.step_size} GNF letters...")
        with elapsed() as t:
            for bucket in selected_buckets:
                track.nf_descendants(bucket, length=args.step_size)
        print(f"   Done in {t.time:.2f} seconds.")

        # Clear braids <= length.
        for bucket in list(stats[stats['length'] <= process_length]['bucket']):
            track.discard_bucket(bucket)
        
        if should_halt:
            break


if args.database is not None:
    conn = sqlite3.connect(args.database)
    conn.executescript(DB_PRAGMAS)
    conn.execute(DB_SCHEMA)
    try:
        main(conn=conn)
    finally:
        conn.close()
else:
    main(conn=None)
