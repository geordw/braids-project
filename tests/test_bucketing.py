import pandas as pd
import pandas.testing as pd_test

from peyl import DGNF, JonesCellRep
from peyl.bucketing import Stats, Tracker


def test_contruct():
    rep = JonesCellRep(n=4, r=1, p=2)
    track = Tracker(
        rep=rep,
        bucket_capacity=10,
        bucket_braidstats=[Stats.CanonicalLength, Stats.ProjectiveLength],
        other_braidstats=[Stats.NonzeroTerms],
    )

    # Test that we can add a single braid.
    braid = DGNF.identity(rep.n)
    image = rep.polymat_evaluate_braid(braid)
    track.add_braids_images([braid], image[None, ...])
    assert track.buckets.keys() == {(0, 1)}

    # Check that the stats for the braids in this bucket are correct.
    pd_test.assert_frame_equal(
        track.bucket_stats((0, 1)),
        pd.DataFrame(
            columns=['braid', 'length', 'projlen', 'nz_terms'],
            data=[(braid, 0, 1, 3)],
        ),
    )

    # Check that the stats across all buckets are correct.
    pd_test.assert_frame_equal(
        track.stats(),
        pd.DataFrame(
            columns=['bucket', 'count', 'length', 'projlen', 'reservoir_count', 'reservoir_weight_avg', 'reservoir_weight_avg_seen'],
            data=[((0, 1), 1, 0, 1, 1, 1.0, 1.0)],
        ),
    )


    # Test that the length and projlen show up in the frame.
    track.stats()
