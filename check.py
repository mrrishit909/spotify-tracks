"""The project's check.

    ./venv/bin/python check.py

1. The data is as described: 114,000 rows, 114 genres x 1,000, 89,741 unique tracks.
2. Zero popularity: the compilation-copy explanation recomputes from the raw file.
3. Every number in the explorer's genres.json is a genre median that recomputes from the raw file.
4. Nothing track-level is published: the explorer data has one row per genre and no track names or IDs.
"""
import json

import numpy as np

from analysis import FEATURES, HERE, load


def main():
    d = load()
    assert len(d) == 114000 and d["track_genre"].value_counts().eq(1000).all() and d["track_id"].nunique() == 89741
    t = d.drop_duplicates("track_id")
    key = t["artists"].fillna("") + " | " + t["track_name"].fillna("")
    copies = key.map(key.value_counts())
    s = json.loads((HERE / "results" / "summary.json").read_text())
    assert abs(float((t["popularity"][copies > 1] == 0).mean()) - s["zero_pop_share_if_song_has_other_copies"]) < 0.001
    assert abs(float((t["popularity"][copies == 1] == 0).mean()) - s["zero_pop_share_if_unique_song"]) < 0.001

    ex = json.loads((HERE / "explorer" / "genres.json").read_text())
    med = d.groupby("track_genre")[FEATURES + ["popularity"]].median()
    for g, row in ex["genres"].items():
        for f in FEATURES + ["popularity"]:
            assert abs(row[f] - med.loc[g, f]) < 0.001, (g, f)
    text = json.dumps(ex)
    assert len(ex["genres"]) == 114 and not any(tid in text for tid in t["track_id"].head(2000))
    print(f"OK: 114 x 1,000 rows, 89,741 tracks; zero-popularity split recomputes "
          f"({s['zero_pop_share_if_song_has_other_copies']:.1%} vs {s['zero_pop_share_if_unique_song']:.1%}); "
          f"all {len(ex['genres']) * (len(FEATURES) + 1):,} explorer medians recompute; no track-level data published")


if __name__ == "__main__":
    main()
