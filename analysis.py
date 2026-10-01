"""Step 2: what goes with popularity on Spotify, and what makes each genre sound like itself? (Descriptive only.)

    ./venv/bin/python analysis.py

No model is trained on this data: it comes from Spotify's Web API, and Spotify's developer terms forbid using
Spotify content to train machine-learning or AI models. Everything here is counting, medians and rank correlations.
Only aggregates are written (results/), never the track list itself.
  1. Data quality: duplicates, genre sampling, missing beats, zero popularity.
  2. Popularity by genre.
  3. Audio features vs popularity, pooled and within genre (genre is a confounder).
  4. Genre fingerprints (medians) for the explorer page.
  5. Why are 14% of tracks at popularity 0?
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FEATURES = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness",
            "loudness", "tempo", "duration_min"]


def load():
    d = pd.read_csv(HERE / "data" / "raw" / "dataset.csv", index_col=0)
    d["duration_min"] = d["duration_ms"] / 60000
    return d


def main():
    d = load()
    (HERE / "results").mkdir(exist_ok=True)
    tracks = d.drop_duplicates("track_id")                      # one row per track for track-level questions
    genres_per_track = d.groupby("track_id")["track_genre"].nunique()

    # 2. popularity by genre (a track counts in every genre it was listed under)
    g = d.groupby("track_genre")["popularity"].agg(["median", "mean", lambda s: (s == 0).mean()]).rename(columns={"<lambda_0>": "zero_share"})
    g = g.sort_values("median", ascending=False)

    # 3. rank correlation with popularity: pooled across genres vs within each genre (then averaged)
    pooled = tracks[FEATURES + ["popularity"]].corr("spearman")["popularity"].drop("popularity")
    within = d.groupby("track_genre").apply(lambda x: x[FEATURES + ["popularity"]].corr("spearman")["popularity"].drop("popularity"), include_groups=False)
    within_mean = within.mean()
    same_sign = (np.sign(within).eq(np.sign(within_mean), axis=1)).mean()

    # 4. genre fingerprints
    fp = d.groupby("track_genre")[FEATURES + ["popularity"]].median().round(3)
    fp["explicit_share"] = d.groupby("track_genre")["explicit"].mean().round(3)
    fp.to_csv(HERE / "results" / "genre_fingerprints.csv")

    # 5. zero popularity: are they re-releases of songs that exist elsewhere on Spotify?
    key = tracks["artists"].fillna("") + " | " + tracks["track_name"].fillna("")
    copies = key.map(key.value_counts())
    zero = tracks["popularity"] == 0
    zero_share_dup = float(zero[copies > 1].mean())
    zero_share_single = float(zero[copies == 1].mean())

    # fairer view: keep each song's most popular release (compilation copies get the plays credited elsewhere)
    best_ids = tracks.assign(song=key).sort_values("popularity", ascending=False).drop_duplicates("song")["track_id"]
    best = d[d["track_id"].isin(best_ids)]
    gb = best.groupby("track_genre")["popularity"].median().sort_values(ascending=False)
    best_tracks = best.drop_duplicates("track_id")
    pooled_best = best_tracks[FEATURES + ["popularity"]].corr("spearman")["popularity"].drop("popularity")
    within_best = best.groupby("track_genre").apply(lambda x: x[FEATURES + ["popularity"]].corr("spearman")["popularity"].drop("popularity"), include_groups=False).mean()
    fp["popularity_best_copy"] = gb.reindex(fp.index)
    fp.to_csv(HERE / "results" / "genre_fingerprints.csv")
    compilation_share = d.assign(dup=d["track_id"].map(dict(zip(tracks["track_id"], copies > 1)))).groupby("track_genre")["dup"].mean()

    explicit = tracks.groupby("explicit")["popularity"].median()
    within_explicit = d.groupby("track_genre").apply(lambda x: x.loc[x["explicit"], "popularity"].median() - x.loc[~x["explicit"], "popularity"].median()
                                                     if x["explicit"].sum() >= 20 else np.nan, include_groups=False).dropna()
    out = {
        "rows": len(d), "unique_tracks": len(tracks), "genres": int(d["track_genre"].nunique()),
        "rows_per_genre": sorted(set(d["track_genre"].value_counts())),
        "tracks_in_2plus_genres": int((genres_per_track > 1).sum()), "max_genres_for_one_track": int(genres_per_track.max()),
        "exact_duplicate_rows": int(d.duplicated().sum()),
        "zero_popularity_share": round(float((tracks["popularity"] == 0).mean()), 4),
        "no_beat_detected": int((d["tempo"] == 0).sum()), "missing_text_rows": int(d[["artists", "album_name", "track_name"]].isna().any(axis=1).sum()),
        "genre_popularity_top": g["median"].head(8).to_dict(), "genre_popularity_bottom": g["median"].tail(8).to_dict(),
        "genres_with_median_zero": int((g["median"] == 0).sum()),
        "best_copy_tracks": int(len(best_tracks)), "best_copy_genre_top": gb.head(8).to_dict(), "best_copy_genre_bottom": gb.tail(8).to_dict(),
        "best_copy_genres_with_median_zero": int((gb == 0).sum()),
        "spearman_pooled_best_copy": pooled_best.round(3).to_dict(), "spearman_within_genre_best_copy": within_best.round(3).to_dict(),
        "copy_share_in_zero_median_genres": compilation_share[g.index[g["median"] == 0]].round(2).to_dict(),
        "spearman_pooled": pooled.round(3).to_dict(), "spearman_within_genre_mean": within_mean.round(3).to_dict(),
        "within_genre_same_sign_share": same_sign.round(2).to_dict(),
        "zero_pop_share_if_song_has_other_copies": round(zero_share_dup, 3), "zero_pop_share_if_unique_song": round(zero_share_single, 3),
        "songs_with_multiple_track_ids": int((copies > 1).sum()),
        "explicit_median_popularity": {str(k): float(v) for k, v in explicit.items()},
        "explicit_minus_clean_within_genre": {"median": round(float(within_explicit.median()), 1), "genres": int(len(within_explicit)),
                                               "share_positive": round(float((within_explicit > 0).mean()), 2)},
    }
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
