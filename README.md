# Spotify tracks: what goes with popularity, and how genres sound

An exploration of the popular Spotify Tracks Dataset (114,000 rows, 114 genres, Spotify's audio features and popularity),
with an **[interactive genre explorer](https://mrrishit909.github.io/projects/spotify-tracks/explorer/)**.

Build log: https://mrrishit909.github.io/projects/spotify-tracks/

## Why there's no machine-learning model here

The data was collected from Spotify's Web API, and Spotify's Developer Terms forbid using Spotify content to train
machine-learning or AI models. The uploader's "BSD" label on the copy can't override that. So this project is
descriptive only: counts, medians and rank correlations. It also publishes no track-level data: the raw file isn't
committed, and the explorer shows genre medians. An openly licensed music dataset would be the right base for the ML version.

## Data

Spotify Tracks Dataset by maharshipandya (Hugging Face / Kaggle):
https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset, collected with Spotify's Web API.

## Steps

| Step | File | What it does |
|---|---|---|
| 1 | `download.py` | the 20 MB CSV (not committed) |
| 2 | `analysis.py` → `results/` | data quality; popularity by genre; why 14% of tracks score 0; audio features vs popularity |
| 3 | `charts.py`, `explorer/`, `check.py` | charts; the explorer (genre medians only); checks |

## What the data turned out to be

- **Not a sample of Spotify, but 114 genre searches × 1,000 tracks each** (the dataset card says 125 genres). So genre sizes say
  nothing about real listening.
- **114,000 rows, but 89,741 unique tracks.** 16,299 tracks are listed under 2–9 genres, and 450 rows are exact duplicates.
- **10.5% of unique tracks have popularity 0, and most are compilation copies.** Spotify credits plays to a song's original
  release, so copies on albums like "Halloween Party 2022" score 0. 41.8% of songs that also exist under another track
  ID score 0, against 5.2% of songs that appear once.

## Findings

- **Seven genres, including rock, country, jazz and soul, have a median popularity of 0**, because 55–67% of their sampled tracks
  are compilation copies. Keeping each song's most popular release cuts that to three, and lifts "dance" from 1 to 67.
  Rock (5) and country (2) stay low because the originals often aren't in the sample at all. **Popularity by genre in
  this dataset mostly measures how it was collected.**
- **How a track sounds says almost nothing about how popular it is.** Every Spearman correlation between an audio feature and
  popularity is between −0.19 and +0.10 overall, and within genres the averages are all under 0.05. The largest is
  instrumentalness (−0.19 pooled), mostly because instrumental genres are less popular, not instrumental tracks within a genre.
- **Explicit tracks are about as popular as clean ones:** median 37 vs 33 overall, and +1 point within genres
  (higher in 61% of 72 genres).
- **Genres have clear fingerprints:** classical and ambient are acoustic and instrumental (around 0.9), death metal has
  energy 0.96 (with grindcore), comedy has speechiness 0.92, and the most danceable are kids' music (0.79), Chicago house (0.78), and latino and reggaeton (0.76).

## Check

`check.py` confirms the dataset's shape and recomputes the zero-popularity split (41.8% vs 5.2%). It also recomputes
all 1,254 genre medians in the explorer from the raw file, and confirms no track IDs are published.

## Run it

```
python3 -m venv venv && ./venv/bin/pip install pandas scipy matplotlib
./venv/bin/python download.py && ./venv/bin/python analysis.py && ./venv/bin/python charts.py && ./venv/bin/python check.py
```

## Not done

- No predictive model (see above).
- Your own listening history: Spotify lets you request your data export (Account → Privacy), which could be analysed
  the same way.
