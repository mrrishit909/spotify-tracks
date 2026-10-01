"""Step 3: charts and the explorer's data (aggregates only).

    ./venv/bin/python charts.py   -> charts/01_zero.png, 02_features.png, 03_fingerprints.png, explorer/genres.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
INK, DIM, GRID, BG, BLUE, ORANGE = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
    "axes.titlelocation": "left", "axes.titlepad": 12,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / "charts" / name, dpi=150)
    plt.close(fig)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    s = json.loads((HERE / "results" / "summary.json").read_text())
    fp = pd.read_csv(HERE / "results" / "genre_fingerprints.csv", index_col=0)

    sel = ["rock", "country", "jazz", "soul", "latin", "dance", "pop", "metal", "k-pop", "hip-hop"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    y = np.arange(len(sel))
    for i, gname in enumerate(sel):
        a, b = fp.loc[gname, "popularity"], fp.loc[gname, "popularity_best_copy"]
        ax.plot([a, b], [i, i], color="#3a3a3a", linewidth=2)
    ax.scatter(fp.loc[sel, "popularity"], y, color=ORANGE, s=50, zorder=3, label="all rows (compilation copies included)")
    ax.scatter(fp.loc[sel, "popularity_best_copy"], y, color=BLUE, s=50, zorder=3, label="each song's most popular release only")
    ax.set_yticks(y, sel)
    ax.invert_yaxis()
    ax.set_xlabel("median Spotify popularity (0-100) in the genre sample")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, labelcolor=INK, loc="upper right", fontsize=9)
    ax.set_title("Median popularity of 0 for rock? Compilation copies, not real taste")
    save(fig, "01_zero.png")

    feats = list(s["spearman_pooled_best_copy"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    yv = np.arange(len(feats))
    ax.scatter([s["spearman_pooled_best_copy"][f] for f in feats], yv, color=ORANGE, s=50, label="all tracks pooled", zorder=3)
    ax.scatter([s["spearman_within_genre_best_copy"][f] for f in feats], yv, color=BLUE, s=50, label="within each genre (average)", zorder=3)
    ax.axvline(0, color=DIM, linewidth=1)
    ax.set_yticks(yv, [f.replace("_min", " (min)") for f in feats])
    ax.invert_yaxis()
    ax.set_xlim(-0.3, 0.3)
    ax.set_xlabel("rank correlation with popularity (Spearman)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, labelcolor=INK, loc="lower right", fontsize=9)
    ax.set_title("How a track sounds says almost nothing about how popular it is")
    save(fig, "02_features.png")

    show = ["classical", "ambient", "acoustic", "jazz", "country", "pop", "k-pop", "hip-hop", "reggaeton", "edm", "techno", "metal", "death-metal", "comedy"]
    cols = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness"]
    m = fp.loc[show, cols]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    im = ax.imshow(m.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#0f1a2b", BLUE, "#cfe2fb"]), vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(m.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=BG if v > 0.6 else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(show)), show)
    ax.grid(False)
    ax.set_title("Genre fingerprints: median audio features (0-1)")
    save(fig, "03_fingerprints.png")

    (HERE / "explorer").mkdir(exist_ok=True)
    out = {"source": "Spotify Tracks Dataset (Hugging Face, maharshipandya), audio features from Spotify's Web API; medians per genre",
           "features": list(fp.columns), "genres": {g: {k: (None if pd.isna(v) else float(v)) for k, v in r.items()} for g, r in fp.iterrows()}}
    (HERE / "explorer" / "genres.json").write_text(json.dumps(out, separators=(",", ":")))
    print("charts/01_zero.png, 02_features.png, 03_fingerprints.png, explorer/genres.json")


if __name__ == "__main__":
    main()
