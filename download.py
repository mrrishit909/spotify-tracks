"""Step 1: download the Spotify Tracks Dataset (maharshipandya, Hugging Face; 114,000 rows, 20 MB). Not committed.

    ./venv/bin/python download.py
"""
import ssl
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
OUT = Path(__file__).parent / "data" / "raw" / "dataset.csv"

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    OUT.write_bytes(urllib.request.urlopen(URL, context=ctx, timeout=120).read())
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
