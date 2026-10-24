"""Download the SONIC v1.1 ONNX checkpoint + sample motion from Hugging Face.

Uses parallel HTTP range requests with per-chunk resume, which is far faster than a
single stream on a throttled connection and survives dropped transfers.

    python scripts/download_sonic.py            # into models/sonic
    python scripts/download_sonic.py --workers 12
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = "nvidia/GEAR-SONIC"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
FILES = [
    "config.json",
    "LICENSE",
    "sonic_v1_1/observation_config.yaml",
    "sonic_v1_1/model_config.yaml",
    "sonic_v1_1/config.yaml",
    "sonic_v1_1/model_encoder.onnx",
    "sonic_v1_1/model_decoder.onnx",
    "sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl",
    "sample_data/robot_filtered/210531/walk_forward_amateur_001__A001_M.pkl",
]
CHUNK = 4 * 1024 * 1024


def _open(url: str, headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "g1-sim-downloader", **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def remote_size(url: str) -> int:
    with _open(url, {"Range": "bytes=0-0"}) as r:
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            return int(cr.split("/")[-1])
        return int(r.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, dest: Path, tries: int = 8) -> int:
    """Download bytes [start, end] into dest at offset start, resuming on failure."""
    got = 0
    for attempt in range(tries):
        try:
            with _open(url, {"Range": f"bytes={start + got}-{end}"}) as r, open(dest, "r+b") as f:
                f.seek(start + got)
                while True:
                    buf = r.read(256 * 1024)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
            if start + got > end:
                return got
        except Exception as e:  # noqa: BLE001
            time.sleep(min(2 ** attempt, 20))
            print(f"    retry {attempt + 1} for bytes {start}-{end}: {e}", file=sys.stderr)
    raise RuntimeError(f"failed range {start}-{end} of {url}")


def download(rel: str, out_dir: Path, workers: int) -> None:
    url = BASE + rel
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size(url)
    if dest.exists() and dest.stat().st_size == size:
        print(f"  ok       {rel} ({size / 1e6:.1f} MB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        f.truncate(size)
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    print(f"  fetching {rel} ({size / 1e6:.1f} MB, {len(ranges)} chunks, {workers} workers)")
    t0 = time.perf_counter()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for n in ex.map(lambda r: fetch_range(url, r[0], r[1], tmp), ranges):
            done += n
            pct = 100 * done / size
            rate = done / 1e6 / max(time.perf_counter() - t0, 1e-6)
            print(f"\r    {pct:5.1f}%  {rate:5.1f} MB/s", end="", flush=True)
    print()
    os.replace(tmp, dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "sonic")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    for rel in FILES:
        download(rel, args.out, args.workers)
    print("done")


if __name__ == "__main__":
    main()
