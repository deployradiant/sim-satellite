"""Download raw files into the local sim-satellite data cache."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

try:
    from sim_satellite.data import SatelliteDataCache
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from sim_satellite.data import SatelliteDataCache


def download_urls(urls: list[str], cache_dir: str | Path | None = None, source: str = "manual") -> list[Path]:
    """Download URLs into data/raw/<source> and return local paths."""

    cache = SatelliteDataCache(cache_dir).ensure()
    destination_dir = cache.raw_dir / source
    destination_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for url in urls:
        destination = destination_dir / _filename_from_url(url)
        urllib.request.urlretrieve(url, destination)
        downloaded.append(destination)

    return downloaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download raw data into the local cache.")
    parser.add_argument("urls", nargs="+", help="File URLs to download.")
    parser.add_argument("--cache-dir", default=None, help="Cache root. Defaults to ./data.")
    parser.add_argument("--source", default="manual", help="Source name stored under raw/.")
    args = parser.parse_args(argv)

    for path in download_urls(args.urls, cache_dir=args.cache_dir, source=args.source):
        print(path)
    return 0


def _filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"URL does not contain a filename: {url}")
    return name


if __name__ == "__main__":
    raise SystemExit(main())
