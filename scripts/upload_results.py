#!/usr/bin/env python3
"""
Upload this machine's overnight results to the shared results website.

    UPLOAD_TOKEN=<the token you were given> python scripts/upload_results.py --tag pi

Zips the output folder (out/mc by default) and POSTs it to the deployed
site, in chunks small enough for a Netlify Function to accept in one
request (a real overnight run's output is easily tens of MB -- sent in one
piece, that gets cut off mid-upload with a broken pipe, since Netlify
Functions run on Lambda under the hood and cap a single request body around
6 MB). The server reassembles the chunks on its end; see
netlify/functions/upload.mts. Standard library only -- nothing new to
install on any of the four machines, which matters most on the one you are
least likely to sit down at again before Tuesday.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://cuboat.netlify.app/api/upload"
CHUNK_SIZE = 4 * 1024 * 1024  # comfortably under Netlify Functions' ~6 MB request-body cap


def upload_chunk(url: str, token: str, tag: str, filename: str,
                  upload_id: str, index: int, total: int, data: bytes) -> None:
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "x-upload-token": token,
            "x-machine-tag": tag,
            "x-filename": filename,
            "x-upload-id": upload_id,
            "x-chunk-index": str(index),
            "x-chunk-total": str(total),
            "content-type": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True,
                    help="which machine this is, e.g. pi, laptop2, friend")
    ap.add_argument("--dir", default="out/mc", help="folder to zip and upload")
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()

    token = os.environ.get("UPLOAD_TOKEN")
    if not token:
        print("Set UPLOAD_TOKEN as an environment variable first, e.g.\n"
              "  UPLOAD_TOKEN=... python scripts/upload_results.py --tag pi")
        return 1

    src = Path(args.dir)
    if not src.exists():
        print(f"{src} does not exist yet -- has the run produced any output?")
        return 1

    zip_path = shutil.make_archive(f"/tmp/{args.tag}_results", "zip", root_dir=src)
    data = Path(zip_path).read_bytes()
    size_kb = len(data) / 1024
    print(f"zipped {src} -> {zip_path} ({size_kb:.0f} KB)")

    filename = f"{args.tag}_results.zip"
    upload_id = f"{args.tag}-{int(time.time())}"
    total_chunks = max(1, (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE)

    for i in range(total_chunks):
        chunk = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        print(f"  uploading chunk {i + 1}/{total_chunks} ({len(chunk) / 1024:.0f} KB)")
        try:
            upload_chunk(args.url, token, args.tag, filename, upload_id, i, total_chunks, chunk)
        except urllib.error.HTTPError as e:
            print(f"upload failed: HTTP {e.code} -- {e.read().decode()[:300]}")
            if e.code == 401:
                print("(check UPLOAD_TOKEN matches what you were given)")
            return 1
        except urllib.error.URLError as e:
            print(f"upload failed: {e.reason}")
            return 1

    print("uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
