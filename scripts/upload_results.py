#!/usr/bin/env python3
"""
Upload this machine's overnight results to the shared results website.

    UPLOAD_TOKEN=<the token you were given> python scripts/upload_results.py --tag pi

Zips the output folder (out/mc by default) and POSTs it to the deployed
site. Standard library only -- nothing new to install on any of the four
machines, which matters most on the one you are least likely to sit down at
again before Tuesday.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://cuboat.netlify.app/api/upload"


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
    size_kb = Path(zip_path).stat().st_size / 1024
    print(f"zipped {src} -> {zip_path} ({size_kb:.0f} KB)")

    data = Path(zip_path).read_bytes()
    req = urllib.request.Request(
        args.url, data=data, method="POST",
        headers={
            "x-upload-token": token,
            "x-machine-tag": args.tag,
            "x-filename": f"{args.tag}_results.zip",
            "content-type": "application/zip",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(resp.read().decode())
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
