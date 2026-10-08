"""Download and verify the processed CA-URC HDF5 release data (standard library only).

Examples:
  python tools/download_data.py --dataset all
  python tools/download_data.py --dataset snapshot
  python tools/download_data.py --dataset all --check-only

Each ZIP is downloaded and extracted separately to limit temporary disk usage.
Files and archives are checked against the bundled SHA-256 manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
import urllib.error
import urllib.request
import zipfile

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = "https://github.com/mrp5560/ca-urc-jgr-solid-earth/releases/download/v1.0.0"
BLOCK = 4 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_target(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in relative or "\\" in relative:
        raise ValueError(f"Unsafe archive path: {relative}")
    target = root.joinpath(*path.parts)
    resolved = target.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"Path escapes destination (possibly through a symlink): {relative}")
    return target


def valid_file(root, record):
    target = safe_target(root, record["path"])
    return target.is_file() and target.stat().st_size == record["size_bytes"] and sha256_file(target) == record["sha256"]


def download(url, destination, archive):
    if destination.is_file() and destination.stat().st_size == archive["size_bytes"] and sha256_file(destination) == archive["sha256"]:
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CA-URC-data-downloader/1.0"})
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                for block in iter(lambda: response.read(BLOCK), b""):
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
            if size != archive["size_bytes"] or digest.hexdigest() != archive["sha256"]:
                raise ValueError(f"Archive checksum/size mismatch: {archive['name']}")
            os.replace(temporary, destination)
            return
        except (OSError, urllib.error.URLError, ValueError) as error:
            if attempt == 3:
                raise RuntimeError(f"Download failed after {attempt} attempts: {url}\n{error}") from error
            print(f"Retrying {archive['name']} ({attempt}/3): {error}", file=sys.stderr, flush=True)
            time.sleep(2 ** attempt)


def extract_verified(path, root, archive):
    expected = {item["path"]: item for item in archive["files"]}
    with zipfile.ZipFile(path) as source:
        members = source.infolist()
        names = [member.filename for member in members]
        if len(set(names)) != len(names) or set(names) != set(expected):
            raise ValueError(f"Archive members differ from manifest: {archive['name']}")
        for member in members:
            record = expected[member.filename]
            target = safe_target(root, member.filename)
            if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16) or member.file_size != record["size_bytes"]:
                raise ValueError(f"Invalid archive member: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".download-partial")
            digest, size = hashlib.sha256(), 0
            with source.open(member) as inp, temporary.open("wb") as out:
                for block in iter(lambda: inp.read(BLOCK), b""):
                    digest.update(block)
                    size += len(block)
                    out.write(block)
            if size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Extracted HDF5 checksum mismatch: {member.filename}")
            os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["full", "snapshot", "all"], default="all")
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY, help="Destination root; preserves repository-relative paths.")
    parser.add_argument("--manifest", type=Path, default=REPOSITORY / "data/RELEASE_DATA_MANIFEST.json")
    parser.add_argument("--release-url", default=DEFAULT_RELEASE, help="Override only when using a mirror with the same checksums.")
    parser.add_argument("--check-only", action="store_true", help="Verify existing HDF5 files without downloading or changing files.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded ZIPs in <repo-root>/.data-download.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1:
        parser.error("Unsupported release manifest schema")
    archives = [archive for archive in manifest["archives"] if args.dataset == "all" or archive["dataset"] == args.dataset]
    if not archives:
        parser.error("No archives selected")
    if any(archive.get("state") != "complete" or not archive.get("sha256") or any(not record.get("sha256") for record in archive["files"]) for archive in archives):
        parser.error("Release manifest is incomplete; obtain the manifest from the published release.")
    root = args.repo_root.resolve()
    cache = root / ".data-download"
    total = sum(archive["file_count"] for archive in archives)
    failed, verified = [], 0
    for index, archive in enumerate(archives, 1):
        invalid = [record for record in archive["files"] if not valid_file(root, record)]
        if args.check_only:
            failed.extend(record["path"] for record in invalid)
            verified += len(archive["files"]) - len(invalid)
            print(f"Checked [{index}/{len(archives)}] {archive['name']}: {len(archive['files'])-len(invalid)}/{len(archive['files'])} valid", flush=True)
            continue
        if not invalid:
            verified += len(archive["files"])
            print(f"Already verified [{index}/{len(archives)}] {archive['name']}", flush=True)
            continue
        print(f"Downloading [{index}/{len(archives)}] {archive['name']} ({archive['size_bytes']/1024**2:.1f} MiB)", flush=True)
        cache.mkdir(parents=True, exist_ok=True)
        local = safe_target(cache, archive["name"])
        download(args.release_url.rstrip("/") + "/" + archive["name"], local, archive)
        extract_verified(local, root, archive)
        verified += len(archive["files"])
        if not args.keep_archives:
            local.unlink()
        print(f"Verified and extracted {verified}/{total} HDF5 files", flush=True)
    print(f"Verified {verified}/{total} HDF5 files.", flush=True)
    if failed:
        for name in failed[:20]:
            print(f"Missing or checksum mismatch: {name}", file=sys.stderr)
        if len(failed) > 20:
            print(f"... and {len(failed)-20} additional invalid files", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
