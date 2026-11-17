"""Fetch the official Gotham CSVs using ZIP ranges, verifying every member's CRC."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import struct
import zipfile
import zlib

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RECORD = "https://zenodo.org/api/records/14502760"


def session():
    client = requests.Session()
    client.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    return client


def get_range(client, url, start, end, stream=False):
    result = client.get(url, headers={"Range": f"bytes={start}-{end}"},
                        timeout=(30, 180), stream=stream)
    result.raise_for_status()
    if result.status_code != 206 or not result.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
        result.close()
        raise RuntimeError("The server did not return the requested ZIP byte range")
    return result


class RemoteZip:
    def __init__(self, client, url, size):
        self.client, self.url, self.size, self.position = client, url, size, 0

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        result = get_range(self.client, self.url, self.position, self.position + size - 1).content
        self.position += len(result)
        return result


def digest_file(path):
    sha, crc = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            sha.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return sha.hexdigest(), crc & 0xffffffff


def fetch_member(url, member, destination):
    path = destination / member["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == member["size"]:
        sha, crc = digest_file(path)
        if crc == member["crc"]:
            return {**member, "sha256": sha, "crc32_verified": True}
    with session() as client:
        offset = member["offset"]
        header = get_range(client, url, offset, offset + 29).content
        values = struct.unpack("<4s5H3I2H", header)
        if values[0] != b"PK\x03\x04" or values[3] != member["method"]:
            raise ValueError(f"Invalid ZIP local header: {member['name']}")
        start = offset + 30 + values[-2] + values[-1]
        inflater = zlib.decompressobj(-zlib.MAX_WBITS) if member["method"] == 8 else None
        if member["method"] not in {0, 8}:
            raise ValueError("Unsupported ZIP compression")
        sha, crc, size, received = hashlib.sha256(), 0, 0, 0
        part = path.with_name(path.name + ".part")
        with get_range(client, url, start, start + member["compressed"] - 1, stream=True) as response:
            with part.open("wb") as stream:
                for block in response.iter_content(1024 ** 2):
                    received += len(block)
                    decoded = inflater.decompress(block) if inflater else block
                    stream.write(decoded)
                    sha.update(decoded)
                    crc = zlib.crc32(decoded, crc)
                    size += len(decoded)
                if inflater:
                    tail = inflater.flush()
                    stream.write(tail)
                    sha.update(tail)
                    crc = zlib.crc32(tail, crc)
                    size += len(tail)
                    if not inflater.eof:
                        raise ValueError(f"Truncated deflate stream: {member['name']}")
        if received != member["compressed"] or size != member["size"] or crc & 0xffffffff != member["crc"]:
            raise ValueError(f"ZIP size/CRC mismatch: {member['name']}")
        os.replace(part, path)
        return {**member, "sha256": sha.hexdigest(), "crc32_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("/data/SDP_Dataset/Gotham"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with session() as client:
        response = client.get(RECORD, timeout=60)
        response.raise_for_status()
        record = response.json()
        (args.destination / "zenodo_record.json").write_text(json.dumps(record, indent=2) + "\n")
        source = record["files"][0]
        url = source["links"]["self"]
        with zipfile.ZipFile(RemoteZip(client, url, source["size"])) as archive:
            members = [{"name": i.filename, "offset": i.header_offset, "compressed": i.compress_size,
                        "size": i.file_size, "crc": i.CRC, "method": i.compress_type}
                       for i in archive.infolist()]
    (args.destination / "zip_members.json").write_text(json.dumps(members, indent=2) + "\n")
    selected = [m for m in members if m["name"] == "README.md" or
                (m["name"].startswith("processed/") and m["name"].endswith(".csv"))]
    print(f"Fetching {len(selected)} ZIP members ({sum(m['compressed'] for m in selected):,} compressed bytes)", flush=True)
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_member, url, member, args.destination / "raw") for member in selected]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f"[{len(completed)}/{len(selected)}] CRC verified: {result['name']}", flush=True)
    manifest = {"record": RECORD, "zip_url": url, "zip_size": source["size"],
                "published_archive_checksum": source["checksum"], "whole_archive_downloaded": False,
                "verification": "Official ZIP central directory and per-member CRC32; saved CSV SHA256",
                "members": sorted(completed, key=lambda m: m["name"])}
    (args.destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded and verified {len(completed)} official members to {args.destination / 'raw'}", flush=True)


if __name__ == "__main__":
    main()
